#include "racing/vehicle.hpp"

#include <algorithm>
#include <cmath>

namespace racing {

namespace {
constexpr double kG = 9.81;

// Longitudinal slip stiffness per newton of load, 1/unit-slip. Puts peak
// longitudinal force near 9% slip ratio, which is about right for a slick.
constexpr double kSlipStiffnessPerN = 20.0;

// How much force is LOST past the peak of the slip curve, at 100% over-demand.
// A tire asked for more than it has does not simply give its maximum -- it goes
// past the peak and gives less. That is what wheelspin and a locked wheel are,
// and without it over-driving is free.
constexpr double kSlipFalloff = 0.18;
}  // namespace

double VehicleState::speed() const { return std::sqrt(vx * vx + vy * vy); }

Vehicle::Vehicle(const VehicleParams& p) : p_(p) {
  // Static front axle load is m*g*l_r/L, so the front weight fraction fixes
  // l_r directly. Getting this backwards puts the CG on the wrong side of the
  // car and quietly inverts its handling balance.
  lr_ = p_.front_weight_frac * p_.wheelbase;
  lf_ = p_.wheelbase - lr_;
}

double Vehicle::mu_at(double fz) const {
  // Tires lose grip as they are loaded harder. Modelled as a power law against
  // a reference load, which is the usual first-order treatment.
  if (fz <= 1.0) return p_.mu_peak;
  return p_.mu_peak * std::pow(fz / p_.mu_ref_load, -p_.mu_load_sensitivity);
}

double Vehicle::axle_grip(double fz_axle, double lateral_transfer) const {
  // Split the axle load across its two wheels and add up what each can make.
  //
  // This is where lateral load transfer costs grip, and it costs it because mu
  // is sub-linear: mu(F/2 + d)*(F/2 + d) + mu(F/2 - d)*(F/2 - d) is always less
  // than 2*mu(F/2)*(F/2) for d > 0. Averaging the load first and calling mu
  // once -- which is what a bicycle model normally does -- throws that away and
  // makes the car corner identically whatever its roll distribution is.
  const double half = 0.5 * std::max(fz_axle, 0.0);
  const double d = std::abs(lateral_transfer);
  const double outer = half + d;
  // The inside wheel can be fully unloaded; it cannot be pulled off the ground
  // and start generating negative grip.
  const double inner = std::max(half - d, 0.0);
  return mu_at(outer) * outer + mu_at(inner) * inner;
}

double Vehicle::downforce_at(double v) const {
  return 0.5 * p_.air_density * p_.cl_a * v * v;
}

double Vehicle::drag_at(double v) const {
  return 0.5 * p_.air_density * p_.cd_a * v * v;
}

Vehicle::Deriv Vehicle::derivative(const VehicleState& s, const VehicleInput& u,
                                   VehicleTelemetry* out) const {
  // Fuel is carried as mass, so a full car is genuinely harder to stop and
  // turn, not merely slower in a straight line.
  const double m = p_.mass + std::max(u.extra_mass, 0.0);
  const double L = p_.wheelbase;
  const double delta = std::clamp(u.steer, -1.0, 1.0) * p_.max_steer;
  const double v = std::sqrt(s.vx * s.vx + s.vy * s.vy);

  // --- aero -------------------------------------------------------------
  // Aerodynamic forces go with AIRSPEED, not ground speed, so a headwind adds
  // downforce as well as drag. The two scales carry the wake of any car ahead
  // and the state of the rear wing: a tow removes drag, dirty air removes
  // downforce, DRS removes both.
  const double rho = u.air_density > 0.0 ? u.air_density : p_.air_density;
  const double rho_scale = rho / std::max(p_.air_density, 1e-6);
  const double v_air = std::max(0.0, v + u.headwind);
  const double q = v_air * v_air / std::max(v * v, 1e-6);  // airspeed^2 / v^2

  const double downforce = downforce_at(v) * q * rho_scale * u.downforce_scale;
  const double drag = drag_at(v) * q * rho_scale * u.drag_scale;

  // --- normal loads -----------------------------------------------------
  // Static split, plus the aero balance, plus longitudinal transfer using the
  // acceleration carried from the previous step.
  const double transfer = m * s.ax * p_.cg_height / L;
  double fz_f = m * kG * p_.front_weight_frac +
                downforce * p_.aero_balance_front - transfer;
  double fz_r = m * kG * (1.0 - p_.front_weight_frac) +
                downforce * (1.0 - p_.aero_balance_front) + transfer;
  fz_f = std::max(fz_f, 0.0);
  fz_r = std::max(fz_r, 0.0);

  // Lateral transfer, again from the acceleration carried across the step.
  // Total is m*ay*h/t; the roll stiffness distribution decides which axle does
  // more of it, which is the classic balance adjustment: stiffen one end and it
  // transfers more, loses more grip, and the car pushes away from it.
  const double lat_total = m * s.ay * p_.cg_height / std::max(p_.track_width, 1e-3);
  const double lat_f = lat_total * p_.roll_stiffness_front;
  const double lat_r = lat_total * (1.0 - p_.roll_stiffness_front);

  // --- grip available ----------------------------------------------------
  // `grip_scale` is the surface; `grip_front`/`grip_rear` are the tires.
  const double kf = std::max(0.0, u.grip_scale * u.grip_front);
  const double kr = std::max(0.0, u.grip_scale * u.grip_rear);
  const double grip_f = axle_grip(fz_f, lat_f) * kf;
  const double grip_r = axle_grip(fz_r, lat_r) * kr;

  // Effective per-axle mu, for telemetry and for the ellipse below.
  const double mu_f = fz_f > 1.0 ? grip_f / fz_f : p_.mu_peak * kf;
  const double mu_r = fz_r > 1.0 ? grip_r / fz_r : p_.mu_peak * kr;

  // Apply the lateral ceiling by sharing it out in proportion to axle load, so
  // the cap cannot silently alter the car's balance the way a flat per-axle
  // limit would.
  const double fz_tot = std::max(fz_f + fz_r, 1.0);
  const double lat_cap = p_.max_lateral_g * m * kG;
  const double fy_f_cap = std::max(std::min(grip_f, lat_cap * fz_f / fz_tot), 1.0);
  const double fy_r_cap = std::max(std::min(grip_r, lat_cap * fz_r / fz_tot), 1.0);

  // --- longitudinal demand ----------------------------------------------
  //
  // Both pedals are interpreted as a fraction of what is available *now*,
  // rather than as a fraction of a fixed maximum force. That is not a
  // convenience: with a fixed 55 kN brake and roughly 20 kN of grip, every
  // command past about -0.36 locked all four wheels, so the usable range of the
  // pedal was a third of its travel and everything beyond it was an identical
  // full lock-up. Worse, a locked axle has no lateral force left at all, so the
  // rear would let go under any brake application deep enough to matter and the
  // car simply spun at the first hard corner.
  double fx_f = 0.0, fx_r = 0.0;
  double demand_f = 0.0, demand_r = 0.0;
  const double thr = std::clamp(u.throttle, -1.0, 1.0);
  bool driving = false;

  if (thr >= 0.0) {
    driving = true;
    // Power-limited above a few m/s, traction-limited below it. With a
    // powertrain attached, `drive_force` already accounts for the gear the car
    // is in -- which is what makes a hairpin exit traction-limited and the end
    // of a straight power-limited.
    const double f_available = u.drive_force >= 0.0
                                   ? u.drive_force
                                   : p_.max_power / std::max(v, 5.0);
    const double drive = thr * f_available;
    demand_r = drive * p_.drivetrain_rear_frac;
    demand_f = drive * (1.0 - p_.drivetrain_rear_frac);
  } else {
    // Split in proportion to each axle's grip, which is what a correctly
    // balanced brake system does: both axles then reach their limit together
    // instead of one locking while the other is still working.
    const double total = std::max(grip_f + grip_r, 1.0);
    // The pedal can ask for a little MORE than the tyres have. Without that
    // headroom a driver physically cannot lock a wheel -- demand is defined as
    // a fraction of available grip, so it can never exceed it, and the lockup
    // the telemetry reports would be dead code. Real drivers lock wheels; the
    // last 20% of the pedal is where they do it.
    constexpr double kBrakeOverRange = 1.2;
    const double demand =
        std::min(-thr * total * kBrakeOverRange, p_.max_brake_force);
    demand_f = -demand * grip_f / total;
    demand_r = -demand * grip_r / total;
  }

  // The slip curve past its peak. A tire asked for more than it has does not
  // hold at maximum -- it slips more and gives LESS. Over-demand of 100% costs
  // about a fifth of the force, and the extra sliding is what heats and wears
  // the tire.
  auto apply_slip = [&](double demand, double cap, double* slip_ratio,
                        bool* saturated) {
    const double mag = std::abs(demand);
    const double limit = std::max(cap, 1.0);
    const double over = std::max(0.0, mag - limit) / limit;
    *saturated = over > 0.0;
    const double delivered =
        std::min(mag, limit) * (1.0 - kSlipFalloff * std::min(over, 1.0));
    // Slip ratio implied by the force being asked for, growing sharply once the
    // tire is past its peak.
    *slip_ratio = delivered / std::max(kSlipStiffnessPerN * limit, 1.0) *
                  (1.0 + 4.0 * over);
    return std::copysign(delivered, demand);
  };

  double slip_f = 0.0, slip_r = 0.0;
  bool over_f = false, over_r = false;
  fx_f = apply_slip(demand_f, grip_f, &slip_f, &over_f);
  fx_r = apply_slip(demand_r, grip_r, &slip_r, &over_r);

  // Rolling resistance always opposes motion.
  const double roll = p_.rolling_resistance * (fz_f + fz_r);
  const double roll_signed = v > 0.1 ? -roll : 0.0;

  // Gravity along the road. Small at Bahrain, but it is the reason a car is
  // slower up the hill out of the last corner than the flat-road model says.
  const double gravity_x = -m * kG * u.grade;

  // --- slip angles ------------------------------------------------------
  // Guard the denominator: below walking pace this ratio is meaningless, and
  // the kinematic blend further down is what actually carries low-speed
  // behaviour.
  const double vx_safe = std::max(std::abs(s.vx), 1.0);
  const double alpha_f = delta - std::atan2(s.vy + lf_ * s.r, vx_safe);
  const double alpha_r = -std::atan2(s.vy - lr_ * s.r, vx_safe);

  // --- lateral forces: linear, then the friction ellipse -----------------
  double fy_f = p_.cornering_stiffness_front_per_n * fz_f * alpha_f * kf;
  double fy_r = p_.cornering_stiffness_rear_per_n * fz_r * alpha_r * kr;

  // A tire spending capacity longitudinally has less left for cornering. The
  // longitudinal force is already clamped above, so this gives the lateral
  // direction whatever the ellipse leaves.
  bool sat_f = over_f, sat_r = over_r;
  {
    const double cap = fy_f_cap;
    const double used = std::min(std::abs(fx_f) / cap, 1.0);
    const double fy_lim = cap * std::sqrt(std::max(0.0, 1.0 - used * used));
    if (std::abs(fy_f) > fy_lim) {
      fy_f = std::copysign(fy_lim, fy_f);
      sat_f = true;
    }
  }
  {
    const double cap = fy_r_cap;
    const double used = std::min(std::abs(fx_r) / cap, 1.0);
    const double fy_lim = cap * std::sqrt(std::max(0.0, 1.0 - used * used));
    if (std::abs(fy_r) > fy_lim) {
      fy_r = std::copysign(fy_lim, fy_r);
      sat_r = true;
    }
  }

  if (out) {
    out->fz_front = fz_f;
    out->fz_rear = fz_r;
    out->fz_fl = std::max(0.5 * fz_f - lat_f, 0.0);
    out->fz_fr = std::max(0.5 * fz_f + lat_f, 0.0);
    out->fz_rl = std::max(0.5 * fz_r - lat_r, 0.0);
    out->fz_rr = std::max(0.5 * fz_r + lat_r, 0.0);
    out->alpha_front = alpha_f;
    out->alpha_rear = alpha_r;
    out->fy_front = fy_f;
    out->fy_rear = fy_r;
    out->fx_front = fx_f;
    out->fx_rear = fx_r;
    out->downforce = downforce;
    out->drag = drag;
    out->mu_front = mu_f;
    out->mu_rear = mu_r;
    out->front_saturated = sat_f;
    out->rear_saturated = sat_r;
    out->slip_ratio = driving ? slip_r : std::max(slip_f, slip_r);
    out->wheelspin = driving && over_r;
    out->lockup = !driving && (over_f || over_r);
    out->mass = m;

    // Frictional work at the contact patch: force times the speed the rubber is
    // sliding at. Lateral slip velocity is v*sin(alpha); longitudinal is the
    // slip ratio times road speed. This is the only quantity the tire model
    // needs, and it is why a car that is sliding cooks its tires.
    const double lat_slip_f = std::abs(v * std::sin(alpha_f));
    const double lat_slip_r = std::abs(v * std::sin(alpha_r));
    out->slip_power_front =
        std::abs(fy_f) * lat_slip_f + std::abs(fx_f) * slip_f * v;
    out->slip_power_rear =
        std::abs(fy_r) * lat_slip_r + std::abs(fx_r) * slip_r * v;
  }

  // --- rigid body -------------------------------------------------------
  const double cd = std::cos(delta), sd = std::sin(delta);
  const double fx_body =
      fx_r + fx_f * cd - fy_f * sd - drag + roll_signed + gravity_x;
  const double fy_body = fy_r + fy_f * cd + fx_f * sd;

  Deriv d;
  d.dvx = fx_body / m + s.vy * s.r;
  d.dvy = fy_body / m - s.vx * s.r;
  d.dr = (lf_ * (fy_f * cd + fx_f * sd) - lr_ * fy_r) / p_.yaw_inertia;

  if (out) {
    out->lateral_g = fy_body / (m * kG);
    out->longitudinal_g = fx_body / (m * kG);
  }

  // --- low-speed blend ---------------------------------------------------
  // The slip-angle terms above divide by longitudinal velocity, so as the car
  // slows they stop meaning anything: an infinitesimal sideways drift reads as
  // an enormous slip angle and the tire forces blow up. Below `blend_speed_hi`
  // the lateral state is relaxed toward what simple kinematics says a car with
  // this steering angle must be doing, which has no such singularity.
  if (v < p_.blend_speed_hi) {
    const double lambda = std::clamp(
        (v - p_.blend_speed_lo) / (p_.blend_speed_hi - p_.blend_speed_lo),
        0.0, 1.0);
    const double r_kin = s.vx * std::tan(delta) / L;
    const double vy_kin = lr_ * r_kin;
    constexpr double kRelax = 0.05;  // s
    d.dr = lambda * d.dr + (1.0 - lambda) * (r_kin - s.r) / kRelax;
    d.dvy = lambda * d.dvy + (1.0 - lambda) * (vy_kin - s.vy) / kRelax;
  }

  const double cp = std::cos(s.psi), sp = std::sin(s.psi);
  d.dx = s.vx * cp - s.vy * sp;
  d.dy = s.vx * sp + s.vy * cp;
  d.dpsi = s.r;
  return d;
}

VehicleTelemetry Vehicle::telemetry(const VehicleState& s,
                                    const VehicleInput& u) const {
  VehicleTelemetry t;
  derivative(s, u, &t);
  return t;
}

void Vehicle::step(VehicleState* s, const VehicleInput& u, double dt,
                   VehicleTelemetry* out) const {
  const VehicleState s0 = *s;

  auto advance = [&](const VehicleState& base, const Deriv& d, double h) {
    VehicleState o = base;
    o.x = s0.x + h * d.dx;
    o.y = s0.y + h * d.dy;
    o.psi = s0.psi + h * d.dpsi;
    o.vx = s0.vx + h * d.dvx;
    o.vy = s0.vy + h * d.dvy;
    o.r = s0.r + h * d.dr;
    return o;
  };

  const Deriv k1 = derivative(s0, u, out);
  const Deriv k2 = derivative(advance(s0, k1, 0.5 * dt), u, nullptr);
  const Deriv k3 = derivative(advance(s0, k2, 0.5 * dt), u, nullptr);
  const Deriv k4 = derivative(advance(s0, k3, dt), u, nullptr);

  const double h = dt / 6.0;
  s->x = s0.x + h * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
  s->y = s0.y + h * (k1.dy + 2.0 * k2.dy + 2.0 * k3.dy + k4.dy);
  s->psi = s0.psi + h * (k1.dpsi + 2.0 * k2.dpsi + 2.0 * k3.dpsi + k4.dpsi);
  s->vx = s0.vx + h * (k1.dvx + 2.0 * k2.dvx + 2.0 * k3.dvx + k4.dvx);
  s->vy = s0.vy + h * (k1.dvy + 2.0 * k2.dvy + 2.0 * k3.dvy + k4.dvy);
  s->r = s0.r + h * (k1.dr + 2.0 * k2.dr + 2.0 * k3.dr + k4.dr);

  // A car cannot be driven backwards here, and letting vx cross zero would put
  // the slip-angle terms into a regime the model does not represent.
  s->vx = std::max(s->vx, 0.0);

  // Accelerations produced by the tire and aero *forces*, carried into the next
  // step's load transfer.
  //
  // These are not dvx/dt and dvy/dt. The body-frame equations carry rotating
  // frame terms -- dvx/dt = fx/m + vy*r and dvy/dt = fy/m - vx*r -- and load
  // transfer is caused by force, not by the velocity vector swinging round
  // inside the body frame. Using dvx/dt directly was a genuine bug with a
  // spectacular failure mode: as soon as the car developed any sideslip, the
  // vy*r term read as enormous braking (-99 m/s^2 of a measured -98.6 at 45
  // degrees of slip), transferred every newton off the rear axle, and the rear
  // lost all grip. That fed back into more sideslip, so the car spun itself
  // from any moderate steering input above about 180 kph -- which looked for a
  // while like an unlearnable environment.
  s->ax = (s->vx - s0.vx) / dt - s0.vy * s0.r;
  s->ay = (s->vy - s0.vy) / dt + s0.vx * s0.r;
}

// --- steady-state limits ---------------------------------------------------

double Vehicle::max_corner_speed(double kappa) const {
  const double k = std::abs(kappa);
  if (k < 1e-6) return 150.0;  // effectively straight; capped by drag elsewhere

  // Balance required lateral force m*v^2*k against what the tires can give at
  // that speed. Grip rises with v^2 through downforce while demand also rises
  // with v^2, so the two do not simply cancel: load sensitivity means grip
  // grows slightly slower than linearly in load, which is what makes a finite
  // solution exist. Bisection is used because there is no closed form.
  auto surplus = [&](double v) {
    const double df = downforce_at(v);
    const double fz_f = p_.mass * kG * p_.front_weight_frac +
                        df * p_.aero_balance_front;
    const double fz_r = p_.mass * kG * (1.0 - p_.front_weight_frac) +
                        df * (1.0 - p_.aero_balance_front);

    // Lateral load transfer depends on the lateral acceleration, which is what
    // we are solving for -- so iterate. Two passes is plenty: the correction is
    // a couple of percent and the second pass moves it by a fraction of that.
    double ay = 0.0;
    double grip = 0.0;
    for (int i = 0; i < 3; ++i) {
      const double lat = p_.mass * ay * p_.cg_height / std::max(p_.track_width, 1e-3);
      grip = axle_grip(fz_f, lat * p_.roll_stiffness_front) +
             axle_grip(fz_r, lat * (1.0 - p_.roll_stiffness_front));
      grip = std::min(grip, p_.max_lateral_g * p_.mass * kG);
      ay = grip / p_.mass;
    }
    return grip - p_.mass * v * v * k;
  };

  double lo = 0.1, hi = 150.0;
  if (surplus(hi) > 0.0) return hi;
  for (int i = 0; i < 60; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (surplus(mid) > 0.0) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

double Vehicle::max_long_accel(double v, double lat_accel) const {
  const double df = downforce_at(v);
  const double fz_r = p_.mass * kG * (1.0 - p_.front_weight_frac) +
                      df * (1.0 - p_.aero_balance_front);
  // Only the driven axle can put power down, and it is already carrying the
  // lateral transfer that this cornering implies.
  const double lat = p_.mass * lat_accel * p_.cg_height /
                     std::max(p_.track_width, 1e-3);
  const double grip_r = axle_grip(fz_r, lat * (1.0 - p_.roll_stiffness_front));

  // Whatever lateral force is already being used comes out of the same ellipse.
  const double fy_used = p_.mass * std::abs(lat_accel) *
                         (1.0 - p_.front_weight_frac);
  const double left = std::sqrt(std::max(
      0.0, 1.0 - (fy_used / std::max(grip_r, 1.0)) * (fy_used / std::max(grip_r, 1.0))));
  const double f_traction = grip_r * left;

  const double f_power = p_.max_power / std::max(v, 5.0);
  const double f_drive = std::min(f_traction, f_power);
  const double resist = drag_at(v) + p_.rolling_resistance *
                                         (p_.mass * kG + df);
  return (f_drive - resist) / p_.mass;
}

double Vehicle::max_long_decel(double v, double lat_accel) const {
  const double df = downforce_at(v);
  const double fz_f = p_.mass * kG * p_.front_weight_frac +
                      df * p_.aero_balance_front;
  const double fz_r = p_.mass * kG * (1.0 - p_.front_weight_frac) +
                      df * (1.0 - p_.aero_balance_front);
  // Braking uses every tire, unlike acceleration.
  const double lat = p_.mass * lat_accel * p_.cg_height /
                     std::max(p_.track_width, 1e-3);
  const double grip = axle_grip(fz_f, lat * p_.roll_stiffness_front) +
                      axle_grip(fz_r, lat * (1.0 - p_.roll_stiffness_front));
  const double fy_used = p_.mass * std::abs(lat_accel);
  const double left = std::sqrt(std::max(
      0.0, 1.0 - (fy_used / std::max(grip, 1.0)) * (fy_used / std::max(grip, 1.0))));
  const double f_brake = std::min(grip * left, p_.max_brake_force);

  // Drag helps a great deal at speed -- worth tens of metres of braking
  // distance at the end of a long straight.
  const double resist = drag_at(v) + p_.rolling_resistance *
                                         (p_.mass * kG + df);
  return -(f_brake + resist) / p_.mass;
}

}  // namespace racing
