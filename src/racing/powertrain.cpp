#include "racing/powertrain.hpp"

#include <algorithm>
#include <cmath>

namespace racing {

namespace {
constexpr int kGears = 8;
constexpr int kCurvePoints = 7;
constexpr double kRpmToRad = 3.141592653589793 / 30.0;
}  // namespace

Powertrain::Powertrain(const PowertrainConfig& cfg, const VehicleParams& vp)
    : cfg_(cfg), vp_(vp) {
  // Scale the torque curve so that its PEAK POWER equals the fitted
  // `max_power`, rather than making peak torque equal to something arbitrary.
  //
  // That matters because max_power is the number the calibration identified
  // against a real top speed, and top speed is set by power against drag. Peak
  // torque is then whatever it has to be for the curve to hit that power.
  double best = 0.0;
  for (int i = 0; i < 64; ++i) {
    const double rpm = cfg_.idle_rpm +
                       (cfg_.redline_rpm - cfg_.idle_rpm) * i / 63.0;
    best = std::max(best, torque_fraction(rpm) * rpm * kRpmToRad);
  }
  peak_torque_ = best > 0.0 ? vp_.max_power / best : 0.0;
}

double Powertrain::torque_fraction(double rpm) const {
  const double lo = cfg_.idle_rpm, hi = cfg_.redline_rpm;
  const double u = std::clamp((rpm - lo) / std::max(hi - lo, 1.0), 0.0, 1.0) *
                   (kCurvePoints - 1);
  const int i = std::min(static_cast<int>(u), kCurvePoints - 2);
  const double f = u - i;
  return cfg_.torque_curve[i] * (1.0 - f) + cfg_.torque_curve[i + 1] * f;
}

void Powertrain::reset(PowertrainState* s) const {
  *s = PowertrainState{};
  s->ers_charge_mj = cfg_.ers_capacity_mj * std::clamp(cfg_.ers_start_charge, 0.0, 1.0);
}

double Powertrain::rpm_at(double speed, int gear) const {
  gear = std::clamp(gear, 0, kGears - 1);
  const double wheel_rad_s = std::max(speed, 0.0) / std::max(vp_.wheel_radius, 1e-3);
  const double engine_rad_s = wheel_rad_s * cfg_.gear_ratios[gear] * cfg_.final_drive;
  return std::clamp(engine_rad_s / kRpmToRad, cfg_.idle_rpm, cfg_.redline_rpm * 1.02);
}

int Powertrain::select_gear(double speed, int current) const {
  current = std::clamp(current, 0, kGears - 1);
  // Hysteresis: the up and down thresholds are deliberately far apart, or the
  // gearbox hunts either side of a shift point and the torque cut fires every
  // step.
  if (current < kGears - 1 &&
      rpm_at(speed, current) > cfg_.redline_rpm * cfg_.shift_up_frac) {
    return current + 1;
  }
  if (current > 0 &&
      rpm_at(speed, current) < cfg_.redline_rpm * cfg_.shift_down_frac) {
    return current - 1;
  }
  return current;
}

void Powertrain::new_lap(PowertrainState* s) const {
  s->ers_deployed_lap_mj = 0.0;
}

void Powertrain::update(PowertrainState* s, double speed, double throttle,
                        double braking_power_w, double dt) const {
  if (!cfg_.enabled) {
    s->rpm = 0.0;
    s->ice_power_w = vp_.max_power;
    s->ers_power_w = 0.0;
    return;
  }

  // --- gearbox -------------------------------------------------------------
  s->shift_timer = std::max(0.0, s->shift_timer - dt);
  const int wanted = select_gear(speed, s->gear);
  if (wanted != s->gear && s->shift_timer <= 0.0) {
    s->gear = wanted;
    s->shift_timer = cfg_.shift_time_s;
  }
  s->rpm = rpm_at(speed, s->gear);

  // --- engine --------------------------------------------------------------
  // Torque is cut mid-shift, which is the small dip in the acceleration trace.
  const double cut = s->shift_timer > 0.0 ? 0.0 : 1.0;
  const double engine_torque = peak_torque_ * torque_fraction(s->rpm) * cut *
                               std::clamp(throttle, 0.0, 1.0);
  s->ice_power_w = engine_torque * s->rpm * kRpmToRad;

  // --- hybrid --------------------------------------------------------------
  // Harvest whenever the brakes are doing work, capped by the recovery rate and
  // by how much room is left in the store.
  if (braking_power_w > 0.0) {
    const double rate_w = std::min(braking_power_w, cfg_.ers_harvest_kw * 1000.0);
    const double room = std::max(0.0, cfg_.ers_capacity_mj - s->ers_charge_mj);
    s->ers_charge_mj = std::min(cfg_.ers_capacity_mj,
                                s->ers_charge_mj + std::min(rate_w * dt * 1e-6, room));
  }

  // Deploy on a committed throttle, above the speed where it is worth having,
  // while there is charge and lap allowance left. A real driver or a strategy
  // model would be cleverer about *where*; see the note in powertrain.hpp.
  const double lap_left = cfg_.ers_max_deploy_per_lap_mj - s->ers_deployed_lap_mj;
  s->deploying = throttle > 0.55 && speed > 25.0 && s->ers_charge_mj > 1e-6 &&
                 lap_left > 1e-6 && s->shift_timer <= 0.0;
  if (s->deploying) {
    const double want_mj = cfg_.ers_deploy_kw * 1000.0 * dt * 1e-6;
    const double spend = std::min({want_mj, s->ers_charge_mj, lap_left});
    s->ers_charge_mj -= spend;
    s->ers_deployed_lap_mj += spend;
    s->ers_power_w = spend * 1e6 / std::max(dt, 1e-9);
  } else {
    s->ers_power_w = 0.0;
  }

  s->drive_force_n = drive_force(*s, speed);
}

double Powertrain::drive_force(const PowertrainState& s, double speed) const {
  if (!cfg_.enabled) {
    return vp_.max_power / std::max(speed, 5.0);
  }
  // Engine force through the gearing. This is what makes a car out of a hairpin
  // traction-limited rather than power-limited: in first gear the same engine
  // torque arrives at the contact patch twenty times over.
  const double ratio = cfg_.gear_ratios[std::clamp(s.gear, 0, kGears - 1)] *
                       cfg_.final_drive;
  const double engine_torque =
      s.rpm > 0.0 ? s.ice_power_w / std::max(s.rpm * kRpmToRad, 1e-6) : 0.0;
  const double from_ice = engine_torque * ratio / std::max(vp_.wheel_radius, 1e-3);

  // The hybrid is applied as power rather than through the gearbox, because the
  // MGU-K sits on the crankshaft and its contribution is flat in torque terms
  // across the range where it is deployed.
  const double from_ers = s.ers_power_w / std::max(speed, 5.0);
  return from_ice + from_ers;
}

}  // namespace racing
