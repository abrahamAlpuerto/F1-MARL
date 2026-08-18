// The physics that was added after the car could already drive: the air, the
// fuel, the tyres, load transfer, the powertrain and DRS.
//
// These are mostly behavioural rather than numerical. The question is not "is
// this constant right" -- the calibration answers that -- but "does the effect
// exist, does it point the right way, and is it bounded". Every one of the bugs
// found while building this passed a numerical check and failed one of those.

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch.hpp"
#include "racing/atmosphere.hpp"
#include "racing/config.hpp"
#include "racing/interaction.hpp"
#include "racing/powertrain.hpp"
#include "racing/race.hpp"
#include "racing/tyres.hpp"
#include "racing/vehicle.hpp"

using namespace racing;

namespace {

std::string track_path() {
  return std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
}

EnvConfig one_car() {
  EnvConfig c;
  c.track.path = track_path();
  c.field.n_teams = 1;
  c.field.cars_per_team = 1;
  c.track.randomize_start = false;
  c.track.episode_distance = 2000.0;
  c.seed = 3;
  return c;
}

std::unique_ptr<RaceEnv> make(const EnvConfig& c) {
  auto track = std::make_shared<const Track>(Track::load(c.track.path));
  return std::make_unique<RaceEnv>(c, track, 0);
}

}  // namespace

// --- the air ---------------------------------------------------------------

TEST_CASE("air density follows temperature, pressure and humidity",
          "[racing][atmosphere]") {
  AtmosphereConfig a;  // the calibration conditions
  REQUIRE(air_density(a) == Approx(1.1918).margin(0.0005));

  // Hot air is thinner.
  AtmosphereConfig hot = a;
  hot.air_temperature_c = 40.0;
  REQUIRE(air_density(hot) < air_density(a));

  // Altitude -- lower pressure -- is thinner still.
  AtmosphereConfig high = a;
  high.pressure_pa = 84000.0;  // roughly Mexico City
  REQUIRE(air_density(high) < 0.9 * air_density(a));

  // And humid air is LESS dense than dry, which is the counter-intuitive one:
  // water vapour is lighter than the nitrogen it displaces.
  AtmosphereConfig damp = a;
  damp.humidity = 1.0;
  AtmosphereConfig dry = a;
  dry.humidity = 0.0;
  REQUIRE(air_density(damp) < air_density(dry));
}

TEST_CASE("wind resolves to a headwind or a tailwind by heading",
          "[racing][atmosphere]") {
  AtmosphereConfig a;
  a.wind_speed = 10.0;
  a.wind_direction = 0.0;  // blowing towards +x

  // Driving into it is a headwind; with it, a tailwind; across it, neither.
  REQUIRE(headwind_component(a, 3.141592653589793) == Approx(10.0));
  REQUIRE(headwind_component(a, 0.0) == Approx(-10.0));
  REQUIRE(headwind_component(a, 1.5707963267948966) == Approx(0.0).margin(1e-9));
}

TEST_CASE("thinner air means less drag and less downforce", "[racing][atmosphere]") {
  Vehicle v{VehicleParams{}};
  VehicleState s;
  s.vx = 80.0;

  VehicleInput dense;
  dense.throttle = 1.0;
  dense.air_density = 1.20;
  VehicleInput thin = dense;
  thin.air_density = 0.90;

  const VehicleTelemetry a = v.telemetry(s, dense);
  const VehicleTelemetry b = v.telemetry(s, thin);
  REQUIRE(b.drag < a.drag);
  REQUIRE(b.downforce < a.downforce);
  REQUIRE(b.drag / a.drag == Approx(0.90 / 1.20).epsilon(0.01));
}

// --- fuel ------------------------------------------------------------------

TEST_CASE("fuel is carried as mass and burns off", "[racing][fuel]") {
  Vehicle v{VehicleParams{}};
  VehicleState s;
  s.vx = 60.0;

  VehicleInput empty;
  empty.throttle = 1.0;
  VehicleInput full = empty;
  full.extra_mass = 100.0;

  REQUIRE(v.telemetry(s, full).mass ==
          Approx(v.telemetry(s, empty).mass + 100.0));

  // A heavy car has more load on its tyres...
  REQUIRE(v.telemetry(s, full).fz_front > v.telemetry(s, empty).fz_front);

  // ...and accelerates less, which is the point.
  VehicleState heavy = s, light = s;
  for (int i = 0; i < 50; ++i) {
    v.step(&heavy, full, 0.01);
    v.step(&light, empty, 0.01);
  }
  REQUIRE(heavy.vx < light.vx);

  // And over a race it burns away.
  EnvConfig c = one_car();
  auto env = make(c);
  env->reset(0);
  const double start = env->cars()[0].fuel_kg;
  REQUIRE(start == Approx(c.fuel.start_kg));

  float act[2] = {0.0f, 0.4f};
  float rew[1];
  while (!env->done()) env->step(act, rew, nullptr);
  REQUIRE(env->cars()[0].fuel_kg < start);
  REQUIRE(env->cars()[0].fuel_kg > 0.0);
}

// --- gradient --------------------------------------------------------------

TEST_CASE("the circuit has a gradient and it costs the car", "[racing][gradient]") {
  const Track t = Track::load(track_path());
  double steepest = 0.0;
  for (double s = 0.0; s < t.length(); s += t.ds()) {
    steepest = std::max(steepest, std::abs(t.grade_at(s)));
  }
  // Bahrain is not flat, but it is not a hill climb either.
  REQUIRE(steepest > 0.01);
  REQUIRE(steepest < 0.20);

  Vehicle v{VehicleParams{}};
  VehicleState flat, uphill;
  flat.vx = uphill.vx = 50.0;
  VehicleInput level;
  level.throttle = 0.5;
  VehicleInput climb = level;
  climb.grade = 0.05;

  for (int i = 0; i < 100; ++i) {
    v.step(&flat, level, 0.01);
    v.step(&uphill, climb, 0.01);
  }
  REQUIRE(uphill.vx < flat.vx);
}

// --- lateral load transfer --------------------------------------------------

TEST_CASE("moving load across an axle costs it grip", "[racing][loadtransfer]") {
  Vehicle v{VehicleParams{}};
  const double fz = 8000.0;

  const double even = v.axle_grip(fz, 0.0);
  const double tilted = v.axle_grip(fz, 1500.0);
  const double extreme = v.axle_grip(fz, 4000.0);  // inside wheel off the ground

  // Grip is sub-linear in load, so splitting it unevenly always loses.
  REQUIRE(tilted < even);
  REQUIRE(extreme < tilted);
  // But not catastrophically: one wheel carrying everything still has grip.
  REQUIRE(extreme > 0.6 * even);

  // Direction of the transfer cannot matter.
  REQUIRE(v.axle_grip(fz, 1500.0) == Approx(v.axle_grip(fz, -1500.0)));

  // And an unloaded axle has nothing to give.
  REQUIRE(v.axle_grip(0.0, 0.0) == Approx(0.0));
}

TEST_CASE("mu_ref_load is a per-wheel reference", "[racing][loadtransfer]") {
  // With no transfer, an axle at load F must make the same grip a single
  // contact patch at F/2 makes, twice. Getting the reference wrong here is
  // silent: it simply scales every grip in the model, and it made the
  // quasi-steady-state lap four seconds quicker than the real one.
  VehicleParams p;
  Vehicle v{p};
  const double fz = 2.0 * p.mu_ref_load;
  const double expected = 2.0 * p.mu_peak * p.mu_ref_load;
  REQUIRE(v.axle_grip(fz, 0.0) == Approx(expected).epsilon(1e-9));
}

// --- tyres ------------------------------------------------------------------

TEST_CASE("tyre grip peaks inside the working window", "[racing][tyres]") {
  TyreConfig cfg;
  const TyreCompound& c = compound(cfg);

  TyreState at_optimum;
  at_optimum.temperature_c = c.optimum_c;
  TyreState cold;
  cold.temperature_c = c.optimum_c - 2.0 * c.window_c;
  TyreState hot;
  hot.temperature_c = c.optimum_c + 2.0 * c.window_c;

  REQUIRE(tyre_grip(cfg, at_optimum) > tyre_grip(cfg, cold));
  REQUIRE(tyre_grip(cfg, at_optimum) > tyre_grip(cfg, hot));
  REQUIRE(tyre_grip(cfg, at_optimum) == Approx(c.grip));

  // Bounded at both ends. An unbounded falloff is how a car that had one cold
  // corner ends up unable to generate the heat to recover.
  for (double t = -20.0; t < 250.0; t += 5.0) {
    TyreState s;
    s.temperature_c = t;
    REQUIRE(tyre_grip(cfg, s) > 0.5);
    REQUIRE(tyre_grip(cfg, s) <= c.grip + 1e-9);
  }
}

TEST_CASE("wear costs grip and is monotonic", "[racing][tyres]") {
  TyreConfig cfg;
  const TyreCompound& c = compound(cfg);
  double last = 1e9;
  for (double w = 0.0; w <= 1.0; w += 0.1) {
    TyreState s;
    s.temperature_c = c.optimum_c;
    s.wear = w;
    const double g = tyre_grip(cfg, s);
    REQUIRE(g <= last + 1e-12);
    last = g;
  }
  // At full wear the window has moved as well as the peak dropping -- a thin
  // tyre wants to run cooler -- so this has to be measured at the SHIFTED
  // optimum, not the fresh one.
  TyreState worn;
  worn.temperature_c = c.optimum_c + cfg.wear_optimum_shift_c;
  worn.wear = 1.0;
  REQUIRE(tyre_grip(cfg, worn) == Approx(c.grip * cfg.grip_worn));
}

TEST_CASE("a worked tyre heats up and an idle one cools down",
          "[racing][tyres]") {
  TyreConfig cfg;
  AtmosphereConfig air;

  TyreState hot_work;
  hot_work.temperature_c = 60.0;
  for (int i = 0; i < 500; ++i) {
    update_tyre(cfg, air, &hot_work, 40000.0, 60.0, 0.01);
  }
  REQUIRE(hot_work.temperature_c > 60.0);

  TyreState idle;
  idle.temperature_c = 120.0;
  for (int i = 0; i < 500; ++i) update_tyre(cfg, air, &idle, 0.0, 60.0, 0.01);
  REQUIRE(idle.temperature_c < 120.0);
  // It settles at ambient rather than overshooting past it.
  REQUIRE(idle.temperature_c > ambient_tyre_temperature(air) - 1.0);
}

TEST_CASE("the tyre model is stable at any step and any load",
          "[racing][tyres]") {
  // The heating/cooling loop is solved over the step rather than stepped
  // forward, so a big dt or a big coefficient cannot overshoot and oscillate.
  TyreConfig cfg;
  AtmosphereConfig air;
  for (double dt : {0.001, 0.01, 0.1, 1.0}) {
    for (double power : {0.0, 5e3, 5e5}) {
      TyreState s;
      s.temperature_c = 80.0;
      for (int i = 0; i < 200; ++i) update_tyre(cfg, air, &s, power, 50.0, dt);
      REQUIRE(std::isfinite(s.temperature_c));
      REQUIRE(s.temperature_c > -50.0);
      // Capped, and the cap must hold at any step size. Solving the heating and
      // the cooling separately fails this at dt = 1: the heat goes in with
      // nothing opposing it.
      REQUIRE(s.temperature_c <= 250.0);
      REQUIRE(s.wear >= 0.0);
      REQUIRE(s.wear <= 1.0);
    }
  }
}

TEST_CASE("a softer compound is quicker and wears faster", "[racing][tyres]") {
  // The entire reason more than one compound exists.
  const TyreCompound& soft = compound_by_index(0);
  const TyreCompound& hard = compound_by_index(n_compounds() - 1);
  REQUIRE(soft.grip > hard.grip);
  REQUIRE(soft.wear_rate > hard.wear_rate);
}

TEST_CASE("tyres wear at a rate that makes a stint a stint", "[racing][tyres]") {
  // Not a numerical check of the constant, a check that it is the right order
  // of magnitude: a set should last tens of laps, not three and not a thousand.
  EnvConfig c = one_car();
  c.track.episode_distance = -1.0;
  c.track.laps = 2;
  auto env = make(c);
  env->reset(0);

  float act[2] = {0.0f, 0.35f};
  float rew[1];
  while (!env->done()) env->step(act, rew, nullptr);

  const double laps = std::max(env->cars()[0].lap, 1);
  const double per_lap = env->cars()[0].tyre_rear.wear / laps;
  // Driven gently -- a third throttle, no racing -- so this is the LOW end of
  // what wear should be. A car being pushed does several times this.
  REQUIRE(per_lap > 0.001);   // a set lasting 1000 laps would be silly
  REQUIRE(per_lap < 0.10);    // one lasting fewer than 10 would be too
}

// --- powertrain --------------------------------------------------------------

TEST_CASE("the gearbox works through its ratios", "[racing][powertrain]") {
  PowertrainConfig cfg;
  VehicleParams vp;
  Powertrain pt(cfg, vp);

  // The gearing must actually reach the redline in top at a plausible top
  // speed. It was out by a factor of two once, and the symptom was a car
  // sitting in first gear at 170 kph against the limiter.
  REQUIRE(pt.rpm_at(85.0, 7) > 0.9 * cfg.redline_rpm);
  REQUIRE(pt.rpm_at(85.0, 7) <= cfg.redline_rpm * 1.03);
  // And first gear must be short enough to be useful out of a hairpin.
  REQUIRE(pt.rpm_at(20.0, 0) > cfg.idle_rpm);
  REQUIRE(pt.rpm_at(30.0, 0) >= 0.9 * cfg.redline_rpm);

  // A gear is selected for every speed, and higher speed never picks a lower
  // gear than a lower speed does.
  int gear = 0;
  int last = -1;
  for (double v = 5.0; v < 95.0; v += 1.0) {
    gear = pt.select_gear(v, gear);
    REQUIRE(gear >= 0);
    REQUIRE(gear < 8);
    REQUIRE(gear >= last);
    last = gear;
  }
  REQUIRE(gear >= 6);  // it got to the top of the box
}

TEST_CASE("peak power matches the fitted figure", "[racing][powertrain]") {
  PowertrainConfig cfg;
  VehicleParams vp;
  Powertrain pt(cfg, vp);

  double best = 0.0;
  for (double rpm = cfg.idle_rpm; rpm <= cfg.redline_rpm; rpm += 50.0) {
    PowertrainState s;
    s.rpm = rpm;
    s.gear = 5;
    pt.update(&s, 60.0, 1.0, 0.0, 0.01);
    best = std::max(best, s.ice_power_w);
  }
  // The curve is scaled so its peak POWER is the calibrated number, because
  // that is what the top-speed fit identified.
  REQUIRE(best == Approx(vp.max_power).epsilon(0.05));
}

TEST_CASE("the hybrid harvests, deploys and respects its lap allowance",
          "[racing][powertrain]") {
  PowertrainConfig cfg;
  VehicleParams vp;
  Powertrain pt(cfg, vp);

  PowertrainState s;
  pt.reset(&s);
  REQUIRE(s.ers_charge_mj == Approx(cfg.ers_capacity_mj * cfg.ers_start_charge));

  // Braking fills it, and it cannot be filled past the store size.
  s.ers_charge_mj = 0.0;
  for (int i = 0; i < 20000; ++i) pt.update(&s, 60.0, 0.0, 1e6, 0.01);
  REQUIRE(s.ers_charge_mj == Approx(cfg.ers_capacity_mj));

  // Throttle empties it, and never past the per-lap allowance.
  double deployed = 0.0;
  for (int i = 0; i < 20000; ++i) {
    const double before = s.ers_charge_mj;
    pt.update(&s, 60.0, 1.0, 0.0, 0.01);
    deployed += before - s.ers_charge_mj;
  }
  REQUIRE(deployed <= cfg.ers_max_deploy_per_lap_mj + 1e-6);
  REQUIRE(s.ers_deployed_lap_mj <= cfg.ers_max_deploy_per_lap_mj + 1e-6);

  // A new lap restores the allowance.
  pt.new_lap(&s);
  REQUIRE(s.ers_deployed_lap_mj == Approx(0.0));

  // Deployment is worth having.
  PowertrainState with_ers = s;
  with_ers.ers_charge_mj = 2.0;
  with_ers.ers_deployed_lap_mj = 0.0;
  pt.update(&with_ers, 60.0, 1.0, 0.0, 0.01);
  PowertrainState without = with_ers;
  without.ers_charge_mj = 0.0;
  pt.update(&without, 60.0, 1.0, 0.0, 0.01);
  REQUIRE(with_ers.drive_force_n > without.drive_force_n);
}

// --- slip -------------------------------------------------------------------

TEST_CASE("over-driving a tyre gives less force, not more", "[racing][slip]") {
  // The falling tail past the peak of the slip curve. Without it, asking for
  // more than the tyre has is free, and wheelspin costs nothing.
  Vehicle v{VehicleParams{}};
  VehicleState s;
  s.vx = 30.0;

  double best_force = 0.0;
  double force_at_full = 0.0;
  for (double thr = 0.1; thr <= 1.0; thr += 0.05) {
    VehicleInput u;
    u.throttle = thr;
    // A deliberately excessive drive force, so the tyre is the limit.
    u.drive_force = 60000.0;
    const VehicleTelemetry t = v.telemetry(s, u);
    best_force = std::max(best_force, std::abs(t.fx_rear));
    force_at_full = std::abs(t.fx_rear);
  }
  REQUIRE(force_at_full < best_force);
}

TEST_CASE("wheelspin and lockup are both reachable", "[racing][slip]") {
  Vehicle v{VehicleParams{}};
  VehicleState s;
  s.vx = 25.0;

  VehicleInput spin;
  spin.throttle = 1.0;
  spin.drive_force = 80000.0;  // far more than the rear tyres can take
  REQUIRE(v.telemetry(s, spin).wheelspin);

  VehicleInput brake;
  brake.throttle = -1.0;
  const VehicleTelemetry b = v.telemetry(s, brake);
  REQUIRE(b.lockup);

  // And neither happens when the car is driven within itself.
  VehicleInput gentle;
  gentle.throttle = 0.3;
  gentle.drive_force = 5000.0;
  REQUIRE_FALSE(v.telemetry(s, gentle).wheelspin);
  REQUIRE_FALSE(v.telemetry(s, gentle).lockup);
}

// --- DRS ---------------------------------------------------------------------

TEST_CASE("DRS zones are found on the straights", "[racing][drs]") {
  const Track t = Track::load(track_path());
  DrsConfig cfg;
  const std::vector<DrsZone> zones = find_drs_zones(t, cfg);

  REQUIRE(zones.size() >= 2);
  REQUIRE(zones.size() <= 8);

  for (const DrsZone& z : zones) {
    const double span = t.delta_s(z.end_s, z.start_s);
    REQUIRE(span >= cfg.zone_min_length_m - t.ds());

    // A zone must actually be straight, or opening the wing there would be a
    // way of leaving the circuit.
    for (double s = 0.0; s < span; s += 20.0) {
      REQUIRE(std::abs(t.kappa_at(z.start_s + s)) <= cfg.zone_max_curvature * 1.5);
    }
    // Detection sits before the zone, not inside it: being close on the
    // straight is too late, following through the corner before it is the test.
    REQUIRE(t.delta_s(z.start_s, z.detection_s) ==
            Approx(cfg.detection_before_m).margin(t.ds() + 1.0));
    REQUIRE(zone_at(zones, t, z.detection_s) == -1);
    REQUIRE(zone_at(zones, t, z.start_s + 1.0) >= 0);
  }

  // Turning it off means there are none.
  cfg.enabled = false;
  REQUIRE(find_drs_zones(t, cfg).empty());
}

TEST_CASE("DRS needs a car in front to be worth anything", "[racing][drs]") {
  // A car on its own must never open the wing, however many zones it drives
  // through. The rule is about following, not about location.
  EnvConfig c = one_car();
  c.track.episode_distance = -1.0;
  c.track.laps = 1;
  auto env = make(c);
  env->reset(0);

  float act[2] = {0.0f, 0.35f};
  float rew[1];
  bool ever_open = false;
  while (!env->done()) {
    env->step(act, rew, nullptr);
    ever_open = ever_open || env->cars()[0].drs_open;
  }
  REQUIRE_FALSE(ever_open);
}

TEST_CASE("an open wing trades downforce for straight-line speed",
          "[racing][drs]") {
  Vehicle v{VehicleParams{}};
  VehicleState s;
  s.vx = 80.0;

  DrsConfig cfg;
  VehicleInput closed;
  closed.throttle = 1.0;
  VehicleInput open = closed;
  open.drag_scale = 1.0 - cfg.drag_reduction;
  open.downforce_scale = 1.0 - cfg.downforce_loss;

  const VehicleTelemetry a = v.telemetry(s, closed);
  const VehicleTelemetry b = v.telemetry(s, open);
  REQUIRE(b.drag < a.drag);
  REQUIRE(b.downforce < a.downforce);

  VehicleState fast = s, slow = s;
  for (int i = 0; i < 200; ++i) {
    v.step(&fast, open, 0.01);
    v.step(&slow, closed, 0.01);
  }
  REQUIRE(fast.vx > slow.vx);
}

// --- it all still runs -------------------------------------------------------

TEST_CASE("a full twenty-car race completes with the physics on",
          "[racing][race]") {
  EnvConfig c;
  c.track.path = track_path();
  c.track.randomize_start = false;
  c.track.episode_distance = -1.0;
  c.track.laps = 1;
  c.seed = 9;
  auto env = make(c);
  env->reset(0);

  REQUIRE(env->n_cars() == 20);
  REQUIRE(env->config().field.n_teams == 10);

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) act[i * 2 + 1] = 0.32f + 0.005f * i;
  std::vector<float> rew(env->n_cars());

  while (!env->done()) {
    env->step(act.data(), rew.data(), nullptr);
    for (const CarState& car : env->cars()) {
      REQUIRE(std::isfinite(car.v.x));
      REQUIRE(std::isfinite(car.tyre_rear.temperature_c));
      REQUIRE(car.tyre_rear.wear >= 0.0);
      REQUIRE(car.tyre_rear.wear <= 1.0);
      REQUIRE(car.fuel_kg >= 0.0);
      REQUIRE(car.powertrain.ers_charge_mj >= -1e-9);
      REQUIRE(car.powertrain.gear >= 0);
      REQUIRE(car.powertrain.gear < 8);
    }
  }
  REQUIRE(env->cars()[0].fuel_kg < c.fuel.start_kg);
}
