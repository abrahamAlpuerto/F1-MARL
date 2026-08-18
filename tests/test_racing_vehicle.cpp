// Vehicle model: does it behave like a car, and like an F1 car specifically.
//
// These are physics assertions, not regression locks. Each one would fail if a
// change broke something a driver would notice, and none of them pin a number
// tightly enough to fight a legitimate recalibration.

#include <cmath>

#include "catch2/catch.hpp"
#include "racing/qss.hpp"
#include "racing/track.hpp"
#include "racing/vehicle.hpp"

using namespace racing;

namespace {
constexpr double kG = 9.81;

VehicleState at_speed(double v) {
  VehicleState s;
  s.vx = v;
  return s;
}
}  // namespace

TEST_CASE("downforce and drag scale with the square of speed",
          "[racing][vehicle]") {
  const Vehicle v{VehicleParams{}};
  const double d1 = v.downforce_at(50.0);
  const double d2 = v.downforce_at(100.0);
  REQUIRE(d2 / d1 == Approx(4.0).epsilon(1e-9));
  REQUIRE(v.drag_at(100.0) / v.drag_at(50.0) == Approx(4.0).epsilon(1e-9));
}

TEST_CASE("downforce reaches a realistic multiple of car weight",
          "[racing][vehicle]") {
  const VehicleParams p;
  const Vehicle v{p};
  const double weight = p.mass * kG;
  // A ground-effect car makes roughly its own weight in downforce somewhere
  // around 180-200 kph, and around 3x by 300 kph.
  REQUIRE(v.downforce_at(250.0 / 3.6) / weight > 1.5);
  REQUIRE(v.downforce_at(250.0 / 3.6) / weight < 2.8);
  REQUIRE(v.downforce_at(300.0 / 3.6) / weight > 2.3);
  REQUIRE(v.downforce_at(300.0 / 3.6) / weight < 4.0);
}

TEST_CASE("cornering speed rises with radius but lateral load is bounded",
          "[racing][vehicle]") {
  const VehicleParams p;
  const Vehicle v{p};

  double prev = 0.0;
  for (double R : {15.0, 25.0, 50.0, 100.0, 200.0, 400.0}) {
    const double speed = v.max_corner_speed(1.0 / R);
    REQUIRE(speed > prev);  // a wider corner is never slower
    prev = speed;

    // Without the lateral ceiling this diverges: grip and demand both scale
    // with v^2, so above a certain radius the two curves never cross and the
    // solver reports an unbounded cornering speed.
    const double lat_g = speed * speed / R / kG;
    REQUIRE(lat_g <= p.max_lateral_g + 1e-6);
  }
}

TEST_CASE("slow-corner grip is mechanical, fast-corner grip is aerodynamic",
          "[racing][vehicle]") {
  const Vehicle v{VehicleParams{}};
  // At a hairpin there is almost no downforce, so lateral load stays near what
  // the tires alone can do. In a fast corner it must be far higher, and that
  // difference is the entire reason a wake effect can matter later.
  const double slow = v.max_corner_speed(1.0 / 20.0);
  const double fast = v.max_corner_speed(1.0 / 150.0);
  const double slow_g = slow * slow / 20.0 / kG;
  const double fast_g = fast * fast / 150.0 / kG;
  REQUIRE(slow_g < 2.6);
  REQUIRE(fast_g > 1.6 * slow_g);
}

TEST_CASE("the friction ellipse is respected under combined load",
          "[racing][vehicle]") {
  const VehicleParams p;
  const Vehicle veh{p};

  // Full brake and full steer at once: neither axle may exceed its own
  // friction circle, or the car would be able to stop and turn at full
  // capability simultaneously.
  VehicleState s = at_speed(70.0);
  s.r = 0.2;
  s.vy = -1.0;
  VehicleInput u;
  u.steer = 1.0;
  u.throttle = -1.0;

  const VehicleTelemetry t = veh.telemetry(s, u);
  const double f_mag = std::hypot(t.fx_front, t.fy_front);
  const double r_mag = std::hypot(t.fx_rear, t.fy_rear);
  REQUIRE(f_mag <= t.mu_front * t.fz_front * 1.001);
  REQUIRE(r_mag <= t.mu_rear * t.fz_rear * 1.001);
}

TEST_CASE("braking hard while turning gives up cornering force",
          "[racing][vehicle]") {
  const Vehicle veh{VehicleParams{}};
  VehicleState s = at_speed(70.0);
  s.vy = -2.0;   // sliding, so there is a real slip angle to generate force
  s.r = 0.1;

  VehicleInput coast;
  coast.steer = 0.6;
  coast.throttle = 0.0;
  VehicleInput brake = coast;
  brake.throttle = -1.0;

  const double fy_coast = std::abs(veh.telemetry(s, coast).fy_front);
  const double fy_brake = std::abs(veh.telemetry(s, brake).fy_front);
  REQUIRE(fy_brake < fy_coast);
}

TEST_CASE("longitudinal load transfer moves grip to the front under braking",
          "[racing][vehicle]") {
  const Vehicle veh{VehicleParams{}};
  VehicleInput u;

  VehicleState decel = at_speed(70.0);
  decel.ax = -30.0;
  VehicleState accel = at_speed(70.0);
  accel.ax = +10.0;

  const VehicleTelemetry td = veh.telemetry(decel, u);
  const VehicleTelemetry ta = veh.telemetry(accel, u);
  REQUIRE(td.fz_front > ta.fz_front);
  REQUIRE(td.fz_rear < ta.fz_rear);
  // Total load is unchanged: transfer moves it, it does not create it.
  REQUIRE(td.fz_front + td.fz_rear ==
          Approx(ta.fz_front + ta.fz_rear).epsilon(1e-9));
}

TEST_CASE("the car is stable at low speed rather than singular",
          "[racing][vehicle]") {
  // The dynamic model divides by longitudinal velocity, so without the
  // kinematic blend this state produces enormous slip angles and diverges.
  const Vehicle veh{VehicleParams{}};
  VehicleState s;
  s.vx = 0.5;
  s.vy = 0.2;
  s.r = 0.5;

  VehicleInput u;
  u.steer = 1.0;
  u.throttle = 0.3;
  for (int i = 0; i < 2000; ++i) {
    veh.step(&s, u, 0.01);
    REQUIRE(std::isfinite(s.vx));
    REQUIRE(std::isfinite(s.vy));
    REQUIRE(std::isfinite(s.r));
    REQUIRE(std::abs(s.r) < 50.0);
  }
}

TEST_CASE("stepping is a pure function of state and input", "[racing][vehicle]") {
  // Nothing may be carried in the Vehicle itself, or replay would depend on
  // how many times the object had been stepped before.
  const Vehicle veh{VehicleParams{}};
  VehicleInput u;
  u.steer = 0.3;
  u.throttle = 0.8;

  VehicleState a = at_speed(60.0);
  VehicleState b = at_speed(60.0);
  for (int i = 0; i < 50; ++i) veh.step(&a, u, 0.01);
  // A different history through the same object must not change the result.
  VehicleState scratch = at_speed(12.0);
  for (int i = 0; i < 500; ++i) veh.step(&scratch, u, 0.01);
  for (int i = 0; i < 50; ++i) veh.step(&b, u, 0.01);

  REQUIRE(a.vx == b.vx);
  REQUIRE(a.vy == b.vy);
  REQUIRE(a.r == b.r);
  REQUIRE(a.x == b.x);
  REQUIRE(a.y == b.y);
}

TEST_CASE("a full-throttle car accelerates and reaches a terminal speed",
          "[racing][vehicle]") {
  const Vehicle veh{VehicleParams{}};
  VehicleState s = at_speed(1.0);
  VehicleInput u;
  u.throttle = 1.0;

  double prev = s.vx;
  for (int i = 0; i < 4000; ++i) {
    veh.step(&s, u, 0.01);
    REQUIRE(s.vx >= prev - 1e-9);  // never slows under full power in a straight line
    prev = s.vx;
  }
  // Drag has to win eventually, and at a speed an F1 car actually reaches.
  REQUIRE(s.vx * 3.6 > 280.0);
  REQUIRE(s.vx * 3.6 < 380.0);
}
