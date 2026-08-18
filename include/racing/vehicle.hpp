// Dynamic bicycle model with a load-sensitive tire and a friction ellipse.
//
// The model is deliberately the standard one from the autonomous racing
// literature rather than anything more elaborate. What it does include, beyond
// a textbook bicycle model, is the handful of effects that decide whether an F1
// car's behaviour is recognisable:
//
//   * Aerodynamic downforce, which roughly triples the grip between a hairpin
//     and a fast sweeper. Without it a car corners at the same speed
//     everywhere, and dirty air -- the thing that makes following another car
//     hard, and so the thing that makes racing interesting -- costs nothing.
//   * Tire load sensitivity, so grip grows slower than load. Without it,
//     downforce buys unlimited cornering speed.
//   * Longitudinal load transfer, so the front axle gains grip under braking
//     and loses it under power. This is what makes trail-braking work.
//   * LATERAL load transfer onto four corner loads. Because grip is sub-linear
//     in load, two wheels at (Fz/2 +/- d) always make less than two at Fz/2, so
//     transferring weight across the car costs the axle grip. That is the whole
//     mechanism behind chassis balance, and a model without it corners the same
//     whatever the setup.
//   * A friction ellipse per axle, so a tire cannot brake and turn at full
//     capacity at once, with a falling tail past the peak -- over-commanding
//     the tire gives you LESS force, which is what wheelspin and a locked
//     wheel actually are.
//
// The vehicle knows nothing about other cars, the weather, the fuel load or the
// state of its tires, and it must not: two cars could not then be stepped
// independently. All of that reaches it through `VehicleInput`, which is the
// one place the outside world touches the physics.

#pragma once

#include "racing/config.hpp"

namespace racing {

struct VehicleState {
  double x = 0.0, y = 0.0, psi = 0.0;   // world pose (m, m, rad)
  double vx = 0.0, vy = 0.0;            // body-frame velocity (m/s)
  double r = 0.0;                       // yaw rate (rad/s)

  // Carried between steps so load transfer can use it. Holding it fixed across
  // a step, rather than solving the implicit loop, is standard and at 100 Hz
  // the error is far below the model's own fidelity.
  double ax = 0.0, ay = 0.0;            // body-frame acceleration (m/s^2)

  double speed() const;
};

struct VehicleInput {
  double steer = 0.0;     // [-1, 1], scaled to the roadwheel angle
  double throttle = 0.0;  // [-1, 1]; positive drives, negative brakes

  // --- what the world is doing to this car, this step ---------------------
  //
  // Every one of these defaults to "nothing", so a bare VehicleInput is the
  // car the calibration was fitted against: sea-level air, no wind, no fuel,
  // fresh tires at temperature, flat road, clean air.

  // The wake of the car ahead, and DRS. Less drag is a tow; less downforce is
  // dirty air. See AeroConfig and DrsConfig.
  double downforce_scale = 1.0;
  double drag_scale = 1.0;

  // The surface. Below 1 on the run-off, so a car that has gone wide loses
  // time rather than being teleported back onto the circuit.
  double grip_scale = 1.0;

  // The tires, per axle, from TyreState. Front and rear differ because they
  // heat and wear at different rates -- which is what understeer at the end of
  // a stint actually is.
  double grip_front = 1.0;
  double grip_rear = 1.0;

  // Fuel on board, in kilograms. A hundred kilograms on a 798 kg car is an
  // eighth of its mass and it burns off over a race.
  double extra_mass = 0.0;

  // Air density, kg/m^3. Zero or less means "use VehicleParams::air_density",
  // which is the density the aero coefficients were fitted at.
  double air_density = 0.0;

  // Headwind component along the car's heading, m/s. Positive opposes the car,
  // so it raises the airspeed the aero sees without changing the ground speed.
  double headwind = 0.0;

  // Road gradient as sin(slope). Positive is uphill, and it costs m*g*sin --
  // which at Bahrain's modest elevation change is small but not nothing.
  double grade = 0.0;

  // Force the powertrain can put through the driven wheels, in newtons.
  // Negative means "no powertrain model, fall back to max_power / v".
  double drive_force = -1.0;
};

// Everything the physics worked out on the way, for tests, diagnostics, and the
// telemetry the feed carries.
struct VehicleTelemetry {
  double fz_front = 0.0, fz_rear = 0.0;       // N, per axle
  // Per corner, after lateral transfer. An inside wheel can reach zero, which
  // is a car up on two wheels and is why these are clamped.
  double fz_fl = 0.0, fz_fr = 0.0, fz_rl = 0.0, fz_rr = 0.0;

  double alpha_front = 0.0, alpha_rear = 0.0; // rad, slip angles
  double fy_front = 0.0, fy_rear = 0.0;       // N
  double fx_front = 0.0, fx_rear = 0.0;       // N
  double downforce = 0.0, drag = 0.0;         // N
  double mu_front = 0.0, mu_rear = 0.0;
  bool front_saturated = false, rear_saturated = false;

  // Longitudinal slip on the driven/braked axle, and what it means.
  double slip_ratio = 0.0;
  bool wheelspin = false, lockup = false;

  // Frictional work rate at each axle's contact patches, in watts. This is what
  // heats and wears a tire, and it is the one output the tire model consumes.
  double slip_power_front = 0.0, slip_power_rear = 0.0;

  double lateral_g = 0.0, longitudinal_g = 0.0;
  double mass = 0.0;  // including fuel
};

class Vehicle {
 public:
  explicit Vehicle(const VehicleParams& p);

  // Advance by `dt` using RK4. Deterministic and free of hidden state: the
  // result is a pure function of (state, input, dt, params).
  //
  // `out`, if given, is filled from the first RK4 stage -- the forces at the
  // state the step began from. That is free, because that stage is computed
  // anyway, and it is what the tyre model needs. Asking for telemetry
  // separately would mean a fifth force evaluation every step per car.
  void step(VehicleState* s, const VehicleInput& u, double dt,
            VehicleTelemetry* out = nullptr) const;

  // Forces at the current state, without integrating.
  VehicleTelemetry telemetry(const VehicleState& s, const VehicleInput& u) const;

  // --- steady-state limits ------------------------------------------------
  // These answer "what could this car do here", which is what a quasi-steady-
  // state lap simulation needs, and what the calibration is fitted against.

  // Fastest speed sustainable through curvature `kappa` in steady state.
  // Solved by bisection because grip depends on speed through downforce, so
  // the balance point cannot be written in closed form.
  double max_corner_speed(double kappa) const;

  // Longitudinal acceleration available at `v` while already using `lat_accel`
  // of lateral capability. Negative for the braking limit.
  double max_long_accel(double v, double lat_accel) const;
  double max_long_decel(double v, double lat_accel) const;

  double downforce_at(double v) const;
  double drag_at(double v) const;

  // Total grip an axle can make, given its load and how much of it has moved
  // across the car. Exposed because it is the one place load sensitivity and
  // lateral transfer meet, and it is worth being able to test directly.
  double axle_grip(double fz_axle, double lateral_transfer) const;

  const VehicleParams& params() const { return p_; }

 private:
  struct Deriv {
    double dx, dy, dpsi, dvx, dvy, dr;
  };
  Deriv derivative(const VehicleState& s, const VehicleInput& u,
                   VehicleTelemetry* out) const;

  double mu_at(double fz) const;

  VehicleParams p_;
  double lf_ = 0.0, lr_ = 0.0;  // CG to front / rear axle, metres
};

}  // namespace racing
