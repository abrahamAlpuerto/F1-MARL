// The engine, the gearbox, and the hybrid.
//
// A single "max power" number says the car pushes equally hard at every speed
// in every gear, which is not how anything with a gearbox behaves. Three
// effects come out of doing it properly, and all three are visible:
//
//   * Gearing. First gear multiplies engine torque twenty-odd times, so a car
//     out of a hairpin is traction-limited rather than power-limited; top gear
//     multiplies it six times and the car is fighting drag. The same throttle
//     pedal means completely different things at the two ends of a lap.
//   * Shifts. Torque is cut for a few hundredths of a second on each change.
//     Small, but it is there in the trace and it is what makes an acceleration
//     curve look like a real one rather than a smooth arc.
//   * The hybrid, which is the interesting one, because it is a RESOURCE. The
//     car harvests under braking and spends on the straights, and it can only
//     spend so much per lap. Where to spend it is a decision -- and one a team
//     can coordinate, because two cars that both deploy down the same straight
//     while nose to tail get the tow as well.
//
// Deployment here is automatic: hard on the throttle, above a sensible speed,
// while there is charge and lap budget left. Making it a third agent action --
// "when do I spend my battery" -- is the obvious next step and is deliberately
// not taken yet, because it changes the action space everywhere.

#pragma once

#include "racing/config.hpp"

namespace racing {

struct PowertrainState {
  int gear = 2;                 // 0-based index into gear_ratios
  double rpm = 0.0;
  double shift_timer = 0.0;     // seconds of torque cut remaining

  double ers_charge_mj = 0.0;
  double ers_deployed_lap_mj = 0.0;
  bool deploying = false;

  // Telemetry for the feed.
  double ice_power_w = 0.0;
  double ers_power_w = 0.0;
  double drive_force_n = 0.0;   // what the rear axle could put down
};

class Powertrain {
 public:
  Powertrain(const PowertrainConfig& cfg, const VehicleParams& vp);

  void reset(PowertrainState* s) const;

  // Engine speed implied by road speed in a given gear.
  double rpm_at(double speed, int gear) const;

  // Pick a gear for this speed. Hysteresis between the up and down thresholds
  // keeps it from hunting either side of a shift point.
  int select_gear(double speed, int current) const;

  // Advance the gearbox and the battery, and work out how much force the rear
  // axle can be given this step.
  //
  // `throttle` is the driver's demand, 0 to 1. `braking_power_w` is what the
  // brakes are dissipating, which is what there is to harvest from.
  void update(PowertrainState* s, double speed, double throttle,
              double braking_power_w, double dt) const;

  // Called when a car crosses the line: the deployment allowance is per lap.
  void new_lap(PowertrainState* s) const;

  // Force available at the driven wheels right now, from engine plus hybrid.
  double drive_force(const PowertrainState& s, double speed) const;

  double peak_torque_nm() const { return peak_torque_; }
  const PowertrainConfig& config() const { return cfg_; }

 private:
  double torque_fraction(double rpm) const;

  PowertrainConfig cfg_;
  VehicleParams vp_;
  double peak_torque_ = 0.0;  // Nm, scaled so peak POWER matches max_power
};

}  // namespace racing
