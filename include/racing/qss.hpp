// Quasi-steady-state lap simulation.
//
// The classical motorsport lap-time method: assume the car is always at some
// limit -- cornering, traction, braking or power -- and sweep the track twice,
// once forwards under acceleration and once backwards under braking. The
// envelope of the two is the fastest speed profile the car could hold.
//
// Two uses here, and the first is the important one:
//
//   * Calibration. It converts the vehicle parameters into a predicted lap
//     time, which can be checked against a real one. Verstappen set 92.608 s on
//     exactly the line this track was built from, so if the model is honest the
//     solver should land near it. That is the whole realism argument, and it is
//     falsifiable rather than a matter of taste.
//   * A reference speed profile. The scripted drivers in examples/drivers.py
//     use it as their target speed, and it is the yardstick a learned policy
//     is measured against -- "4% off the quasi-steady-state bound" says
//     something about the policy, where a raw lap time does not.
//
// It is NOT the simulator. It has no notion of a driver, a line choice, or
// transient behaviour, and it cannot be raced against. It answers "how quick is
// this car, at best" and nothing else.

#pragma once

#include <vector>

#include "racing/track.hpp"
#include "racing/vehicle.hpp"

namespace racing {

struct QssResult {
  std::vector<double> speed;   // m/s at each track sample
  double lap_time = 0.0;       // s
  bool converged = false;
  int iterations = 0;
};

// `friction_scale` multiplies the grip available, so a caller can ask what the
// same car would do on a track with more or less grip than nominal without
// rebuilding the vehicle.
QssResult solve_qss(const Track& track, const Vehicle& vehicle,
                    int max_iterations = 12);

// Lap time implied by an arbitrary speed profile on this track's grid.
double lap_time_from_profile(const Track& track,
                             const std::vector<double>& speed);

}  // namespace racing
