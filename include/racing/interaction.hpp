// What cars do to each other: the aerodynamic wake, and contact.
//
// These are the two places the field stops being a set of independent cars.
// Everything else in the engine steps one car at a time and does not care that
// the others exist; these two functions are the exception, and they are kept
// here, separate and side-effect-free apart from the state they are handed, so
// that "how does following work" is one file rather than a behaviour smeared
// through the environment.
//
// Both work in the track's Frenet frame rather than in world coordinates, and
// that is deliberate. At a hairpin the circuit doubles back on itself, so two
// cars a whole straight apart in the race can be twenty metres apart in world
// space. Comparing arc lengths never makes that mistake.

#pragma once

#include <vector>

#include "racing/car.hpp"
#include "racing/config.hpp"
#include "racing/track.hpp"

namespace racing {

// Wake strength one car sees from another: 1 directly behind and touching,
// falling to 0 with distance and with lateral offset.
//
//     w(d, y) = exp(-d / decay_length) * exp(-(y / width)^2)
//
// `d` must be positive -- the follower is behind the leader -- and anything
// beyond `range` is 0 so the sum has compact support.
double wake_strength(const AeroConfig& cfg, double gap, double lateral_offset);

// Fill in `downforce_factor`, `drag_factor`, `wake` and `wake_source` for every
// car from the positions of all the others.
//
// A car takes the strongest single wake it is sitting in rather than the sum of
// all of them. Summing lets a queue of cars stack up an arbitrarily large
// effect -- five cars nose to tail and the one at the back has no downforce at
// all -- which is not what happens and makes trains of cars behave absurdly.
void apply_wake(const AeroConfig& cfg, const Track& track,
                std::vector<CarState>* cars);

struct Contact {
  int a = -1;         // the car behind, or the one on the inside
  int b = -1;
  double severity = 0.0;  // 0 to 1: a brush against a genuine hit
};

// Push apart any cars that are overlapping, scrub some speed, and add a yaw
// disturbance. Appends what happened to `out`, which the environment turns into
// events and penalties.
//
// Pairs are visited in a fixed order and resolved in place, so the result does
// not depend on how the field happens to be laid out in memory -- which matters
// because a replay has to be reproducible.
//
// `dt` matters: the speed scrub and the yaw kick are RATES, applied for the
// length of the step. Applying them as fixed amounts per call -- which is what
// this did first -- means a touch that lasts a fifth of a second is applied
// twenty times, and the numbers that look reasonable per contact are lethal
// per step. Two cars rubbing for a second lost 99% of their speed and the whole
// field ground to a halt on the exit of Turn 1.
void resolve_contacts(const ContactConfig& cfg, const VehicleParams& vp,
                      const Track& track, double dt,
                      std::vector<CarState>* cars, std::vector<Contact>* out);

// Race positions, gaps, and lap accounting off the back of `distance`.
// Positions are 1-based and dense; a retired car keeps the position it held.
void classify(std::vector<CarState>* cars);

// --- DRS -------------------------------------------------------------------

struct DrsZone {
  double detection_s = 0.0;  // where the gap is measured
  double start_s = 0.0;      // where the wing may open
  double end_s = 0.0;        // and where it must close
};

// Find the stretches of a circuit worth opening a rear wing on.
//
// Derived from the geometry rather than hand-entered per track: a zone is a
// run of road straight enough and long enough that the downforce being given
// up is downforce the car was not using. That means this works on any circuit
// the engine is given, and it cannot drift out of step with the geometry the
// way a table of arc lengths would.
std::vector<DrsZone> find_drs_zones(const Track& track, const DrsConfig& cfg);

// Which zone contains `s`, or -1. Zones wrap with the circuit.
int zone_at(const std::vector<DrsZone>& zones, const Track& track, double s);

// Time gap, in seconds, from `me` to the nearest car ahead of it on the road.
// Large if there is nobody there. This is the quantity the DRS rule is written
// against -- the car in front on the ROAD, which is not always the car in front
// in the classification once anybody has been lapped.
double gap_to_car_ahead(const Track& track, const std::vector<CarState>& cars,
                        int me, double max_gap_m = 200.0);

}  // namespace racing
