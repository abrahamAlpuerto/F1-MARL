// Everything the engine knows about one car.
//
// Split out from race.hpp so the interaction models (wake, contact) can be
// written and tested against it without pulling in the environment, and so a
// renderer or a scripted driver can be handed one of these without knowing how
// a race is run.
//
// The fields fall into three groups, and it is worth keeping them straight:
//
//   * physical      -- `v` and `f`. What the integrator owns.
//   * race          -- distance, lap, position, times. Bookkeeping.
//   * interaction   -- downforce_factor, gaps, contact. Recomputed from
//                      scratch every physics step, never accumulated. If you
//                      find yourself wanting to remember one of these between
//                      steps, it belongs in the race group instead.

#pragma once

#include "racing/powertrain.hpp"
#include "racing/track.hpp"
#include "racing/tyres.hpp"
#include "racing/vehicle.hpp"

namespace racing {

// Why a car is out. Carried on the car and repeated on the RETIRE event, so a
// timing screen can say "accident damage" rather than just showing a car
// vanishing from the classification.
//
// Append only: these appear in the feed by name.
enum RetireReason : int {
  RETIRE_NONE = 0,
  RETIRE_COLLISION,   // accumulated damage from contact with other cars
  RETIRE_BARRIER,     // hit something solid hard enough to end it
  RETIRE_OFF_TRACK,   // left the circuit where that is configured to be fatal
  RETIRE_REASONS
};

const char* retire_reason_name(int reason);

struct CarState {
  // --- identity ----------------------------------------------------------
  int index = 0;  // 0-based position in the field, fixed for the race
  int team = 0;   // 0-based team id

  // --- physical ----------------------------------------------------------
  VehicleState v;
  Frenet f;

  // --- race bookkeeping --------------------------------------------------
  // `distance` is metres past the START/FINISH LINE, and it is monotonic, so it
  // is the thing to sort on to get the race order. Arc length `f.s` wraps at
  // the line and lap counts alone cannot separate two cars on the same lap.
  //
  // Note what it is NOT: metres covered since the car started moving. A car on
  // the eighth row begins the race a couple of hundred metres behind the line,
  // so it starts NEGATIVE and crosses zero when it takes the start. Using
  // metres-covered instead makes two cars level in the classification while
  // they are still a straight apart on the road, and the leaderboard flickers
  // between cars that cannot see each other.
  double distance = 0.0;
  double race_time = 0.0;       // s since the lights went out
  int lap = 0;                  // laps completed
  double lap_start_time = 0.0;  // race_time when the current lap began
  double last_lap_time = 0.0;   // 0 until the first lap is complete
  double best_lap_time = 0.0;   // 0 until the first lap is complete

  int position = 1;             // 1-based race position, by `distance`
  bool finished = false;
  double finish_time = 0.0;

  // --- track limits ------------------------------------------------------
  bool off_track = false;
  double off_track_time = 0.0;  // s spent continuously beyond the corridor

  // --- damage ------------------------------------------------------------
  // Accumulated, never repaired: 0 is a car as it left the garage and
  // DamageConfig::retire_threshold is one that cannot continue. It belongs in
  // the race group rather than the interaction group precisely because it is
  // the one thing about an accident that persists after the accident.
  //
  // `retired` is terminal. A retired car keeps its last position and is skipped
  // everywhere -- physics, wake, contact, the neighbour observation, the DRS
  // gap -- so it is out of the race in every sense except that the feed still
  // reports where it stopped, which is what a viewer needs to draw it.
  double damage = 0.0;
  double barrier_impact = 0.0;  // severity of the wall hit this step, else 0
  bool retired = false;         // out of the race; stops being simulated
  int retire_reason = 0;        // RetireReason; only meaningful once retired

  // --- interaction, recomputed every physics step ------------------------
  // `downforce_factor` is the one the renderer wants: 1.0 is clean air, and
  // anything below it is how much grip this car is giving up to sit where it
  // is. `drag_factor` is its counterpart on the straights -- below 1.0 means a
  // tow. `wake` is the raw wake strength both are derived from, which is the
  // convenient thing to shade a car by.
  double downforce_factor = 1.0;
  double drag_factor = 1.0;
  double wake = 0.0;            // 0 in clean air, -> 1 directly behind
  int wake_source = -1;         // index of the car whose wake this is, or -1

  double gap_ahead = 0.0;       // s to the car classified in front, 0 if none
  double gap_behind = 0.0;      // s to the car classified behind, 0 if none

  bool contact = false;         // touched another car this step
  double contact_severity = 0.0;

  // --- consumables, which is what makes a stint a stint -------------------
  // These are the state that persists and degrades. Everything above is either
  // physical or recomputed; these three are the reason lap 30 is not lap 1.
  double fuel_kg = 0.0;
  TyreState tyre_front;
  TyreState tyre_rear;
  PowertrainState powertrain;

  // --- DRS ---------------------------------------------------------------
  // `armed` is set at a detection point if the car was close enough to the one
  // ahead; `open` is armed AND inside the zone that detection point belongs to.
  // Keeping them separate is the rule: being close on the straight is too late.
  bool drs_armed = false;
  bool drs_open = false;
  int drs_zone = -1;

  // --- what the tyres are doing, for telemetry ---------------------------
  double slip_ratio = 0.0;
  bool wheelspin = false;
  bool lockup = false;
  double lateral_g = 0.0;
  double longitudinal_g = 0.0;

  // --- last applied controls, for the feed -------------------------------
  double steer = 0.0;
  double throttle = 0.0;
};

}  // namespace racing
