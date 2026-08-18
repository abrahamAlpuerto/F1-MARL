// Tyres that have a temperature and wear out.
//
// A friction ellipse on its own says a tyre is the same on lap 1 and lap 40,
// on an out-lap and at the end of a stint, in traffic and in clean air. It is
// none of those things, and the differences are most of what a race is about.
//
// Three effects, all of which a viewer can see:
//
//   * Warm-up. A tyre off the blankets is below its working range and the first
//     lap is slow. So is the lap after a car has trailed round behind someone
//     without loading them up.
//   * Overheating. Working a tyre harder than it can shed heat pushes it past
//     the window and the grip goes away. This is what actually punishes a car
//     for sitting in dirty air for five laps: it is sliding more to keep up,
//     which cooks the tyres, so when it finally gets clear it cannot attack.
//   * Wear. Grip falls with the rubber left, so the order at the end of a stint
//     is not the order at the start, and a softer compound trades life for pace.
//
// The model is a single core temperature per axle driven by frictional work,
// which is the standard first-order treatment. It is not a carcass/surface
// two-node model and it does not track individual wheels or camber; see the
// note at the bottom of docs/RACING_ENGINE.md for where the line is drawn.

#pragma once

#include "racing/config.hpp"

namespace racing {

struct TyreState {
  double temperature_c = 80.0;
  double wear = 0.0;  // 0 fresh, 1 fully worn

  // Carried for telemetry rather than used by the model.
  double grip = 1.0;         // the multiplier this state currently gives
  double slip_power_w = 0.0; // frictional work rate that heated it this step
};

// The three compounds. Softer is quicker and wears faster, which is the entire
// trade and the reason more than one exists.
const TyreCompound& compound(const TyreConfig& cfg);
const TyreCompound& compound_by_index(int index);
int n_compounds();

// Grip multiplier from temperature and wear, 0 to ~1. Peak is 1.0 at the
// compound's optimum temperature when fresh.
double tyre_grip(const TyreConfig& cfg, const TyreState& t);

// Advance one axle's tyres by `dt`.
//
// `slip_power_w` is the frictional work rate the contact patch is doing, which
// is what heats a tyre and what wears it. `airspeed` drives the cooling: a car
// on a straight is pushing far more air through its wheels than one in a
// hairpin, which is why tyres cool down the straights and cook in the corners.
void update_tyre(const TyreConfig& cfg, const AtmosphereConfig& air,
                 TyreState* t, double slip_power_w, double airspeed, double dt,
                 bool is_front = false);

// The temperature a tyre with nothing being asked of it settles at.
double ambient_tyre_temperature(const AtmosphereConfig& air);

}  // namespace racing
