#include "racing/tyres.hpp"

#include <algorithm>
#include <cmath>

namespace racing {

namespace {

// Soft, medium, hard.
//
// `wear_rate` is fraction of life per megajoule of frictional work, and
// `heat_rate` is kelvin per megajoule.
//
// A hard lap of Bahrain turns out to do around four megajoules of frictional
// work per axle, not the one the first guess assumed, so the medium loses
// roughly 3% of its life a lap -- a stint of about thirty laps, which is right.
// The first numbers were five times too aggressive and wore a set out in three.
const TyreCompound kCompounds[] = {
    {"soft",   1.030, 92.0, 25.0, 0.0110, 36.0},
    {"medium", 1.000, 95.0, 27.0, 0.0065, 34.0},
    {"hard",   0.972, 99.0, 30.0, 0.0040, 32.0},
};
constexpr int kCompoundCount =
    static_cast<int>(sizeof(kCompounds) / sizeof(kCompounds[0]));

// How much of the tyre's grip is left when it is completely worn is a config
// value; how fast it gets there is this shape. Wear costs little at first and
// bites at the end, which is what a driver means by the tyre "falling off a
// cliff".
double wear_grip(const TyreConfig& cfg, double wear) {
  const double w = std::clamp(wear, 0.0, 1.0);
  return 1.0 - (1.0 - cfg.grip_worn) * w * w;
}

}  // namespace

int n_compounds() { return kCompoundCount; }

const TyreCompound& compound_by_index(int index) {
  return kCompounds[std::clamp(index, 0, kCompoundCount - 1)];
}

const TyreCompound& compound(const TyreConfig& cfg) {
  return compound_by_index(cfg.compound);
}

double ambient_tyre_temperature(const AtmosphereConfig& air) {
  // A stationary tyre sits between the air and the tarmac it is standing on,
  // nearer the tarmac because that is what it is touching.
  return 0.35 * air.air_temperature_c + 0.65 * air.track_temperature_c;
}

double tyre_grip(const TyreConfig& cfg, const TyreState& t) {
  if (!cfg.enabled) return 1.0;
  const TyreCompound& c = compound(cfg);

  // Wear moves the window as well as lowering the peak: a thin tyre has less
  // rubber to carry heat away from the surface, so it wants to run cooler.
  const double optimum = c.optimum_c + cfg.wear_optimum_shift_c * t.wear;
  const double x = (t.temperature_c - optimum) / std::max(c.window_c, 1e-6);

  // Quadratic either side of the optimum, with different slopes: a cold tyre
  // and a greasy overheated one are both worse than one in the window, and
  // they are not equally bad.
  const double edge = x <= 0.0 ? cfg.grip_cold : cfg.grip_hot;
  // Bounded, and deliberately so.
  //
  // Temperature, grip and sliding form a feedback loop: less grip means more
  // sliding, more sliding means more heat, and more heat means less grip again.
  // Wound tight enough, that loop is unstable in BOTH directions, and both were
  // observed. Too little heat and the field settled at ambient on minimum grip,
  // too slow to ever warm the tyres up. Too much and it ran away to 170 C and
  // stayed there, sliding its way to more heat.
  //
  // A wide window and a bounded falloff keep the effect real -- an out-lap is
  // slow, a car that has been abusing its tyres is slower -- without letting it
  // dominate. Roughly a quarter of the grip range, across 27 K either side of
  // optimum, which is also closer to what a real tyre does than the cliff the
  // first version had.
  const double falloff = std::min(x * x, 2.0);
  const double thermal = std::max(0.72, 1.0 - (1.0 - edge) * falloff);

  return c.grip * thermal * wear_grip(cfg, t.wear);
}

void update_tyre(const TyreConfig& cfg, const AtmosphereConfig& air,
                 TyreState* t, double slip_power_w, double airspeed, double dt,
                 bool is_front) {
  t->slip_power_w = slip_power_w;
  if (!cfg.enabled) {
    t->grip = 1.0;
    return;
  }
  const TyreCompound& c = compound(cfg);
  const double work_mj = std::max(0.0, slip_power_w) * dt * 1e-6;

  // Heating and cooling solved together, as the one first-order system they
  // are:
  //
  //     dT/dt = A - k (T - T_ref)
  //
  // with A the heating rate from the contact patch and k the Newton cooling
  // coefficient. The exact solution over the step is unconditionally stable and
  // independent of dt, which matters because adding the heat explicitly and
  // then solving only the cooling is neither: at a large step it integrates the
  // heat forward without any cooling to oppose it, and a sustained heavy load
  // took a tyre past 1200 C.
  //
  // k rises with airspeed, which is why tyres cool down the straights and cook
  // through a corner sequence -- and why a car stuck in traffic never gets a
  // break.
  const double heat_rate =
      c.heat_rate * (is_front ? cfg.front_heat_scale : 1.0);
  const double gain = heat_rate * std::max(0.0, slip_power_w) * 1e-6;  // K/s
  const double ref = ambient_tyre_temperature(air);
  const double coeff =
      std::max(cfg.cooling_rate + cfg.cooling_speed_factor * std::max(airspeed, 0.0),
               1e-9);
  const double equilibrium = ref + gain / coeff;
  t->temperature_c =
      equilibrium + (t->temperature_c - equilibrium) * std::exp(-coeff * dt);

  // A physical ceiling. Rubber blisters and delaminates long before this; the
  // model has no way to represent a tyre coming apart, so it caps instead of
  // pretending a 500 C tyre is a thing that keeps working.
  t->temperature_c = std::min(t->temperature_c, 250.0);

  // Wear. Proportional to the same frictional work, scaled by how hot it is:
  // rubber above its window goes off far faster, which is the mechanism behind
  // "he destroyed his tyres chasing".
  // Capped: an overheating tyre wears faster, but not without limit, or a car
  // that gets hot once destroys a set in a single lap.
  const double heat_factor = std::min(
      2.0, 1.0 + 0.9 * std::max(0.0, (t->temperature_c - c.optimum_c) /
                                         std::max(c.window_c, 1e-6)));
  t->wear = std::min(1.0, t->wear + c.wear_rate * work_mj * heat_factor);

  t->grip = tyre_grip(cfg, *t);
}

}  // namespace racing
