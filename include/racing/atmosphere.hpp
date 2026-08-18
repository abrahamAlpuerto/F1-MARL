// The air the car is driving through.
//
// Air density is the term every aerodynamic force is multiplied by, so it is
// not a detail: the difference between a cool evening and a hot afternoon is
// several percent of both downforce and drag, worth a few kph of top speed and
// a measurable slice of cornering grip.
//
// Header-only because it is three formulas and they belong next to each other.

#pragma once

#include <cmath>

#include "racing/config.hpp"

namespace racing {

// Saturation vapour pressure of water, in pascals. Tetens' equation, which is
// accurate to a fraction of a percent over any temperature a race is run in.
inline double saturation_vapour_pressure(double temperature_c) {
  return 610.78 * std::exp(17.27 * temperature_c / (temperature_c + 237.3));
}

// Density of moist air, kg/m^3.
//
// Humid air is LESS dense than dry air, which is the opposite of most people's
// intuition: a water molecule weighs 18 against nitrogen's 28, so every one of
// them displaces something heavier. The effect is small -- under half a percent
// at racing temperatures -- but it is free to get right.
inline double air_density(const AtmosphereConfig& a) {
  const double t_k = a.air_temperature_c + 273.15;
  const double pv = a.humidity * saturation_vapour_pressure(a.air_temperature_c);
  const double pd = a.pressure_pa - pv;
  // Specific gas constants for dry air and for water vapour.
  return pd / (287.058 * t_k) + pv / (461.495 * t_k);
}

// Component of the wind along a car's heading, in m/s.
//
// Positive is a headwind: it opposes the car, so the airspeed the aero sees is
// higher than the ground speed. The same wind is a tailwind on the other side
// of the circuit, which is why it is worth resolving per car per step rather
// than treating it as a constant penalty.
inline double headwind_component(const AtmosphereConfig& a, double heading) {
  if (a.wind_speed <= 0.0) return 0.0;
  // `wind_direction` is where the wind is blowing TO, so a car heading the same
  // way has a tailwind.
  return -a.wind_speed * std::cos(a.wind_direction - heading);
}

}  // namespace racing
