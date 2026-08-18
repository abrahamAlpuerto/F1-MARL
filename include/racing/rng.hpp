// Counter-based keyed randomness.
//
// Every draw is a pure function of (seed, env, stream, step, car, sub). There
// is deliberately no generator object, no `next()`, and no mutable state
// anywhere in this header.
//
// This is not a stylistic preference, and it cannot be retrofitted. Two
// properties depend on it:
//
//   * Bit-exact replay under any thread count. Nothing depends on the order in
//     which environments or cars are visited, so an 8-thread run and a
//     1-thread run produce identical logs.
//   * Common random numbers, which is what makes comparing two policies
//     possible. Running the same race twice with one policy swapped out has to
//     draw *identical* luck, or the difference in finishing order is
//     confounded by noise. Because the key contains state coordinates rather
//     than a call count, a policy that queries the environment a different
//     number of times still sees the same draws.
//
// To "advance" a stream, add a key component. Never carry a counter.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace racing {

// One stream per concern. A new source of randomness must take a new id --
// never reuse or renumber -- so that existing draws do not shift underneath
// results that were already collected.
enum Stream : uint32_t {
  STREAM_RESET_POSE = 0,   // starting position and heading jitter
  STREAM_RESET_SPEED = 1,  // starting speed jitter
  STREAM_SENSOR = 2,       // observation noise, if ever enabled
  STREAM_SCENARIO = 3,     // scenario/difficulty sampling
};

inline constexpr uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Fold the key components into one 64-bit hash. Each component goes through its
// own mixing round so that (step=1, car=2) and (step=2, car=1) cannot collide,
// which a plain XOR or sum would allow.
inline constexpr uint64_t rng_key(uint64_t seed, uint32_t env, uint32_t stream,
                                  uint32_t step, uint32_t car,
                                  uint32_t sub = 0) noexcept {
  uint64_t h = splitmix64(seed);
  h = splitmix64(h ^ (0xD1B54A32D192ED03ull * (env + 1)));
  h = splitmix64(h ^ (0xA0761D6478BD642Full * (stream + 1)));
  h = splitmix64(h ^ (0xE7037ED1A0B428DBull * (step + 1)));
  h = splitmix64(h ^ (0x8EBC6AF09C88C6E3ull * (car + 1)));
  h = splitmix64(h ^ (0x589965CC75374CC3ull * (sub + 1)));
  return h;
}

// Uniform on [0, 1), 53-bit mantissa.
inline double uniform(uint64_t seed, uint32_t env, uint32_t stream,
                      uint32_t step, uint32_t car, uint32_t sub = 0) noexcept {
  return static_cast<double>(rng_key(seed, env, stream, step, car, sub) >> 11) *
         0x1.0p-53;
}

// Standard normal via Box-Muller on two independent sub-draws, clamped to
// +/-4 sigma. An unbounded tail would occasionally place a car somewhere
// impossible, and because the draw is keyed that placement would reproduce on
// every replay of that seed rather than washing out.
inline double normal(uint64_t seed, uint32_t env, uint32_t stream,
                     uint32_t step, uint32_t car, uint32_t sub = 0) noexcept {
  const double u1 = uniform(seed, env, stream, step, car, sub * 2u + 1u);
  const double u2 = uniform(seed, env, stream, step, car, sub * 2u + 2u);
  const double r = std::sqrt(-2.0 * std::log(u1 + 1e-300));
  const double z = r * std::cos(6.283185307179586 * u2);
  return std::clamp(z, -4.0, 4.0);
}

// Uniform on [lo, hi).
inline double uniform_range(double lo, double hi, uint64_t seed, uint32_t env,
                            uint32_t stream, uint32_t step, uint32_t car,
                            uint32_t sub = 0) noexcept {
  return lo + (hi - lo) * uniform(seed, env, stream, step, car, sub);
}

}  // namespace racing
