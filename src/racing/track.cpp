#include "racing/track.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace racing {
namespace {

// Wrap an angle into [-pi, pi).
inline double wrap_pi(double a) {
  constexpr double kTwoPi = 6.283185307179586;
  a = std::fmod(a + 3.141592653589793, kTwoPi);
  if (a < 0.0) a += kTwoPi;
  return a - 3.141592653589793;
}

std::vector<double> get_array(const nlohmann::json& j, const char* key,
                              size_t expect) {
  if (!j.contains(key)) {
    throw std::runtime_error(std::string("track file missing key: ") + key);
  }
  auto v = j.at(key).get<std::vector<double>>();
  if (expect && v.size() != expect) {
    throw std::runtime_error(std::string("track key '") + key +
                             "' has the wrong length");
  }
  return v;
}

}  // namespace

Track Track::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open track file: " + path);

  nlohmann::json j;
  in >> j;

  Track t;
  t.name_ = j.value("name", "unnamed");
  t.length_ = j.at("lap_length_m").get<double>();
  t.ds_ = j.at("ds_m").get<double>();
  t.ref_lap_time_ = j.value("reference_lap_time_s", 0.0);

  const size_t n = j.at("n").get<size_t>();
  t.x_ = get_array(j, "x", n);
  t.y_ = get_array(j, "y", n);
  // Elevation is optional: a track built before the profile existed simply
  // renders flat rather than failing to load.
  t.z_ = j.contains("z") ? get_array(j, "z", n) : std::vector<double>(n, 0.0);
  t.theta_ = get_array(j, "theta", n);
  t.kappa_ = get_array(j, "kappa", n);
  t.nx_ = get_array(j, "nx", n);
  t.ny_ = get_array(j, "ny", n);
  t.half_left_ = get_array(j, "half_width_left", n);
  t.half_right_ = get_array(j, "half_width_right", n);
  t.tel_speed_ = get_array(j, "telemetry_speed_ms", n);
  if (j.contains("sector_s")) {
    t.sector_s_ = j.at("sector_s").get<std::vector<double>>();
  }

  // The grid must actually be uniform, because every lookup below turns an arc
  // length into an index by division. A file that violated it would produce
  // subtly wrong curvature rather than an error.
  const double expect_ds = t.length_ / static_cast<double>(n);
  if (std::abs(expect_ds - t.ds_) > 1e-6 * std::max(1.0, t.ds_)) {
    throw std::runtime_error("track ds_m is inconsistent with lap_length_m / n");
  }
  return t;
}

double Track::wrap_s(double s) const {
  s = std::fmod(s, length_);
  return s < 0.0 ? s + length_ : s;
}

double Track::delta_s(double a, double b) const {
  double d = std::fmod(a - b, length_);
  if (d < 0.0) d += length_;
  if (d > 0.5 * length_) d -= length_;
  return d;
}

double Track::kappa_at(double s) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(kappa_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  return kappa_[i0] * (1.0 - f) + kappa_[i1] * f;
}

double Track::half_width_at(double s) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(half_left_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  return half_left_[i0] * (1.0 - f) + half_left_[i1] * f;
}

double Track::z_at(double s) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(z_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  return z_[i0] * (1.0 - f) + z_[i1] * f;
}

double Track::grade_at(double s) const {
  // Central difference around the loop. A forward difference would put a step
  // in the gradient at the start/finish point, because the circuit is closed.
  const int n = static_cast<int>(z_.size());
  if (n < 3 || ds_ <= 0.0) return 0.0;
  const int i = static_cast<int>(wrap_s(s) / ds_) % n;
  const double dz = z_[(i + 1) % n] - z_[(i - 1 + n) % n];
  const double run = 2.0 * ds_;
  // sin(slope), not tan: the two differ by under half a percent at any gradient
  // a circuit has, but the force term wants the sine and it is free to be right.
  return dz / std::sqrt(run * run + dz * dz);
}

double Track::telemetry_speed_at(double s) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(tel_speed_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  return tel_speed_[i0] * (1.0 - f) + tel_speed_[i1] * f;
}

void Track::pose_at(double s, double* x, double* y, double* theta) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(x_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  if (x) *x = x_[i0] * (1.0 - f) + x_[i1] * f;
  if (y) *y = y_[i0] * (1.0 - f) + y_[i1] * f;
  if (theta) {
    // Interpolate through the shortest arc, or the sample either side of the
    // +/-pi branch cut would average to a heading pointing backwards.
    const double d = wrap_pi(theta_[i1] - theta_[i0]);
    *theta = theta_[i0] + f * d;
  }
}

void Track::to_world(double s, double e_y, double* x, double* y) const {
  const double u = wrap_s(s) / ds_;
  const int i = static_cast<int>(u);
  const double f = u - i;
  const int n = static_cast<int>(x_.size());
  const int i0 = i % n, i1 = (i + 1) % n;
  const double cx = x_[i0] * (1.0 - f) + x_[i1] * f;
  const double cy = y_[i0] * (1.0 - f) + y_[i1] * f;
  const double ux = nx_[i0] * (1.0 - f) + nx_[i1] * f;
  const double uy = ny_[i0] * (1.0 - f) + ny_[i1] * f;
  const double mag = std::sqrt(ux * ux + uy * uy);
  *x = cx + e_y * ux / mag;
  *y = cy + e_y * uy / mag;
}

Frenet Track::refine(double x, double y, double heading, int i) const {
  const int n = static_cast<int>(x_.size());
  i = ((i % n) + n) % n;

  // Project onto the two segments meeting at sample i and keep the better one.
  // Taking the nearest *sample* alone would quantise arc length to ds and make
  // progress advance in visible steps.
  double best_t = 0.0, best_d2 = 1e300;
  int best_i = i;
  for (int k = -1; k <= 0; ++k) {
    const int a = ((i + k) % n + n) % n;
    const int b = (a + 1) % n;
    const double ax = x_[a], ay = y_[a];
    const double bx = x_[b] - ax, by = y_[b] - ay;
    const double len2 = bx * bx + by * by;
    double t = len2 > 0.0 ? ((x - ax) * bx + (y - ay) * by) / len2 : 0.0;
    t = std::clamp(t, 0.0, 1.0);
    const double px = ax + t * bx, py = ay + t * by;
    const double d2 = (x - px) * (x - px) + (y - py) * (y - py);
    if (d2 < best_d2) {
      best_d2 = d2;
      best_t = t;
      best_i = a;
    }
  }

  Frenet f;
  f.s = wrap_s((best_i + best_t) * ds_);

  // Signed offset, using the left normal so positive is left of the line.
  double cx, cy, th;
  pose_at(f.s, &cx, &cy, &th);
  f.e_y = -(x - cx) * std::sin(th) + (y - cy) * std::cos(th);
  f.e_psi = wrap_pi(heading - th);
  return f;
}

Frenet Track::project(double x, double y, double heading, double s_hint) const {
  const int n = static_cast<int>(x_.size());
  // A car covers at most a few metres per physics step, so a window of a few
  // tens of samples is generous. Widening it costs time in the innermost loop
  // of the simulator.
  constexpr int kWindow = 24;
  const int centre = static_cast<int>(wrap_s(s_hint) / ds_);

  int best = centre;
  double best_d2 = 1e300;
  for (int k = -kWindow; k <= kWindow; ++k) {
    const int i = ((centre + k) % n + n) % n;
    const double dx = x - x_[i], dy = y - y_[i];
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }
  return refine(x, y, heading, best);
}

Frenet Track::project_global(double x, double y, double heading) const {
  const int n = static_cast<int>(x_.size());
  int best = 0;
  double best_d2 = 1e300;
  for (int i = 0; i < n; ++i) {
    const double dx = x - x_[i], dy = y - y_[i];
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }
  return refine(x, y, heading, best);
}

}  // namespace racing
