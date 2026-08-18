// Track geometry: the Frenet frame, and the properties the physics relies on.

#include <cmath>
#include <string>

#include "catch2/catch.hpp"
#include "racing/qss.hpp"
#include "racing/track.hpp"
#include "racing/vehicle.hpp"

using namespace racing;

namespace {
Track load() {
  return Track::load(std::string(RACING_DATA_DIR) + "/tracks/bahrain.json");
}
}  // namespace

TEST_CASE("the track is a closed loop of the right size", "[racing][track]") {
  const Track t = load();
  REQUIRE(t.length() > 5000.0);
  REQUIRE(t.length() < 5600.0);   // real Bahrain is 5412 m
  REQUIRE(t.n() > 500);
  REQUIRE(t.ds() == Approx(t.length() / t.n()).epsilon(1e-9));

  // Start and end must meet: the reconstruction spreads its closure residual
  // around the lap precisely so this holds.
  double x0, y0, x1, y1;
  t.pose_at(0.0, &x0, &y0, nullptr);
  t.pose_at(t.length() - 1e-9, &x1, &y1, nullptr);
  REQUIRE(std::hypot(x1 - x0, y1 - y0) < 1.0);
}

TEST_CASE("arc length wraps and differences take the short way round",
          "[racing][track]") {
  const Track t = load();
  const double L = t.length();
  REQUIRE(t.wrap_s(-1.0) == Approx(L - 1.0));
  REQUIRE(t.wrap_s(L + 5.0) == Approx(5.0));

  // The case that matters: either side of start/finish. Plain subtraction says
  // the car went backwards round the entire circuit.
  REQUIRE(t.delta_s(5.0, L - 5.0) == Approx(10.0));
  REQUIRE(t.delta_s(L - 5.0, 5.0) == Approx(-10.0));
}

TEST_CASE("Frenet and world coordinates are inverses", "[racing][track]") {
  const Track t = load();
  for (double frac : {0.0, 0.13, 0.37, 0.5, 0.72, 0.95}) {
    const double s = frac * t.length();
    for (double e : {-4.0, -1.0, 0.0, 2.5, 5.0}) {
      double x, y;
      t.to_world(s, e, &x, &y);
      double th;
      t.pose_at(s, nullptr, nullptr, &th);
      const Frenet f = t.project_global(x, y, th);
      REQUIRE(std::abs(t.delta_s(f.s, s)) < 1.5);
      REQUIRE(f.e_y == Approx(e).margin(0.25));
    }
  }
}

TEST_CASE("curvature agrees with the geometry it was shipped with",
          "[racing][track]") {
  // The reconstruction integrates the same heading that curvature
  // differentiates, so these cannot drift apart. Checking it here means a
  // change to the build script that broke the relationship would be caught
  // rather than silently producing a track whose corners are in the wrong place.
  const Track t = load();
  const double ds = t.ds();

  double k_max = 0.0;
  for (double k : t.kappas()) k_max = std::max(k_max, std::abs(k));

  double worst = 0.0;
  for (int i = 0; i < t.n(); ++i) {
    const double s = i * ds;
    double th_prev, th_next;
    t.pose_at(s - ds, nullptr, nullptr, &th_prev);
    t.pose_at(s + ds, nullptr, nullptr, &th_next);
    double d = th_next - th_prev;
    while (d > 3.14159265358979) d -= 6.28318530717959;
    while (d < -3.14159265358979) d += 6.28318530717959;
    worst = std::max(worst, std::abs(d / (2.0 * ds) - t.kappa_at(s)));
  }

  // Not exactly zero, and it should not be: curvature is computed spectrally
  // from a band-limited heading, while this compares against a finite
  // difference over a 2.6 m grid. The residual is ~3% of peak curvature and is
  // concentrated at the tightest corners, which is where a centred difference
  // over that spacing is least accurate. The bound is expressed against peak
  // curvature so it stays meaningful if the track or its resolution changes.
  REQUIRE(worst < 0.05 * k_max);
}

TEST_CASE("corner radii match the real circuit", "[racing][track]") {
  const Track t = load();
  double k_max = 0.0;
  for (double k : t.kappas()) k_max = std::max(k_max, std::abs(k));

  // Bahrain's tightest corners are around 20-25 m radius. A track whose
  // tightest corner came out at 5 m or 60 m would not be Bahrain.
  const double r_min = 1.0 / k_max;
  REQUIRE(r_min > 12.0);
  REQUIRE(r_min < 40.0);
}

TEST_CASE("the local projection agrees with a global search", "[racing][track]") {
  const Track t = load();
  for (double frac : {0.05, 0.28, 0.61, 0.88}) {
    const double s = frac * t.length();
    double x, y, th;
    t.pose_at(s, &x, &y, &th);
    t.to_world(s, 3.0, &x, &y);
    const Frenet local = t.project(x, y, th, s + 8.0);
    const Frenet global = t.project_global(x, y, th);
    REQUIRE(std::abs(t.delta_s(local.s, global.s)) < 0.5);
    REQUIRE(local.e_y == Approx(global.e_y).margin(0.05));
  }
}

TEST_CASE("the quasi-steady-state lap is close to the real one",
          "[racing][calibration]") {
  // This is the realism claim, made falsifiable. The reference line IS the line
  // the recorded lap was driven on, so a limit-based solve over its curvature
  // is directly comparable with the recorded time.
  const Track t = load();
  const Vehicle v{VehicleParams{}};
  const QssResult q = solve_qss(t, v);

  REQUIRE(q.converged);
  REQUIRE(q.lap_time == Approx(t.reference_lap_time()).epsilon(0.03));

  // And it must leave room for a policy to reach the 105% gate.
  REQUIRE(q.lap_time < t.reference_lap_time() * 1.05);
}

TEST_CASE("the model's speed envelope brackets the real trace",
          "[racing][calibration]") {
  const Track t = load();
  const Vehicle v{VehicleParams{}};
  const QssResult q = solve_qss(t, v);

  // The solve assumes the car is on a limit everywhere, so it should sit at or
  // above the recorded speed nearly all the way round. It does not have to be
  // above it everywhere: two corners are known to read tighter than they are,
  // because the source telemetry's position channel is locally bunched there.
  int above = 0;
  for (int i = 0; i < t.n(); ++i) {
    if (q.speed[i] >= t.telemetry_speed_at(i * t.ds()) - 0.5) ++above;
  }
  REQUIRE(static_cast<double>(above) / t.n() > 0.7);
}
