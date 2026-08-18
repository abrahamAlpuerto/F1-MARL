// Check the vehicle model against a real F1 lap, and fit the aero and grip
// parameters to it.
//
//     racing_calibrate                 report only
//     racing_calibrate --fit           fit, then report
//
// The argument for realism
// ------------------------
// The track was built from Verstappen's fastest lap at the 2024 Bahrain GP, and
// that lap took 92.608 s. Because the reference line IS the line he drove, a
// quasi-steady-state solve over its curvature is directly comparable: if the
// car model is right, the solver should land close to 92.608 s, and its speed
// profile should track the recorded one corner by corner.
//
// That is a much stronger claim than "the parameters look like F1 numbers",
// and it is falsifiable. The QSS solve is also an *optimistic* bound -- it
// assumes the car is on some limit at every instant, and it drives the racing
// line exactly -- so it should come out slightly quicker than the real lap
// rather than slower. A solve that is slower than reality means the model is
// wrong somewhere, not that the driver was exceptional.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "racing/config.hpp"
#include "racing/qss.hpp"
#include "racing/track.hpp"
#include "racing/vehicle.hpp"

using namespace racing;

namespace {

struct Fit {
  double lap_time = 0.0;
  double rms_speed_err = 0.0;   // m/s against the recorded trace
  double max_lat_g = 0.0;
  double max_accel_g = 0.0;
  double max_brake_g = 0.0;
  double top_speed = 0.0;
  double min_speed = 1e9;
};

Fit evaluate(const Track& track, const VehicleParams& p) {
  const Vehicle veh(p);
  const QssResult q = solve_qss(track, veh);

  Fit f;
  f.lap_time = q.lap_time;

  const int n = track.n();
  const double ds = track.ds();
  double sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v_real = track.telemetry_speed_at(i * ds);
    const double d = q.speed[i] - v_real;
    sq += d * d;
    f.top_speed = std::max(f.top_speed, q.speed[i]);
    f.min_speed = std::min(f.min_speed, q.speed[i]);

    const double lat = q.speed[i] * q.speed[i] * std::abs(track.kappas()[i]) / 9.81;
    f.max_lat_g = std::max(f.max_lat_g, lat);

    const int j = (i + 1) % n;
    const double a = (q.speed[j] * q.speed[j] - q.speed[i] * q.speed[i]) /
                     (2.0 * ds) / 9.81;
    f.max_accel_g = std::max(f.max_accel_g, a);
    f.max_brake_g = std::min(f.max_brake_g, a);
  }
  f.rms_speed_err = std::sqrt(sq / n);
  return f;
}

void report(const Track& track, const VehicleParams& p) {
  const Vehicle veh(p);
  const Fit f = evaluate(track, p);
  const QssResult q = solve_qss(track, veh);
  const double ref = track.reference_lap_time();
  const double ds = track.ds();
  const int n = track.n();

  std::printf("\n  --- lap ---\n");
  std::printf("    QSS lap time       : %8.3f s\n", f.lap_time);
  std::printf("    real reference lap : %8.3f s   (VER, 2024 Bahrain GP)\n", ref);
  std::printf("    delta              : %+8.3f s  (%+.2f %%)\n",
              f.lap_time - ref, 100.0 * (f.lap_time - ref) / ref);
  std::printf("    105%% gate target    : %8.3f s\n", ref * 1.05);
  std::printf("    speed trace RMS    : %8.2f m/s\n", f.rms_speed_err);

  std::printf("\n  --- model envelope vs real F1 ---\n");
  std::printf("    peak lateral       : %8.2f g   (real: ~5-6 g)\n", f.max_lat_g);
  std::printf("    peak acceleration  : %8.2f g   (real: ~1.5-2 g)\n", f.max_accel_g);
  std::printf("    peak braking       : %8.2f g   (real: ~5-6 g)\n", f.max_brake_g);
  std::printf("    top speed          : %8.1f kph (real trace: %.1f)\n",
              f.top_speed * 3.6, 301.0);
  std::printf("    minimum speed      : %8.1f kph (real trace: %.1f)\n",
              f.min_speed * 3.6, 66.6);

  std::printf("    aero efficiency    : %8.2f     (real: ~3.5-4.5)\n",
              p.cl_a / p.cd_a);

  std::printf("\n  --- downforce ---\n");
  for (double kph : {100.0, 200.0, 250.0, 300.0}) {
    const double v = kph / 3.6;
    const double df = veh.downforce_at(v);
    std::printf("    %5.0f kph : %7.0f N  (%.2fx car weight)   drag %6.0f N\n",
                kph, df, df / (p.mass * 9.81), veh.drag_at(v));
  }

  std::printf("\n  --- steady-state cornering ---\n");
  for (double R : {20.0, 30.0, 50.0, 100.0, 300.0}) {
    const double v = veh.max_corner_speed(1.0 / R);
    std::printf("    R = %4.0f m : %6.1f kph  (%.2f g lateral)\n", R, v * 3.6,
                v * v / R / 9.81);
  }

  // Compare at the recorded speed minima, which is where the corners actually
  // are. Using nominal corner markers instead would compare at whatever point
  // the marker happens to sit, which is not necessarily the apex.
  std::printf("\n  --- at each recorded apex ---\n");
  std::printf("    %8s %10s %10s %9s\n", "s (m)", "model", "real", "delta");
  int shown = 0;
  for (int i = 0; i < n && shown < 20; ++i) {
    const double vr = track.telemetry_speed_at(i * ds);
    bool is_min = true;
    for (int k = -8; k <= 8 && is_min; ++k) {
      const int j = ((i + k) % n + n) % n;
      if (track.telemetry_speed_at(j * ds) < vr - 1e-9) is_min = false;
    }
    if (!is_min) continue;
    std::printf("    %8.0f %7.1f kph %6.1f kph %+8.1f\n", i * ds,
                q.speed[i] * 3.6, vr * 3.6, (q.speed[i] - vr) * 3.6);
    ++shown;
    i += 8;
  }
}

// Coordinate descent over the handful of parameters that are genuinely
// uncertain. Everything else -- mass, wheelbase, weight distribution, steering
// lock -- is a regulation or geometry figure and is deliberately NOT fitted:
// letting the optimiser move those would buy a better lap time by inventing a
// car that does not exist.
VehicleParams fit(const Track& track, VehicleParams p) {
  struct Knob {
    const char* name;
    double VehicleParams::*field;
    double lo, hi;
  };
  // Bounds are physical, not generous. An unbounded fit to lap time alone finds
  // a degenerate answer: it pushed cd_a to 2.00 and load sensitivity to 0.05,
  // which matched 92.6 s exactly by pairing a car that corners far too well at
  // speed with one that is far too draggy on the straights. Two wrong numbers
  // cancelling is not a calibration. The give-away was aerodynamic efficiency
  // of 2.9 against a real 3.5-4.5.
  const Knob knobs[] = {
      {"cl_a", &VehicleParams::cl_a, 4.0, 6.5},
      {"cd_a", &VehicleParams::cd_a, 1.15, 1.75},
      {"mu_peak", &VehicleParams::mu_peak, 1.55, 2.10},
      {"max_power", &VehicleParams::max_power, 640000.0, 780000.0},
      {"mu_load_sensitivity", &VehicleParams::mu_load_sensitivity, 0.10, 0.22},
  };

  // Lap time alone cannot separate drag from grip -- many wrong pairs give the
  // right total. These three targets can be identified separately, so together
  // they pin the parameters down:
  //
  //   top speed      -> the power-to-drag balance, and almost nothing else
  //   minimum speed  -> mechanical grip, where there is no downforce to help
  //   trace RMS      -> everything in between, especially the fast corners that
  //                     separate cl_a from load sensitivity
  //
  // Lap time is kept as a term but no longer dominates; it is a consequence to
  // be checked rather than a target to be hit.
  const double v_top_real = 301.0 / 3.6;
  const double v_min_real = 66.6 / 3.6;
  auto loss = [&](const VehicleParams& c) {
    const Fit f = evaluate(track, c);
    const double dt = f.lap_time - track.reference_lap_time();
    const double d_top = f.top_speed - v_top_real;
    const double d_min = f.min_speed - v_min_real;
    return 1.0 * dt * dt + 0.6 * d_top * d_top + 0.6 * d_min * d_min +
           1.0 * f.rms_speed_err * f.rms_speed_err;
  };

  double best = loss(p);
  std::printf("  start loss %.4f\n", best);
  for (int sweep = 0; sweep < 12; ++sweep) {
    bool improved = false;
    for (const Knob& k : knobs) {
      const double step = 0.15 * (k.hi - k.lo) / (1.0 + sweep);
      for (int dir = -1; dir <= 1; dir += 2) {
        VehicleParams c = p;
        c.*(k.field) = std::clamp(p.*(k.field) + dir * step, k.lo, k.hi);
        const double l = loss(c);
        if (l < best - 1e-9) {
          best = l;
          p = c;
          improved = true;
        }
      }
    }
    std::printf("  sweep %2d: loss %8.4f   ClA %.3f  CdA %.3f  L/D %.2f  "
                "mu %.3f  P %.0f kW  ls %.3f\n",
                sweep, best, p.cl_a, p.cd_a, p.cl_a / p.cd_a, p.mu_peak,
                p.max_power / 1000.0, p.mu_load_sensitivity);
    if (!improved) break;
  }
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  std::string track_path = "data/tracks/bahrain.json";
  bool do_fit = false;
  std::string out_config;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fit") {
      do_fit = true;
    } else if (a == "--track" && i + 1 < argc) {
      track_path = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out_config = argv[++i];
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }

  try {
    const Track track = Track::load(track_path);
    std::printf("track: %s -- %.1f m over %d samples\n", track.name().c_str(),
                track.length(), track.n());

    VehicleParams p;
    std::printf("\n=== as authored ==========================================");
    report(track, p);

    if (do_fit) {
      std::printf("\n=== fitting ==============================================\n");
      p = fit(track, p);
      std::printf("\n=== fitted ===============================================");
      report(track, p);
      std::printf("\n  fitted parameters:\n");
      std::printf("    cl_a                = %.4f\n", p.cl_a);
      std::printf("    cd_a                = %.4f\n", p.cd_a);
      std::printf("    mu_peak             = %.4f\n", p.mu_peak);
      std::printf("    mu_load_sensitivity = %.4f\n", p.mu_load_sensitivity);
      std::printf("    max_power           = %.0f W\n", p.max_power);

      if (!out_config.empty()) {
        EnvConfig c;
        c.vehicle = p;
        c.track.path = track_path;
        FILE* f = std::fopen(out_config.c_str(), "w");
        if (!f) {
          std::fprintf(stderr, "cannot write %s\n", out_config.c_str());
          return 1;
        }
        const std::string text = c.to_json_string();
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
        std::printf("\n  wrote %s\n", out_config.c_str());
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
