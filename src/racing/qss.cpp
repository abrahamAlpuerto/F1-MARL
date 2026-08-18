#include "racing/qss.hpp"

#include <algorithm>
#include <cmath>

namespace racing {

double lap_time_from_profile(const Track& track,
                             const std::vector<double>& speed) {
  const int n = static_cast<int>(speed.size());
  const double ds = track.ds();
  double t = 0.0;
  for (int i = 0; i < n; ++i) {
    // Trapezoidal in 1/v rather than using a mean speed: over a braking zone
    // the two differ by enough to matter at the third decimal of a lap time,
    // which is the precision the calibration is judged at.
    const double v0 = std::max(speed[i], 1.0);
    const double v1 = std::max(speed[(i + 1) % n], 1.0);
    t += ds * 0.5 * (1.0 / v0 + 1.0 / v1);
  }
  return t;
}

QssResult solve_qss(const Track& track, const Vehicle& vehicle,
                    int max_iterations) {
  const int n = track.n();
  const double ds = track.ds();
  const auto& kappa = track.kappas();

  QssResult res;
  res.speed.assign(n, 0.0);

  // Start from the cornering limit everywhere. Everything after this can only
  // lower it.
  std::vector<double> v_corner(n);
  for (int i = 0; i < n; ++i) {
    v_corner[i] = vehicle.max_corner_speed(kappa[i]);
  }
  res.speed = v_corner;

  // The track is a loop, so neither pass has a natural starting speed. Iterate
  // both until the profile stops changing; on a circuit this settles in a
  // handful of passes because the slowest corner anchors it.
  std::vector<double> prev(n);
  for (int it = 0; it < max_iterations; ++it) {
    prev = res.speed;

    // Forward: how fast can the car still be going, having had to accelerate
    // out of what came before.
    for (int k = 0; k < n; ++k) {
      const int i = k % n;
      const int j = (k + 1) % n;
      const double v = res.speed[i];
      const double lat = v * v * std::abs(kappa[i]);
      const double a = vehicle.max_long_accel(v, lat);
      const double v_next = std::sqrt(std::max(0.0, v * v + 2.0 * a * ds));
      res.speed[j] = std::min(res.speed[j], v_next);
    }

    // Backward: and how fast could it have been going, given it has to be slow
    // enough for what comes next.
    for (int k = n; k > 0; --k) {
      const int i = k % n;
      const int j = (k - 1 + n) % n;
      const double v = res.speed[i];
      const double lat = v * v * std::abs(kappa[i]);
      const double a = std::abs(vehicle.max_long_decel(v, lat));
      const double v_prev = std::sqrt(std::max(0.0, v * v + 2.0 * a * ds));
      res.speed[j] = std::min(res.speed[j], v_prev);
    }

    double max_change = 0.0;
    for (int i = 0; i < n; ++i) {
      max_change = std::max(max_change, std::abs(res.speed[i] - prev[i]));
    }
    res.iterations = it + 1;
    if (max_change < 1e-6) {
      res.converged = true;
      break;
    }
  }

  res.lap_time = lap_time_from_profile(track, res.speed);
  return res;
}

}  // namespace racing
