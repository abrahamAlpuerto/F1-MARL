// Track geometry: the reference line, its curvature, and the Frenet frame.
//
// Built by tools/build_track.py from real FastF1 position telemetry. See that
// script for why the geometry needs care -- in short, the source file carries
// two distance coordinates that drift against each other, and the naive reading
// invents corners that were never driven.
//
// One property is relied on throughout: `kappa` is exactly the derivative of
// the heading of the stored line, because the line was reconstructed by
// integrating that heading. So the curvature the physics integrates and the
// geometry the renderer draws cannot disagree.

#pragma once

#include <string>
#include <vector>

namespace racing {

struct Frenet {
  double s;      // arc length along the reference line, metres, wrapped to [0, L)
  double e_y;    // signed lateral offset, metres; positive is left of the line
  double e_psi;  // heading error against the line's tangent, radians
};

class Track {
 public:
  static Track load(const std::string& path);

  double length() const { return length_; }
  double ds() const { return ds_; }
  int n() const { return static_cast<int>(x_.size()); }
  double reference_lap_time() const { return ref_lap_time_; }
  const std::string& name() const { return name_; }

  // Wrap an arc length into [0, length).
  double wrap_s(double s) const;

  // Shortest signed difference a - b around the loop, in [-L/2, L/2). Plain
  // subtraction is wrong anywhere near start/finish, which is exactly where
  // lap boundaries and most overtakes happen.
  double delta_s(double a, double b) const;

  // Linearly interpolated queries. All wrap.
  double kappa_at(double s) const;
  double half_width_at(double s) const;
  // Elevation. Carried through from the source telemetry and used by nothing
  // in the physics -- the vehicle model is flat. It is here because the
  // visualizer wants it, and deriving it from arc length is the engine's job
  // rather than the renderer's.
  double z_at(double s) const;

  // Road gradient as sin(slope), positive uphill. Costs the car m*g*sin, which
  // at Bahrain's modest elevation change is small but is the reason the climb
  // out of the last corner is slower than a flat-road model says.
  double grade_at(double s) const;
  void pose_at(double s, double* x, double* y, double* theta) const;

  // Frenet -> world.
  void to_world(double s, double e_y, double* x, double* y) const;

  // World -> Frenet. `s_hint` seeds a local search; pass the previous step's
  // arc length. The car moves at most ~1 m per physics step, so a short window
  // is enough and a full scan every step would dominate the step cost.
  Frenet project(double x, double y, double heading, double s_hint) const;

  // Same, but scans the whole track. For resets, and for any case where no
  // trustworthy hint exists.
  Frenet project_global(double x, double y, double heading) const;

  // The real speed trace from the source lap, on the same grid. Not used by the
  // physics -- it is the calibration target (tools/calibrate_vehicle.py) and a
  // reference for the visualizer.
  double telemetry_speed_at(double s) const;

  const std::vector<double>& xs() const { return x_; }
  const std::vector<double>& ys() const { return y_; }
  const std::vector<double>& kappas() const { return kappa_; }
  const std::vector<double>& zs() const { return z_; }
  const std::vector<double>& sector_s() const { return sector_s_; }

 private:
  Frenet refine(double x, double y, double heading, int i) const;

  std::string name_;
  double length_ = 0.0;
  double ds_ = 0.0;
  double ref_lap_time_ = 0.0;
  std::vector<double> x_, y_, z_, theta_, kappa_, nx_, ny_;
  std::vector<double> half_left_, half_right_;
  std::vector<double> tel_speed_;
  std::vector<double> sector_s_;
};

}  // namespace racing
