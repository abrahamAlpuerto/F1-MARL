#include "racing/interaction.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace racing {

namespace {

// Relative velocity of a car resolved into the track frame at its own arc
// length: along the reference line, and to the left of it.
struct TrackVel {
  double along;
  double lateral;
  double e_psi;  // heading of the car relative to the line, carried through so
                 // an impulse in the track frame can be put back into the body
                 // frame exactly rather than assuming the two are aligned
};

TrackVel track_velocity(const Track& track, const CarState& c) {
  double th = 0.0;
  track.pose_at(c.f.s, nullptr, nullptr, &th);

  const double cp = std::cos(c.v.psi), sp = std::sin(c.v.psi);
  const double vx_w = c.v.vx * cp - c.v.vy * sp;
  const double vy_w = c.v.vx * sp + c.v.vy * cp;

  TrackVel t;
  t.along = vx_w * std::cos(th) + vy_w * std::sin(th);
  t.lateral = -vx_w * std::sin(th) + vy_w * std::cos(th);
  t.e_psi = c.v.psi - th;
  return t;
}

// Add `dv` metres per second along the track's left normal to a car, in the
// car's own body frame. The two frames differ by the heading error, so this is
// a rotation rather than a straight assignment -- getting it wrong would leak
// lateral impulses into forward speed, which at a hairpin is a free launch.
void add_lateral_velocity(CarState* c, double dv, double e_psi) {
  c->v.vx += dv * std::sin(e_psi);
  c->v.vy += dv * std::cos(e_psi);
}

void add_longitudinal_velocity(CarState* c, double dv, double e_psi) {
  c->v.vx += dv * std::cos(e_psi);
  c->v.vy -= dv * std::sin(e_psi);
}

// Move a car sideways in the track frame by `dy` metres.
void shift_lateral(const Track& track, CarState* c, double dy) {
  double th = 0.0;
  track.pose_at(c->f.s, nullptr, nullptr, &th);
  c->v.x += dy * -std::sin(th);
  c->v.y += dy * std::cos(th);
  c->f.e_y += dy;
}

// Closing speed at which a touch counts as a full-severity hit. 8 m/s of
// relative motion between two cars is already a substantial accident; most
// racing contact is well under 2.
constexpr double kSeverityScale = 8.0;

}  // namespace

// --- wake ------------------------------------------------------------------

double wake_strength(const AeroConfig& cfg, double gap, double lateral_offset) {
  if (gap <= 0.0 || gap > cfg.range) return 0.0;
  const double along = std::exp(-gap / std::max(cfg.decay_length, 1e-6));
  const double y = lateral_offset / std::max(cfg.width, 1e-6);
  return along * std::exp(-y * y);
}

void apply_wake(const AeroConfig& cfg, const Track& track,
                std::vector<CarState>* cars) {
  const int n = static_cast<int>(cars->size());

  for (int i = 0; i < n; ++i) {
    CarState& me = (*cars)[i];
    me.wake = 0.0;
    me.wake_source = -1;
    me.downforce_factor = 1.0;
    me.drag_factor = 1.0;
  }
  if (!cfg.enabled || n < 2) return;

  for (int i = 0; i < n; ++i) {
    CarState& me = (*cars)[i];
    if (me.retired) continue;

    double best = 0.0;
    int source = -1;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      const CarState& other = (*cars)[j];
      if (other.retired) continue;

      // Positive when `other` is ahead of `me` along the line. delta_s wraps,
      // so this is still right on either side of the start/finish line.
      const double gap = track.delta_s(other.f.s, me.f.s);
      const double w = wake_strength(cfg, gap, me.f.e_y - other.f.e_y);
      if (w > best) {
        best = w;
        source = j;
      }
    }

    me.wake = best;
    me.wake_source = source;
    me.downforce_factor = 1.0 - cfg.wash_max * best;
    me.drag_factor = 1.0 - cfg.tow_max * best;
  }
}

// --- contact ---------------------------------------------------------------

void resolve_contacts(const ContactConfig& cfg, const VehicleParams& vp,
                      const Track& track, double dt,
                      std::vector<CarState>* cars, std::vector<Contact>* out) {
  const int n = static_cast<int>(cars->size());
  for (int i = 0; i < n; ++i) {
    (*cars)[i].contact = false;
    (*cars)[i].contact_severity = 0.0;
  }
  if (!cfg.enabled || n < 2) return;

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      CarState& a = (*cars)[i];
      CarState& b = (*cars)[j];
      if (a.retired || b.retired) continue;

      // Separation in the track frame. `ds` positive means a is ahead of b.
      const double ds = track.delta_s(a.f.s, b.f.s);
      const double dy = a.f.e_y - b.f.e_y;

      const double pen_long = vp.length - std::abs(ds);
      const double pen_lat = vp.width - std::abs(dy);
      if (pen_long <= 0.0 || pen_lat <= 0.0) continue;

      const TrackVel va = track_velocity(track, a);
      const TrackVel vb = track_velocity(track, b);

      // The contact normal is the axis of least penetration -- the standard
      // separating-axis choice. Two cars side by side are pushed apart
      // sideways; one that has run into the back of another is pushed apart
      // along the road. Picking the wrong axis is how a rear-end ends up
      // firing both cars off the circuit sideways.
      const bool lateral = pen_lat < pen_long;

      double severity;
      if (lateral) {
        const double sign = dy >= 0.0 ? 1.0 : -1.0;  // a is left of b
        const double closing = sign * (vb.lateral - va.lateral);
        severity = std::clamp(std::abs(closing) / kSeverityScale, 0.0, 1.0);

        const double push = 0.5 * cfg.separation_gain * pen_lat * sign;
        shift_lateral(track, &a, push);
        shift_lateral(track, &b, -push);

        if (closing > 0.0) {
          const double dv = 0.5 * (1.0 + cfg.restitution) * closing;
          add_lateral_velocity(&a, dv * sign, va.e_psi);
          add_lateral_velocity(&b, -dv * sign, vb.e_psi);
        }

        // A glancing blow rotates each car away from the other. A rate, so a
        // long scrape does not accumulate into a pirouette.
        const double kick = cfg.yaw_kick * severity * sign * dt;
        a.v.r += kick;
        b.v.r -= kick;
      } else {
        const double sign = ds >= 0.0 ? 1.0 : -1.0;  // a is ahead of b
        const double closing = sign * (vb.along - va.along);
        severity = std::clamp(std::abs(closing) / kSeverityScale, 0.0, 1.0);

        // Along-track separation is applied as a velocity correction rather
        // than by teleporting one car up the road: moving a car forward in arc
        // length would hand it free race distance, and the whole classification
        // is built on distance covered.
        if (closing > 0.0) {
          const double dv = 0.5 * (1.0 + cfg.restitution) * closing;
          add_longitudinal_velocity(&a, dv * sign, va.e_psi);
          add_longitudinal_velocity(&b, -dv * sign, vb.e_psi);
        }
      }

      // Both cars lose speed, at a rate for the length of the step. The
      // engine does not try to work out whose fault it was -- it genuinely
      // cannot, and a wrong answer would teach a policy that some contact is
      // free. Penalising both is the honest version and makes the field keep
      // its distance.
      const double scrub = std::max(0.0, 1.0 - cfg.speed_loss * severity * dt);
      a.v.vx = std::max(0.0, a.v.vx * scrub);
      b.v.vx = std::max(0.0, b.v.vx * scrub);

      a.contact = true;
      b.contact = true;
      a.contact_severity = std::max(a.contact_severity, severity);
      b.contact_severity = std::max(b.contact_severity, severity);

      if (out) {
        Contact c;
        // `a` is reported as the car behind, which is the one a viewer will
        // read as having arrived.
        c.a = ds >= 0.0 ? b.index : a.index;
        c.b = ds >= 0.0 ? a.index : b.index;
        c.severity = severity;
        out->push_back(c);
      }
    }
  }
}

// --- classification --------------------------------------------------------

void classify(std::vector<CarState>* cars) {
  const int n = static_cast<int>(cars->size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);

  const std::vector<CarState>& c = *cars;
  std::stable_sort(order.begin(), order.end(), [&](int i, int j) {
    // Retired cars go to the back regardless of how far they got.
    if (c[i].retired != c[j].retired) return c[j].retired;
    // Anyone who has taken the flag is ahead of anyone still running, and
    // among those the earlier finisher wins. Distance alone cannot separate
    // two cars that both covered exactly the race distance.
    if (c[i].finished != c[j].finished) return c[i].finished;
    if (c[i].finished && c[j].finished) return c[i].finish_time < c[j].finish_time;
    return c[i].distance > c[j].distance;
  });

  for (int p = 0; p < n; ++p) {
    (*cars)[order[p]].position = p + 1;
  }

  // Gaps to the car classified either side, in seconds. Distance over speed is
  // the usual approximation and is what a timing screen shows; it goes wrong
  // only when the car it is measured against is nearly stopped, which is why
  // the divisor is floored.
  for (int p = 0; p < n; ++p) {
    CarState& me = (*cars)[order[p]];
    const double v = std::max(me.v.speed(), 5.0);
    me.gap_ahead = p > 0
        ? std::max(0.0, (*cars)[order[p - 1]].distance - me.distance) / v
        : 0.0;
    me.gap_behind = p + 1 < n
        ? std::max(0.0, me.distance - (*cars)[order[p + 1]].distance) / v
        : 0.0;
  }
}

// --- DRS -------------------------------------------------------------------

std::vector<DrsZone> find_drs_zones(const Track& track, const DrsConfig& cfg) {
  std::vector<DrsZone> zones;
  if (!cfg.enabled) return zones;

  const double L = track.length();
  const double step = std::max(track.ds(), 1.0);
  const int n = static_cast<int>(L / step);
  if (n < 8) return zones;

  // Mark every sample that is straight enough to count.
  std::vector<char> straight(n, 0);
  for (int i = 0; i < n; ++i) {
    straight[i] = std::abs(track.kappa_at(i * step)) <= cfg.zone_max_curvature;
  }

  // Walk the loop once, starting from a sample that is NOT straight so a run
  // spanning the start/finish line is found whole rather than as two stubs.
  int origin = 0;
  while (origin < n && straight[origin]) ++origin;
  if (origin == n) return zones;  // the whole circuit is straight: not a circuit

  int run_start = -1;
  for (int k = 0; k <= n; ++k) {
    const int i = (origin + k) % n;
    const bool is_straight = k < n && straight[i];
    if (is_straight && run_start < 0) {
      run_start = k;
    } else if (!is_straight && run_start >= 0) {
      const double length = (k - run_start) * step;
      if (length >= cfg.zone_min_length_m) {
        DrsZone z;
        z.start_s = track.wrap_s((origin + run_start) * step);
        z.end_s = track.wrap_s((origin + k) * step);
        // The detection point sits before the zone, so the gap is measured
        // where the cars are still in the previous corner -- which is the
        // whole point of the rule. Following closely through a corner is what
        // earns the wing, not being close once the straight has already
        // started.
        z.detection_s = track.wrap_s(z.start_s - cfg.detection_before_m);
        zones.push_back(z);
      }
      run_start = -1;
    }
  }
  return zones;
}

int zone_at(const std::vector<DrsZone>& zones, const Track& track, double s) {
  for (size_t i = 0; i < zones.size(); ++i) {
    // delta_s wraps, so this is still right for a zone that straddles the
    // start/finish line.
    const double from_start = track.delta_s(s, zones[i].start_s);
    const double span = track.delta_s(zones[i].end_s, zones[i].start_s);
    if (from_start >= 0.0 && span > 0.0 && from_start < span) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

double gap_to_car_ahead(const Track& track, const std::vector<CarState>& cars,
                        int me, double max_gap_m) {
  const CarState& c = cars[me];
  double nearest = max_gap_m;
  for (size_t j = 0; j < cars.size(); ++j) {
    if (static_cast<int>(j) == me || cars[j].retired) continue;
    const double gap = track.delta_s(cars[j].f.s, c.f.s);
    if (gap > 0.0 && gap < nearest) nearest = gap;
  }
  if (nearest >= max_gap_m) return 1e9;  // nobody in range
  return nearest / std::max(c.v.speed(), 5.0);
}

}  // namespace racing
