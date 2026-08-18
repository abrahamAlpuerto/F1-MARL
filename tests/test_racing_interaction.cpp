// The wake and the contact model -- the two places cars stop being independent.
//
// These tests are about behaviour a viewer would notice, not about numerical
// detail: does a tow actually make a following car quicker on a straight, does
// dirty air actually cost it in a corner, and do two cars pushed together come
// apart again rather than through each other.

#include <cmath>
#include <string>
#include <vector>

#include "catch2/catch.hpp"
#include "racing/config.hpp"
#include "racing/interaction.hpp"
#include "racing/race.hpp"

using namespace racing;

namespace {

std::string track_path() {
  return std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
}

// Two cars, one behind the other, at a chosen gap and lateral offset.
std::vector<CarState> pair_at(const Track& t, double leader_s, double gap,
                              double lat_follower, double lat_leader,
                              double speed = 70.0) {
  std::vector<CarState> cars(2);
  const double follower_s = t.wrap_s(leader_s - gap);

  cars[0].index = 0;
  cars[0].team = 0;
  cars[0].f = Frenet{t.wrap_s(leader_s), lat_leader, 0.0};
  cars[0].v.vx = speed;
  t.pose_at(cars[0].f.s, &cars[0].v.x, &cars[0].v.y, &cars[0].v.psi);
  t.to_world(cars[0].f.s, lat_leader, &cars[0].v.x, &cars[0].v.y);

  cars[1].index = 1;
  cars[1].team = 1;
  cars[1].f = Frenet{follower_s, lat_follower, 0.0};
  cars[1].v.vx = speed;
  t.pose_at(cars[1].f.s, &cars[1].v.x, &cars[1].v.y, &cars[1].v.psi);
  t.to_world(cars[1].f.s, lat_follower, &cars[1].v.x, &cars[1].v.y);
  return cars;
}

}  // namespace

TEST_CASE("the wake is strongest directly behind and close", "[racing][aero]") {
  AeroConfig a;
  const double close = wake_strength(a, 5.0, 0.0);
  REQUIRE(close > 0.7);

  // Falls off with distance...
  REQUIRE(wake_strength(a, 40.0, 0.0) < close);
  REQUIRE(wake_strength(a, 40.0, 0.0) > 0.0);

  // ...and with lateral offset. Pulling out of line is the way out of it,
  // which is the property the whole overtaking mechanism rests on.
  REQUIRE(wake_strength(a, 5.0, 3.0) < 0.5 * close);
  REQUIRE(wake_strength(a, 5.0, 8.0) < 0.01);

  // A car ahead of you is not in your wake, and neither is one beyond range.
  REQUIRE(wake_strength(a, -10.0, 0.0) == 0.0);
  REQUIRE(wake_strength(a, a.range + 1.0, 0.0) == 0.0);
}

TEST_CASE("a follower loses downforce and gains a tow", "[racing][aero]") {
  const Track t = Track::load(track_path());
  AeroConfig a;

  std::vector<CarState> cars = pair_at(t, 1000.0, 8.0, 0.0, 0.0);
  apply_wake(a, t, &cars);

  // The leader is in clean air whatever is behind it.
  REQUIRE(cars[0].downforce_factor == Approx(1.0));
  REQUIRE(cars[0].drag_factor == Approx(1.0));
  REQUIRE(cars[0].wake == Approx(0.0));

  // The follower is not.
  REQUIRE(cars[1].wake > 0.5);
  REQUIRE(cars[1].downforce_factor < 1.0);
  REQUIRE(cars[1].drag_factor < 1.0);
  REQUIRE(cars[1].wake_source == 0);

  // Dirty air costs more than the tow gains, which is what makes following
  // through a corner sequence a real price rather than a free ride.
  const double lost_grip = 1.0 - cars[1].downforce_factor;
  const double gained = 1.0 - cars[1].drag_factor;
  REQUIRE(lost_grip > gained);
}

TEST_CASE("moving offline escapes the wake", "[racing][aero]") {
  const Track t = Track::load(track_path());
  AeroConfig a;

  std::vector<CarState> inline_pair = pair_at(t, 1000.0, 10.0, 0.0, 0.0);
  apply_wake(a, t, &inline_pair);

  std::vector<CarState> offset_pair = pair_at(t, 1000.0, 10.0, 5.0, 0.0);
  apply_wake(a, t, &offset_pair);

  REQUIRE(offset_pair[1].wake < 0.25 * inline_pair[1].wake);
  REQUIRE(offset_pair[1].downforce_factor > inline_pair[1].downforce_factor);
}

TEST_CASE("a car takes the strongest wake, not the sum of them",
          "[racing][aero]") {
  // Five cars nose to tail must not leave the last one with no downforce at
  // all. Summing wakes does exactly that and makes trains behave absurdly.
  const Track t = Track::load(track_path());
  AeroConfig a;

  std::vector<CarState> cars(5);
  for (int i = 0; i < 5; ++i) {
    cars[i].index = i;
    cars[i].f = Frenet{t.wrap_s(1000.0 - i * 8.0), 0.0, 0.0};
    cars[i].v.vx = 70.0;
    t.pose_at(cars[i].f.s, &cars[i].v.x, &cars[i].v.y, &cars[i].v.psi);
  }
  apply_wake(a, t, &cars);

  const double single = wake_strength(a, 8.0, 0.0);
  for (int i = 1; i < 5; ++i) {
    REQUIRE(cars[i].wake == Approx(single));
    REQUIRE(cars[i].downforce_factor > 1.0 - a.wash_max);
  }
}

TEST_CASE("the tow makes a following car measurably faster",
          "[racing][aero][behaviour]") {
  // The whole point, checked end to end: same car, same throttle, same stretch
  // of road, once alone and once in a tow. The one in the tow must arrive
  // faster, or nothing can ever overtake.
  EnvConfig cfg;
  cfg.track.path = track_path();
  cfg.field.n_teams = 2;
  cfg.field.cars_per_team = 1;
  cfg.track.episode_distance = 1500.0;
  cfg.contact.enabled = false;
  // The follower has to be IN the wake for there to be anything to measure.
  // The default grid staggers cars 2.6 m either side of the line, which puts
  // 5.2 m between them -- two wake widths, so the follower sits in clean air
  // and the test measures nothing. It passed anyway, on a wake of 0.005, until
  // the grid spacing changed and the margin vanished.
  cfg.track.grid_stagger = 0.0;
  cfg.track.grid_spacing = 12.0;

  auto top_speed_of_follower = [&](bool aero_on) {
    EnvConfig c = cfg;
    c.aero.enabled = aero_on;
    auto track = std::make_shared<const Track>(Track::load(c.track.path));
    RaceEnv env(c, track, 0);
    env.reset(0);

    float act[4] = {0.0f, 1.0f, 0.0f, 1.0f};
    float rew[2];
    double best = 0.0;
    for (int i = 0; i < 200; ++i) {
      env.step(act, rew, nullptr);
      best = std::max(best, env.cars()[1].v.speed());
    }
    return best;
  };

  const double with_tow = top_speed_of_follower(true);
  const double alone = top_speed_of_follower(false);
  REQUIRE(with_tow > alone);
}

TEST_CASE("overlapping cars are pushed apart, not through each other",
          "[racing][contact]") {
  const Track t = Track::load(track_path());
  ContactConfig cc;
  VehicleParams vp;

  // Side by side, overlapping laterally.
  std::vector<CarState> cars = pair_at(t, 1000.0, 1.0, 0.6, -0.6);
  const double before = std::abs(cars[0].f.e_y - cars[1].f.e_y);
  REQUIRE(before < vp.width);

  std::vector<Contact> out;
  resolve_contacts(cc, vp, t, 0.01, &cars, &out);

  REQUIRE(out.size() == 1);
  REQUIRE(cars[0].contact);
  REQUIRE(cars[1].contact);

  const double after = std::abs(cars[0].f.e_y - cars[1].f.e_y);
  REQUIRE(after > before);
}

TEST_CASE("cars that are not touching are left alone", "[racing][contact]") {
  const Track t = Track::load(track_path());
  ContactConfig cc;
  VehicleParams vp;

  // Far enough apart along the road.
  std::vector<CarState> cars = pair_at(t, 1000.0, 20.0, 0.0, 0.0);
  const double x0 = cars[0].v.x, x1 = cars[1].v.x;

  std::vector<Contact> out;
  resolve_contacts(cc, vp, t, 0.01, &cars, &out);

  REQUIRE(out.empty());
  REQUIRE_FALSE(cars[0].contact);
  REQUIRE_FALSE(cars[1].contact);
  REQUIRE(cars[0].v.x == x0);
  REQUIRE(cars[1].v.x == x1);
}

TEST_CASE("repeated contact resolution converges instead of exploding",
          "[racing][contact]") {
  // A soft response applied every physics step must settle. If it does not,
  // two cars that touch end up in orbit -- which is what a rigid impulse at
  // 100 Hz between two cars already at the grip limit will do.
  const Track t = Track::load(track_path());
  ContactConfig cc;
  VehicleParams vp;

  std::vector<CarState> cars = pair_at(t, 1000.0, 0.5, 0.3, -0.3);
  std::vector<Contact> out;
  for (int i = 0; i < 200; ++i) {
    resolve_contacts(cc, vp, t, 0.01, &cars, &out);
    for (const CarState& c : cars) {
      REQUIRE(std::isfinite(c.v.x));
      REQUIRE(std::isfinite(c.v.vx));
      REQUIRE(std::isfinite(c.v.r));
      REQUIRE(std::abs(c.f.e_y) < 100.0);
    }
  }
  // And they have actually separated.
  REQUIRE(std::abs(cars[0].f.e_y - cars[1].f.e_y) >= vp.width);
}

TEST_CASE("classification orders by distance and finish time",
          "[racing][classify]") {
  std::vector<CarState> cars(4);
  for (int i = 0; i < 4; ++i) {
    cars[i].index = i;
    cars[i].v.vx = 50.0;
  }
  cars[0].distance = 100.0;
  cars[1].distance = 300.0;
  cars[2].distance = 200.0;
  cars[3].distance = 50.0;

  classify(&cars);
  REQUIRE(cars[1].position == 1);
  REQUIRE(cars[2].position == 2);
  REQUIRE(cars[0].position == 3);
  REQUIRE(cars[3].position == 4);

  // The gap to the car in front, in seconds, at 50 m/s.
  REQUIRE(cars[2].gap_ahead == Approx(2.0));
  REQUIRE(cars[1].gap_ahead == Approx(0.0));

  // A finisher outranks anyone still running, and the earlier finisher wins
  // even though both covered the same distance.
  cars[0].finished = true;
  cars[0].finish_time = 90.0;
  cars[3].finished = true;
  cars[3].finish_time = 88.0;
  classify(&cars);
  REQUIRE(cars[3].position == 1);
  REQUIRE(cars[0].position == 2);

  // A retired car goes to the back however far it got.
  cars[1].retired = true;
  classify(&cars);
  REQUIRE(cars[1].position == 4);
}
