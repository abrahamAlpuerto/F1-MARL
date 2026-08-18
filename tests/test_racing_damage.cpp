// Damage, barriers, and the DNF.
//
// The property that matters most here is the BOUNDARY, and it is the reason
// this file exists separately from test_racing_race.cpp: running wide has to
// stay survivable while hitting a wall has to end a race. Those two are a
// couple of metres apart on the same excursion, and a model that gets the
// boundary wrong is either a demolition derby or has no consequences at all.

#include <cmath>
#include <memory>
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

EnvConfig one_car() {
  EnvConfig c;
  c.track.path = track_path();
  c.field.n_teams = 1;
  c.field.cars_per_team = 1;
  c.track.randomize_start = false;
  c.track.episode_distance = 4000.0;
  c.reward.terminate_off_track = false;
  c.seed = 11;
  return c;
}

std::unique_ptr<RaceEnv> make(const EnvConfig& c) {
  auto track = std::make_shared<const Track>(Track::load(c.track.path));
  return std::make_unique<RaceEnv>(c, track, 0);
}

// Two cars side by side on a straight, closing on each other at `closing` m/s.
// Enough to drive resolve_contacts without a whole race around it.
std::vector<CarState> side_by_side(const Track& t, double closing, double s0) {
  std::vector<CarState> cars(2);
  for (int i = 0; i < 2; ++i) {
    cars[i].index = i;
    cars[i].f.s = s0;
    cars[i].v.vx = 60.0;
  }
  // A metre apart laterally: inside the contact box, which is a car's width.
  cars[0].f.e_y = 0.5;
  cars[1].f.e_y = -0.5;

  double th = 0.0;
  for (int i = 0; i < 2; ++i) {
    double x, y;
    t.pose_at(s0, &x, &y, &th);
    t.to_world(s0, cars[i].f.e_y, &x, &y);
    cars[i].v.x = x;
    cars[i].v.y = y;
    cars[i].v.psi = th;
  }
  // Closing on each other, split between them.
  cars[0].v.vy = -0.5 * closing;
  cars[1].v.vy = 0.5 * closing;
  return cars;
}

}  // namespace

// --- the threshold ---------------------------------------------------------

TEST_CASE("light contact costs nothing", "[racing][damage]") {
  // Rubbing is most of what close racing is. If it damages cars, the only
  // stable policy is to leave a car's width everywhere, and the racing stops.
  const Track t = Track::load(track_path());
  ContactConfig cc;
  DamageConfig dmg;
  VehicleParams vp;

  auto cars = side_by_side(t, 0.5, 1500.0);  // 0.5 m/s: a brush
  std::vector<Contact> out;
  for (int k = 0; k < 50; ++k) {
    resolve_contacts(cc, dmg, vp, t, 0.01, &cars, &out);
  }

  REQUIRE(cars[0].contact_severity < dmg.contact_threshold);
  REQUIRE(cars[0].damage == Approx(0.0));
  REQUIRE(cars[1].damage == Approx(0.0));
}

TEST_CASE("damage from contact is a rate, not a per-touch charge",
          "[racing][damage]") {
  // The reason the whole model is written this way. Severity reported for a
  // contact EVENT is the worst single physics step of the touch, and that
  // distribution is saturated -- so anything keyed on it charges a one-step
  // transient the same as a genuine shunt. Twice the contact time has to cost
  // about twice the damage.
  const Track t = Track::load(track_path());
  ContactConfig cc;
  DamageConfig dmg;
  VehicleParams vp;
  std::vector<Contact> out;

  auto brief = side_by_side(t, 8.0, 1500.0);
  resolve_contacts(cc, dmg, vp, t, 0.01, &brief, &out);
  const double one_step = brief[0].damage;
  REQUIRE(one_step > 0.0);

  auto longer = side_by_side(t, 8.0, 1500.0);
  resolve_contacts(cc, dmg, vp, t, 0.02, &longer, &out);
  REQUIRE(longer[0].damage == Approx(2.0 * one_step).epsilon(0.01));
}

TEST_CASE("sustained heavy contact writes a car off", "[racing][damage]") {
  // The mechanism the rate exists to allow: cars locked together at high
  // severity for long enough do reach the terminal threshold. Rare in a race,
  // because it needs the two to stay interlocked, but it has to be reachable or
  // "collision" is not a retirement reason at all.
  const Track t = Track::load(track_path());
  ContactConfig cc;
  DamageConfig dmg;
  VehicleParams vp;
  std::vector<Contact> out;

  auto cars = side_by_side(t, 8.0, 1500.0);
  for (int k = 0; k < 200; ++k) {
    // Re-impose the closing velocity: the solver pushes them apart, and what is
    // being tested is the damage law, not whether two cars can stay locked.
    cars[0].v.vy = -4.0;
    cars[1].v.vy = 4.0;
    cars[0].f.e_y = 0.5;
    cars[1].f.e_y = -0.5;
    resolve_contacts(cc, dmg, vp, t, 0.01, &cars, &out);
  }
  REQUIRE(cars[0].damage >= dmg.retire_threshold);
}

TEST_CASE("damage never exceeds the terminal threshold", "[racing][damage]") {
  // `damage` is published in the feed as 0 to 1. A car that has already taken
  // the flag keeps being simulated to the end of the race and can keep taking
  // knocks, so without a cap the field would run past 1 and the documented
  // range would be a lie.
  const Track t = Track::load(track_path());
  ContactConfig cc;
  DamageConfig dmg;
  VehicleParams vp;
  std::vector<Contact> out;

  auto cars = side_by_side(t, 8.0, 1500.0);
  for (int k = 0; k < 2000; ++k) {
    cars[0].v.vy = -4.0;
    cars[1].v.vy = 4.0;
    cars[0].f.e_y = 0.5;
    cars[1].f.e_y = -0.5;
    resolve_contacts(cc, dmg, vp, t, 0.01, &cars, &out);
  }
  REQUIRE(cars[0].damage == Approx(dmg.retire_threshold));
  REQUIRE(cars[1].damage == Approx(dmg.retire_threshold));
}

// --- the barrier -----------------------------------------------------------

TEST_CASE("the barrier sits beyond the recovery distance", "[racing][damage]") {
  // Load-bearing ordering, not a style preference. A car is recovered once it
  // is more than `recover_distance` off the circuit; if the wall were inside
  // that, a car pinned against it could never get far enough out to satisfy the
  // test, and would sit there for the rest of the race.
  EnvConfig c;
  REQUIRE(c.damage.run_off_width > c.race.recover_distance);
}

TEST_CASE("a gentle touch of the wall is survivable", "[racing][damage]") {
  const Track t = Track::load(track_path());
  DamageConfig dmg;
  std::vector<CarState> cars(1);
  cars[0].index = 0;

  const double s0 = 1500.0;
  const double limit = t.half_width_at(s0) + dmg.run_off_width;
  cars[0].f.s = s0;
  cars[0].f.e_y = limit + 0.2;
  double x, y, th;
  t.pose_at(s0, &x, &y, &th);
  t.to_world(s0, cars[0].f.e_y, &x, &y);
  cars[0].v.x = x;
  cars[0].v.y = y;
  cars[0].v.psi = th;
  cars[0].v.vx = 10.0;
  cars[0].v.vy = 2.0;  // 2 m/s into the wall: a car leaning on it

  std::vector<BarrierHit> hits;
  resolve_barriers(dmg, t, &cars, &hits);

  REQUIRE(hits.size() == 1);
  REQUIRE(hits[0].severity < dmg.barrier_threshold);
  REQUIRE(cars[0].damage == Approx(0.0));
  REQUIRE_FALSE(cars[0].retired);
}

TEST_CASE("hitting the wall hard ends a race", "[racing][damage]") {
  const Track t = Track::load(track_path());
  DamageConfig dmg;
  std::vector<CarState> cars(1);
  cars[0].index = 0;

  const double s0 = 1500.0;
  const double limit = t.half_width_at(s0) + dmg.run_off_width;
  cars[0].f.s = s0;
  cars[0].f.e_y = limit + 0.5;
  double x, y, th;
  t.pose_at(s0, &x, &y, &th);
  t.to_world(s0, cars[0].f.e_y, &x, &y);
  cars[0].v.x = x;
  cars[0].v.y = y;
  cars[0].v.psi = th;
  cars[0].v.vx = 40.0;
  cars[0].v.vy = dmg.impact_speed_full;  // straight in, at full-severity speed

  std::vector<BarrierHit> hits;
  resolve_barriers(dmg, t, &cars, &hits);

  REQUIRE(hits.size() == 1);
  REQUIRE(hits[0].severity == Approx(1.0));
  REQUIRE(cars[0].damage >= dmg.retire_threshold);
  // And it is stopped by the wall rather than carrying on through it.
  REQUIRE(cars[0].v.vx < 40.0);
  REQUIRE(std::abs(cars[0].f.e_y) <= limit + 1e-6);
}

TEST_CASE("resting against the wall is not charged again", "[racing][damage]") {
  // Only impacts cost anything. A car pressed against a barrier, or sliding
  // along one, has no normal speed left to give -- so it must not be billed
  // once per physics step for continuing to be there.
  const Track t = Track::load(track_path());
  DamageConfig dmg;
  std::vector<CarState> cars(1);
  cars[0].index = 0;

  const double s0 = 1500.0;
  const double limit = t.half_width_at(s0) + dmg.run_off_width;
  cars[0].f.s = s0;
  cars[0].f.e_y = limit + 0.3;
  double x, y, th;
  t.pose_at(s0, &x, &y, &th);
  t.to_world(s0, cars[0].f.e_y, &x, &y);
  cars[0].v.x = x;
  cars[0].v.y = y;
  cars[0].v.psi = th;
  cars[0].v.vx = 30.0;
  cars[0].v.vy = 12.0;

  std::vector<BarrierHit> hits;
  resolve_barriers(dmg, t, &cars, &hits);
  const double after_impact = cars[0].damage;
  REQUIRE(after_impact > 0.0);

  // Now it is against the wall and moving away from it. Further steps are free.
  for (int k = 0; k < 20; ++k) resolve_barriers(dmg, t, &cars, &hits);
  REQUIRE(cars[0].damage == Approx(after_impact));
}

// --- what a retirement does to the race ------------------------------------

TEST_CASE("a retired car stops and stays stopped", "[racing][damage]") {
  EnvConfig c = one_car();
  auto env = make(c);
  env->reset(0);

  // Full lock and full throttle from a rolling start puts this car across the
  // run-off and into the wall well inside a few seconds.
  std::vector<float> act{1.0f, 1.0f};
  std::vector<float> rew(1);

  int retire_events = 0;
  int retired_at = -1;
  for (int k = 0; k < 600 && !env->done(); ++k) {
    env->step(act.data(), rew.data(), nullptr);
    for (const RaceEvent& e : env->events()) {
      if (e.type == RaceEvent::RETIRE) {
        ++retire_events;
        REQUIRE(e.reason == RETIRE_BARRIER);
        REQUIRE(e.value >= c.damage.retire_threshold);
      }
    }
    if (env->cars()[0].retired && retired_at < 0) retired_at = k;
  }

  REQUIRE(retired_at >= 0);
  // Reported once, not once per step for the rest of the race.
  REQUIRE(retire_events == 1);

  const CarState& car = env->cars()[0];
  REQUIRE(car.retire_reason == RETIRE_BARRIER);
  REQUIRE(car.v.speed() == Approx(0.0));
  REQUIRE_FALSE(car.drs_open);

  // And it does not creep. Where it stopped is where it stays.
  const double x = car.v.x, y = car.v.y;
  for (int k = 0; k < 20 && !env->done(); ++k) {
    env->step(act.data(), rew.data(), nullptr);
  }
  REQUIRE(env->cars()[0].v.x == Approx(x));
  REQUIRE(env->cars()[0].v.y == Approx(y));
}

TEST_CASE("running wide is still survivable with damage on",
          "[racing][damage][limits]") {
  // The boundary, from the other side. A car that leaves the corridor but does
  // not reach the wall is penalised and recovered, exactly as before damage
  // existed. If this ever fails, the run-off has become too narrow to have a
  // moment in and the racing is the poorer for it.
  EnvConfig c = one_car();
  c.damage.enabled = true;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{0.0f, 0.0f};
  std::vector<float> rew(1);

  // Coast, then a brief steering input: enough to run off the road, not enough
  // to cross thirty metres of run-off.
  bool went_off = false;
  for (int k = 0; k < 400 && !env->done(); ++k) {
    act[0] = k >= 20 && k < 34 ? 0.85f : 0.0f;
    act[1] = k < 20 ? 0.3f : -0.2f;
    env->step(act.data(), rew.data(), nullptr);
    if (env->cars()[0].off_track) went_off = true;
  }

  REQUIRE(went_off);
  REQUIRE_FALSE(env->cars()[0].retired);
}

TEST_CASE("retirement is charged to the reward exactly once",
          "[racing][damage][reward]") {
  EnvConfig c = one_car();
  c.reward.retire_penalty = 100.0;  // large enough to find in the noise
  c.reward.finish_weight = 0.0;     // and not confused with the end-of-race term
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{1.0f, 1.0f};
  std::vector<float> rew(1);

  int big_negatives = 0;
  bool seen_retired = false;
  for (int k = 0; k < 300 && !env->done(); ++k) {
    env->step(act.data(), rew.data(), nullptr);
    if (rew[0] < -50.0) ++big_negatives;
    if (env->cars()[0].retired) seen_retired = true;
  }

  REQUIRE(seen_retired);
  REQUIRE(big_negatives == 1);
}

// --- determinism -----------------------------------------------------------

TEST_CASE("damage and retirements replay exactly", "[racing][damage][determinism]") {
  // The engine's whole discipline: a race that is not bit-exact on replay makes
  // every recorded episode worthless. Damage is accumulated state, so it is
  // exactly the kind of thing that drifts if anything about the order of
  // operations depends on memory layout.
  EnvConfig c;
  c.track.path = track_path();
  c.field.n_teams = 4;
  c.field.cars_per_team = 2;
  c.track.episode_distance = 3000.0;
  c.seed = 99;

  auto run = [&]() {
    auto env = make(c);
    env->reset(7);
    std::vector<float> act(env->n_cars() * 2);
    std::vector<float> rew(env->n_cars());
    for (int i = 0; i < env->n_cars(); ++i) {
      act[i * 2 + 0] = 0.4f * static_cast<float>((i % 3) - 1);
      act[i * 2 + 1] = 0.8f;
    }
    for (int k = 0; k < 400 && !env->done(); ++k) {
      env->step(act.data(), rew.data(), nullptr);
    }
    std::vector<double> out;
    for (const CarState& car : env->cars()) {
      out.push_back(car.damage);
      out.push_back(car.retired ? 1.0 : 0.0);
      out.push_back(static_cast<double>(car.retire_reason));
      out.push_back(car.v.x);
      out.push_back(car.v.y);
    }
    return out;
  };

  const std::vector<double> a = run();
  const std::vector<double> b = run();
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == b[i]);  // bit-exact, not approximately
  }
}

TEST_CASE("damage can be switched off entirely", "[racing][damage]") {
  // Training phase 1 and every test that is not about damage rely on this.
  const Track t = Track::load(track_path());
  DamageConfig dmg;
  dmg.enabled = false;

  std::vector<CarState> cars(1);
  cars[0].index = 0;
  const double s0 = 1500.0;
  cars[0].f.s = s0;
  cars[0].f.e_y = t.half_width_at(s0) + dmg.run_off_width + 5.0;
  cars[0].v.vy = 30.0;

  std::vector<BarrierHit> hits;
  resolve_barriers(dmg, t, &cars, &hits);
  REQUIRE(hits.empty());
  REQUIRE(cars[0].damage == Approx(0.0));
}
