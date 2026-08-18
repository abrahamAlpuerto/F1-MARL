// The race itself: the grid, positions, laps, the flag, and the reward that is
// supposed to make a car race rather than time trial.

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "catch2/catch.hpp"
#include "racing/config.hpp"
#include "racing/race.hpp"

using namespace racing;

namespace {

EnvConfig base_config() {
  EnvConfig c;
  c.track.path = std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
  c.seed = 4242;
  c.field.n_teams = 2;
  c.field.cars_per_team = 2;
  c.track.episode_distance = 800.0;
  return c;
}

std::unique_ptr<RaceEnv> make(const EnvConfig& c) {
  auto track = std::make_shared<const Track>(Track::load(c.track.path));
  return std::make_unique<RaceEnv>(c, track, 0);
}

}  // namespace

TEST_CASE("the grid is a queue, not a heap", "[racing][grid]") {
  EnvConfig c = base_config();
  c.field.n_teams = 4;
  c.field.cars_per_team = 2;
  auto env = make(c);
  env->reset(0);

  REQUIRE(env->n_cars() == 8);

  // Every car starts somewhere different, and inside the circuit.
  for (int i = 0; i < 8; ++i) {
    const CarState& car = env->cars()[i];
    REQUIRE(std::abs(car.f.e_y) < env->track().half_width_at(car.f.s));
    REQUIRE(car.v.speed() > 0.0);
    for (int j = i + 1; j < 8; ++j) {
      REQUIRE(env->track().delta_s(car.f.s, env->cars()[j].f.s) != Approx(0.0));
    }
  }

  // Positions on the grid are 1..n and each appears once.
  std::vector<int> seen(9, 0);
  for (const CarState& car : env->cars()) {
    REQUIRE(car.position >= 1);
    REQUIRE(car.position <= 8);
    ++seen[car.position];
  }
  for (int p = 1; p <= 8; ++p) REQUIRE(seen[p] == 1);
}

TEST_CASE("teams are assigned in blocks and interleaved on the grid",
          "[racing][grid]") {
  EnvConfig c = base_config();
  c.field.n_teams = 3;
  c.field.cars_per_team = 2;
  c.field.grid_order = FieldConfig::GRID_INTERLEAVE;
  auto env = make(c);
  env->reset(0);

  // Car index -> team is a simple block split, so a training loop can reshape
  // rewards by team without asking.
  REQUIRE(env->teams() == std::vector<int>{0, 0, 1, 1, 2, 2});

  // Interleaved, so the front row is not one team's.
  const CarState& pole = env->cars()[0];
  int front_row_teams = 0;
  for (const CarState& car : env->cars()) {
    if (car.position <= 3 && car.team != pole.team) ++front_row_teams;
  }
  REQUIRE(front_row_teams > 0);
}

TEST_CASE("everyone races to the same finish line", "[racing][race]") {
  // The car at the back of the grid has further to go than the car on pole, by
  // exactly the grid gap between them. Without that the flag would fall in
  // eight different places on the circuit.
  EnvConfig c = base_config();
  c.field.n_teams = 2;
  c.field.cars_per_team = 2;
  c.track.grid_spacing = 20.0;
  // The cars here are driven with a fixed throttle and no steering, so they
  // leave the circuit and are carried round by the recovery mechanism. That is
  // fine for a test about where the flag falls, but with barriers in the world
  // it is also a car repeatedly hitting a wall. Damage is not what is under
  // test, so it is off; see test_racing_damage.cpp for what happens when it
  // is on.
  c.damage.enabled = false;
  auto env = make(c);
  env->reset(0);

  // Drive everyone flat until they have all finished.
  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) act[i * 2 + 1] = 0.35f;
  std::vector<float> rew(env->n_cars());

  double first_s = -1.0;
  for (int t = 0; t < 4000 && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
    for (const RaceEvent& e : env->events()) {
      if (e.type != RaceEvent::FINISH) continue;
      const double s = env->cars()[e.car].f.s;
      if (first_s < 0.0) {
        first_s = s;
      } else {
        // All the flags fall within a few metres of the same point, which is
        // as tight as it gets when the check happens once per policy step.
        REQUIRE(std::abs(env->track().delta_s(s, first_s)) < 12.0);
      }
    }
  }
  REQUIRE(first_s >= 0.0);
}

TEST_CASE("laps are counted and timed", "[racing][race]") {
  EnvConfig c = base_config();
  c.field.n_teams = 1;
  c.field.cars_per_team = 1;
  c.track.episode_distance = -1.0;
  c.track.laps = 2;
  c.damage.enabled = false;  // as above: a lap-counting test, not a crash test
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{0.0f, 0.30f};
  std::vector<float> rew(1);
  int laps_seen = 0;
  for (int t = 0; t < 20000 && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
    for (const RaceEvent& e : env->events()) {
      if (e.type == RaceEvent::LAP) {
        ++laps_seen;
        REQUIRE(e.lap == laps_seen);
        REQUIRE(e.value > 30.0);   // a lap here cannot be quicker than this
        REQUIRE(e.value < 400.0);
      }
    }
  }
  REQUIRE(laps_seen >= 1);
  REQUIRE(env->cars()[0].best_lap_time > 0.0);
}

TEST_CASE("gaining a place pays and losing one costs", "[racing][reward]") {
  // The position term is what turns a field of time trials into a race, so it
  // is worth checking it actually fires. Team weight is off here so the effect
  // is not diluted by a team mate.
  EnvConfig c = base_config();
  c.field.n_teams = 4;
  c.field.cars_per_team = 1;
  c.reward.team_weight = 0.0;
  c.reward.progress_weight = 0.0;
  c.reward.finish_weight = 0.0;
  c.contact.enabled = false;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  std::vector<float> rew(env->n_cars());

  // Everyone coasts except the car at the back, which is given full throttle.
  const int last = [&] {
    for (const CarState& car : env->cars()) {
      if (car.position == env->n_cars()) return car.index;
    }
    return 0;
  }();

  bool saw_gain = false, saw_loss = false;
  for (int t = 0; t < 400 && !env->done(); ++t) {
    for (int i = 0; i < env->n_cars(); ++i) {
      act[i * 2 + 1] = i == last ? 1.0f : -0.2f;
    }
    const std::vector<int> before = [&] {
      std::vector<int> p;
      for (const CarState& car : env->cars()) p.push_back(car.position);
      return p;
    }();

    env->step(act.data(), rew.data(), nullptr);

    for (int i = 0; i < env->n_cars(); ++i) {
      const int moved = before[i] - env->cars()[i].position;
      if (moved > 0) {
        REQUIRE(rew[i] > 0.0f);
        saw_gain = true;
      } else if (moved < 0) {
        REQUIRE(rew[i] < 0.0f);
        saw_loss = true;
      }
    }
  }
  REQUIRE(saw_gain);
  REQUIRE(saw_loss);
}

TEST_CASE("team weight blends a car's reward with its team mate's",
          "[racing][reward][team]") {
  // At team_weight 1 both cars on a team must receive exactly the same reward:
  // the team is then one agent with two bodies. That is the cleanest possible
  // check that the mixing is actually wired up.
  EnvConfig c = base_config();
  c.field.n_teams = 2;
  c.field.cars_per_team = 2;
  c.reward.team_weight = 1.0;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) {
    act[i * 2] = 0.1f * i;      // different steering, so own rewards differ
    act[i * 2 + 1] = 0.4f;
  }
  std::vector<float> rew(env->n_cars());

  for (int t = 0; t < 50; ++t) {
    env->step(act.data(), rew.data(), nullptr);
    REQUIRE(rew[0] == Approx(rew[1]));  // team 0
    REQUIRE(rew[2] == Approx(rew[3]));  // team 1
  }
}

TEST_CASE("at team weight zero a car is paid only for its own race",
          "[racing][reward][team]") {
  EnvConfig c = base_config();
  c.field.n_teams = 1;
  c.field.cars_per_team = 2;
  c.reward.team_weight = 0.0;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{0.0f, 1.0f, 0.0f, -1.0f};  // one drives, one brakes
  std::vector<float> rew(2);
  bool differed = false;
  for (int t = 0; t < 60; ++t) {
    env->step(act.data(), rew.data(), nullptr);
    if (std::abs(rew[0] - rew[1]) > 1e-6f) differed = true;
  }
  REQUIRE(differed);
}

TEST_CASE("a car that runs wide is penalised but stays in the race",
          "[racing][race][limits]") {
  EnvConfig c = base_config();
  c.field.n_teams = 1;
  c.field.cars_per_team = 1;
  c.reward.terminate_off_track = false;
  c.track.episode_distance = 4000.0;
  // What this checks is that leaving the corridor is not by itself terminal:
  // the car is penalised, recovered, and races on. Damage is off because the
  // input below is full lock held for six hundred steps, which is not a car
  // running wide -- it is a car driven into the barrier over and over, and
  // that SHOULD end its race. The boundary between the two lives in
  // test_racing_damage.cpp.
  c.damage.enabled = false;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{1.0f, 1.0f};  // full lock, full throttle: straight off
  std::vector<float> rew(1);

  bool went_off = false, recovered = false;
  for (int t = 0; t < 600 && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
    for (const RaceEvent& e : env->events()) {
      if (e.type == RaceEvent::OFF_TRACK) went_off = true;
      if (e.type == RaceEvent::REJOIN) recovered = true;
    }
    REQUIRE_FALSE(env->cars()[0].retired);
  }
  REQUIRE(went_off);
  REQUIRE(recovered);
}

TEST_CASE("training mode retires a car that leaves the circuit",
          "[racing][race][limits]") {
  EnvConfig c = base_config();
  c.field.n_teams = 1;
  c.field.cars_per_team = 1;
  c.reward.terminate_off_track = true;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act{1.0f, 1.0f};
  std::vector<float> rew(1);
  for (int t = 0; t < 600 && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
  }
  REQUIRE(env->done());
  REQUIRE(env->cars()[0].retired);
}

TEST_CASE("the observation never leaks another car's absolute position",
          "[racing][obs]") {
  // A policy may know where a rival is relative to itself. It may not know
  // where either of them is on the planet, or the network would learn the
  // circuit by coordinate rather than by what it can see.
  EnvConfig c = base_config();
  auto env = make(c);
  env->reset(0);

  std::vector<float> obs(env->n_cars() * env->obs_dim());
  env->observe(obs.data());
  for (float v : obs) {
    REQUIRE(std::isfinite(v));
    // World coordinates at Bahrain run to several thousand metres; nothing in
    // a normalised observation should come close.
    REQUIRE(std::abs(v) < 50.0f);
  }
}

TEST_CASE("obs_dim matches what observe actually writes", "[racing][obs]") {
  EnvConfig c = base_config();
  c.curvature_lookahead = 12;
  c.race.n_neighbours = 3;
  auto env = make(c);
  env->reset(0);
  REQUIRE(env->obs_dim() == 4 + 2 + 12 + 3 + 2 + 5 * 3);

  // Write into a buffer with a guard value on the end and check it survives.
  const int n = env->n_cars() * env->obs_dim();
  std::vector<float> obs(n + 1, -12345.0f);
  env->observe(obs.data());
  REQUIRE(obs[n] == -12345.0f);
  for (int i = 0; i < n; ++i) REQUIRE(obs[i] != -12345.0f);
}

TEST_CASE("a full race classifies everyone exactly once", "[racing][race]") {
  EnvConfig c = base_config();
  c.field.n_teams = 4;
  c.field.cars_per_team = 2;
  c.track.episode_distance = 1500.0;
  auto env = make(c);
  env->reset(0);

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) act[i * 2 + 1] = 0.3f + 0.02f * i;
  std::vector<float> rew(env->n_cars());

  while (!env->done()) env->step(act.data(), rew.data(), nullptr);

  const std::vector<int> order = env->finish_order();
  REQUIRE(static_cast<int>(order.size()) == env->n_cars());
  std::vector<int> sorted = order;
  std::sort(sorted.begin(), sorted.end());
  std::vector<int> expected(env->n_cars());
  std::iota(expected.begin(), expected.end(), 0);
  REQUIRE(sorted == expected);

  const std::vector<int> scores = env->team_scores();
  REQUIRE(static_cast<int>(scores.size()) == c.field.n_teams);
  const int total = std::accumulate(scores.begin(), scores.end(), 0);
  // Each car contributes (n - position), so the total is fixed whatever
  // happened: n-1 down to 0.
  REQUIRE(total == env->n_cars() * (env->n_cars() - 1) / 2);
}
