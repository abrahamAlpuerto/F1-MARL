// Determinism, and the guarantees that depend on it.
//
// A race has to replay exactly. Two reasons, and the first is the one that
// bites: the visualizer feed is written from a separate pass over the same
// seed, so if the engine is not reproducible the replay a viewer watches is not
// the race that was scored. The second is ordinary debugging -- an eight-car
// race with contact in it is not something you want to have to reproduce by
// luck.
//
// This cannot be retrofitted, which is why it is tested from the start.

#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "catch2/catch.hpp"
#include "racing/config.hpp"
#include "racing/race.hpp"
#include "racing/rng.hpp"

using namespace racing;

namespace {

EnvConfig test_config() {
  EnvConfig c;
  c.track.path = std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
  c.seed = 12345;
  c.field.n_teams = 2;
  c.field.cars_per_team = 2;
  c.track.episode_distance = 900.0;
  return c;
}

// Deterministic pseudo-policy: a pure function of the observation, so any
// difference between runs comes from the environment rather than the actions.
void scripted_policy(const float* obs, int dim, int t, int car, float* act) {
  double a = 0.0;
  for (int k = 0; k < dim; ++k) a += obs[k] * (k + 1);
  act[0] = static_cast<float>(std::sin(a + t * 0.01 + car));
  act[1] = static_cast<float>(std::cos(a * 0.5 - t * 0.02));
}

// Run a batch and return every observation and reward seen, flattened.
std::vector<double> run(int n_envs, int n_threads, int steps) {
  const EnvConfig cfg = test_config();
  VecRaceEnv env(cfg, n_envs, n_threads);
  const int d = env.obs_dim();
  const int nc = env.n_cars();
  const size_t obs_n = static_cast<size_t>(n_envs) * nc * d;

  std::vector<float> obs(obs_n);
  std::vector<float> act(static_cast<size_t>(n_envs) * nc * 2);
  std::vector<float> rew(static_cast<size_t>(n_envs) * nc);
  std::vector<uint8_t> done(n_envs);

  env.reset_all(0);
  env.observe(obs.data());

  std::vector<double> trace;
  for (int t = 0; t < steps; ++t) {
    for (int i = 0; i < n_envs; ++i) {
      for (int c = 0; c < nc; ++c) {
        const size_t o = (static_cast<size_t>(i) * nc + c) * d;
        scripted_policy(obs.data() + o, d, t, c,
                        act.data() + (static_cast<size_t>(i) * nc + c) * 2);
      }
    }
    env.step(act.data(), obs.data(), rew.data(), done.data());
    trace.insert(trace.end(), obs.begin(), obs.end());
    trace.insert(trace.end(), rew.begin(), rew.end());
    for (uint8_t v : done) trace.push_back(v);
  }
  return trace;
}

}  // namespace

TEST_CASE("a race replays bit-exactly under any thread count",
          "[racing][determinism]") {
  const std::vector<double> ref = run(8, 1, 100);
  for (int threads : {2, 4, 8}) {
    const std::vector<double> other = run(8, threads, 100);
    REQUIRE(other.size() == ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
      // Bit-exact, not approximate. An engine that only agrees to a tolerance
      // has an ordering dependency somewhere, and it will surface later as a
      // replay that does not match the race it claims to be.
      REQUIRE(other[i] == ref[i]);
    }
  }
}

TEST_CASE("the same seed reproduces across separate constructions",
          "[racing][determinism]") {
  REQUIRE(run(4, 1, 60) == run(4, 1, 60));
}

TEST_CASE("keyed draws do not depend on call order", "[racing][rng]") {
  // The point of counter-based randomness: asking for a later draw first must
  // not change any of them.
  const uint64_t seed = 999;
  std::vector<double> forward, backward;
  for (uint32_t step = 0; step < 50; ++step) {
    forward.push_back(uniform(seed, 3, STREAM_RESET_POSE, step, 0));
  }
  for (int step = 49; step >= 0; --step) {
    backward.push_back(
        uniform(seed, 3, STREAM_RESET_POSE, static_cast<uint32_t>(step), 0));
  }
  for (size_t i = 0; i < forward.size(); ++i) {
    REQUIRE(forward[i] == backward[forward.size() - 1 - i]);
  }
}

TEST_CASE("distinct key components give distinct streams", "[racing][rng]") {
  // (step=1, car=2) must not collide with (step=2, car=1), which a plain XOR
  // or sum of the components would allow.
  const uint64_t seed = 7;
  REQUIRE(uniform(seed, 0, 0, 1, 2) != uniform(seed, 0, 0, 2, 1));
  REQUIRE(uniform(seed, 1, 0, 0, 0) != uniform(seed, 0, 1, 0, 0));
  REQUIRE(uniform(seed, 0, 0, 0, 0, 1) != uniform(seed, 0, 0, 0, 1, 0));
}

TEST_CASE("the same grid comes up whatever the cars then do",
          "[racing][determinism]") {
  // Two runs of the same seed with different driving must still start from an
  // identical grid, or comparing two policies on "the same race" is comparing
  // them on two different ones.
  EnvConfig cfg = test_config();
  cfg.track.randomize_start = true;

  auto first_obs = [&](int variant) {
    VecRaceEnv env(cfg, 4, 1);
    const int d = env.obs_dim(), nc = env.n_cars();
    std::vector<float> obs(static_cast<size_t>(4) * nc * d);
    std::vector<float> act(static_cast<size_t>(4) * nc * 2,
                           static_cast<float>(0.1 * variant));
    std::vector<float> rew(static_cast<size_t>(4) * nc);
    std::vector<uint8_t> done(4);

    env.reset_all(0);
    env.observe(obs.data());
    const std::vector<float> first(obs.begin(), obs.end());

    for (int t = 0; t < 10 + variant * 7; ++t) {
      env.step(act.data(), obs.data(), rew.data(), done.data());
    }
    return first;
  };

  REQUIRE(first_obs(0) == first_obs(1));
}

TEST_CASE("observations stay finite under adversarial driving",
          "[racing][robustness]") {
  const EnvConfig cfg = test_config();
  VecRaceEnv env(cfg, 8, 2);
  const int d = env.obs_dim(), nc = env.n_cars();
  std::vector<float> obs(static_cast<size_t>(8) * nc * d);
  std::vector<float> act(static_cast<size_t>(8) * nc * 2);
  std::vector<float> rew(static_cast<size_t>(8) * nc);
  std::vector<uint8_t> done(8);

  env.reset_all(0);
  env.observe(obs.data());

  for (int t = 0; t < 400; ++t) {
    // Full lock and full brake, then full lock the other way, with the cars
    // deliberately steering into each other. If anything can blow the model
    // up -- the vehicle, the wake, or the contact response -- this will.
    for (size_t i = 0; i < act.size() / 2; ++i) {
      act[i * 2] = (t / 7 + i) % 2 ? 1.0f : -1.0f;
      act[i * 2 + 1] = (t / 11) % 2 ? 1.0f : -1.0f;
    }
    env.step(act.data(), obs.data(), rew.data(), done.data());
    for (float v : obs) {
      REQUIRE(std::isfinite(v));
      REQUIRE(std::abs(v) < 1e4f);
    }
    for (float v : rew) REQUIRE(std::isfinite(v));
  }
}

TEST_CASE("a config round-trips through JSON", "[racing][config]") {
  EnvConfig c = test_config();
  c.vehicle.cl_a = 4.321;
  c.reward.progress_weight = 0.077;
  c.reward.team_weight = 0.25;
  c.aero.tow_max = 0.44;
  c.field.n_teams = 5;

  const EnvConfig back = EnvConfig::from_json_string(c.to_json_string());
  REQUIRE(back.vehicle.cl_a == Approx(4.321));
  REQUIRE(back.reward.progress_weight == Approx(0.077));
  REQUIRE(back.reward.team_weight == Approx(0.25));
  REQUIRE(back.aero.tow_max == Approx(0.44));
  REQUIRE(back.field.n_teams == 5);
  REQUIRE(back.seed == c.seed);
}

TEST_CASE("an unknown config key is rejected rather than ignored",
          "[racing][config]") {
  // A typo in a swept parameter name would otherwise run the wrong race while
  // appearing to work.
  REQUIRE_THROWS(EnvConfig::from_json_string(
      R"({"vehicle": {"cl_a": 5.0, "cla_typo": 3.0}})"));
  REQUIRE_THROWS(EnvConfig::from_json_string(R"({"n_carz": 4})"));
  REQUIRE_THROWS(EnvConfig::from_json_string(R"({"aero": {"tow_maxx": 0.5}})"));
}
