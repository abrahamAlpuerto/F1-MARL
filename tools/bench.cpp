// How many races per second, and where the time goes.
//
//     racing_bench [n_envs] [n_threads]
//
// The number that matters for training is environment-steps per second, where
// one environment step advances every car in one race by one policy step. A
// learner on CPU is usually the bottleneck rather than this, but it is worth
// knowing which side of that line you are on before optimising the wrong one.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "racing/config.hpp"
#include "racing/race.hpp"

using namespace racing;

int main(int argc, char** argv) {
  const int n_envs = argc > 1 ? std::atoi(argv[1]) : 256;
  const int n_threads = argc > 2 ? std::atoi(argv[2]) : 4;

  EnvConfig cfg;
  cfg.track.path = std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
  cfg.field.n_teams = 4;
  cfg.field.cars_per_team = 2;
  cfg.track.randomize_start = true;
  cfg.track.episode_distance = 1500.0;
  cfg.reward.terminate_off_track = true;

  VecRaceEnv env(cfg, n_envs, n_threads);
  const int nc = env.n_cars();
  const int d = env.obs_dim();

  std::vector<float> obs(static_cast<size_t>(n_envs) * nc * d);
  std::vector<float> act(static_cast<size_t>(n_envs) * nc * 2, 0.2f);
  std::vector<float> rew(static_cast<size_t>(n_envs) * nc);
  std::vector<uint8_t> done(n_envs);

  env.reset_all(0);
  env.observe(obs.data());

  // A warm-up pass, so the timed section is not measuring page faults on the
  // observation buffer.
  for (int i = 0; i < 20; ++i) {
    env.step(act.data(), obs.data(), rew.data(), done.data());
  }

  constexpr int kSteps = 300;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kSteps; ++i) {
    env.step(act.data(), obs.data(), rew.data(), done.data());
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const double env_steps = double(kSteps) * n_envs;
  const double car_steps = env_steps * nc;

  std::printf("races          : %d\n", n_envs);
  std::printf("cars per race  : %d\n", nc);
  std::printf("threads        : %d\n", n_threads);
  std::printf("obs dim        : %d\n", d);
  std::printf("wall           : %.3f s for %d steps\n", secs, kSteps);
  std::printf("env-steps/s    : %.0f\n", env_steps / secs);
  std::printf("car-steps/s    : %.0f\n", car_steps / secs);
  // One policy step is action_repeat physics steps, and the simulated time it
  // covers is what decides how long a race takes to generate.
  const double sim_per_wall =
      env_steps * cfg.sim.physics_dt * cfg.sim.action_repeat / secs;
  std::printf("sim-seconds/s  : %.0f  (%.0fx real time across the batch)\n",
              sim_per_wall, sim_per_wall);
  return 0;
}
