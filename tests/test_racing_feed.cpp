// The visualizer feed.
//
// This is the one format other people build against, so these tests are about
// the contract rather than about the engine: does the header describe the data
// that was actually written, is the frame rate what was asked for, and can a
// reader that only knows `fields` and `stride` recover every car's position.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "catch2/catch.hpp"
#include "nlohmann/json.hpp"
#include "racing/config.hpp"
#include "racing/feed.hpp"
#include "racing/race.hpp"

using namespace racing;
using nlohmann::json;

namespace {

EnvConfig feed_config() {
  EnvConfig c;
  c.track.path = std::string(RACING_DATA_DIR) + "/tracks/bahrain.json";
  c.seed = 77;
  c.field.n_teams = 2;
  c.field.cars_per_team = 2;
  c.track.episode_distance = 700.0;
  return c;
}

std::unique_ptr<RaceEnv> make(const EnvConfig& c) {
  auto track = std::make_shared<const Track>(Track::load(c.track.path));
  return std::make_unique<RaceEnv>(c, track, 0);
}

// A scratch directory of our own, created once.
//
// Not the working directory. These tests write episode.json, frames.f32 and
// track.json, and the working directory is wherever the binary happened to be
// launched from -- which, when that is the repository root, means the test
// suite quietly drops three files into the source tree and they get committed.
const std::string& temp_dir() {
  static const std::string dir = [] {
    const auto path = std::filesystem::temp_directory_path() / "racing_feed_tests";
    std::filesystem::create_directories(path);
    return path.string();
  }();
  return dir;
}

// Drive a short race with the feed attached and return it.
std::unique_ptr<Feed> run_race(RaceEnv* env, double rate, int max_steps = 4000) {
  auto feed = std::make_unique<Feed>(*env, rate);
  feed->attach(env);
  env->reset(0);

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) act[i * 2 + 1] = 0.35f + 0.02f * i;
  std::vector<float> rew(env->n_cars());

  for (int t = 0; t < max_steps && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
    feed->collect_events(*env);
  }
  return feed;
}

}  // namespace

TEST_CASE("the feed samples at the rate it was asked for", "[racing][feed]") {
  EnvConfig c = feed_config();
  auto env = make(c);
  auto feed = run_race(env.get(), 60.0);

  // Frames captured against elapsed simulated time, so the count follows the
  // race duration rather than the step count.
  const double expected = env->race_time() * 60.0;
  REQUIRE(feed->n_frames() > 0);
  REQUIRE(static_cast<double>(feed->n_frames()) == Approx(expected).margin(3.0));

  // And a different rate gives proportionally different frames.
  auto env2 = make(c);
  auto slow = run_race(env2.get(), 20.0);
  REQUIRE(slow->n_frames() < feed->n_frames());
  REQUIRE(static_cast<double>(feed->n_frames()) ==
          Approx(3.0 * slow->n_frames()).epsilon(0.1));
}

TEST_CASE("the header describes the file that was written", "[racing][feed]") {
  EnvConfig c = feed_config();
  auto env = make(c);
  auto feed = run_race(env.get(), 60.0);
  feed->write(temp_dir(), *env);

  std::ifstream in(temp_dir() + "/episode.json");
  REQUIRE(in.good());
  json j;
  in >> j;

  REQUIRE(j.at("format") == "racing-feed");
  REQUIRE(j.at("version").get<int>() == kFeedVersion);
  REQUIRE(j.at("n_cars").get<int>() == env->n_cars());
  REQUIRE(j.at("n_frames").get<int>() == feed->n_frames());

  // Self-describing: the field list and the stride must agree with each other,
  // because a reader indexes by name and strides by the number.
  const auto fields = j.at("fields").get<std::vector<std::string>>();
  REQUIRE(static_cast<int>(fields.size()) == j.at("stride").get<int>());
  REQUIRE(fields[0] == "x");
  REQUIRE(fields[1] == "y");

  // Every car is named and on a team, and every team is named and coloured.
  REQUIRE(static_cast<int>(j.at("cars").size()) == env->n_cars());
  REQUIRE(static_cast<int>(j.at("teams").size()) == c.field.n_teams);
  for (const auto& car : j.at("cars")) {
    REQUIRE(car.at("team").get<int>() >= 0);
    REQUIRE(car.at("team").get<int>() < c.field.n_teams);
    REQUIRE_FALSE(car.at("name").get<std::string>().empty());
  }
  for (const auto& team : j.at("teams")) {
    REQUIRE(team.at("color").get<std::string>()[0] == '#');
  }

  // The binary file is exactly the size the header implies. A reader that
  // trusts the header and gets a short file renders nothing, so this is the
  // check that matters most to whoever is on the other end.
  std::ifstream fb(temp_dir() + "/frames.f32",
                   std::ios::binary | std::ios::ate);
  REQUIRE(fb.good());
  const std::streamsize bytes = fb.tellg();
  const size_t expected_floats = static_cast<size_t>(feed->n_frames()) *
                                 env->n_cars() * FRAME_FIELDS;
  REQUIRE(static_cast<size_t>(bytes) == expected_floats * sizeof(float));
}

TEST_CASE("frames can be read back the way the format says", "[racing][feed]") {
  EnvConfig c = feed_config();
  auto env = make(c);
  auto feed = run_race(env.get(), 60.0);
  feed->write(temp_dir(), *env);

  std::ifstream in(temp_dir() + "/episode.json");
  json j;
  in >> j;
  const int n_cars = j.at("n_cars").get<int>();
  const int n_frames = j.at("n_frames").get<int>();
  const int stride = j.at("stride").get<int>();
  const auto fields = j.at("fields").get<std::vector<std::string>>();

  auto field_index = [&](const std::string& name) {
    for (size_t i = 0; i < fields.size(); ++i) {
      if (fields[i] == name) return static_cast<int>(i);
    }
    return -1;
  };
  const int fx = field_index("x");
  const int fy = field_index("y");
  const int fpos = field_index("position");
  const int flap = field_index("lap");
  REQUIRE(fx >= 0);
  REQUIRE(fy >= 0);
  REQUIRE(fpos >= 0);
  REQUIRE(flap >= 0);

  std::ifstream fb(temp_dir() + "/frames.f32", std::ios::binary);
  std::vector<float> data(static_cast<size_t>(n_frames) * n_cars * stride);
  fb.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(float)));

  for (int f = 0; f < n_frames; ++f) {
    std::vector<int> seen(n_cars + 1, 0);
    for (int car = 0; car < n_cars; ++car) {
      const size_t o = (static_cast<size_t>(f) * n_cars + car) * stride;
      // Bahrain sits within a few kilometres of its own origin.
      REQUIRE(std::abs(data[o + fx]) < 20000.0f);
      REQUIRE(std::abs(data[o + fy]) < 20000.0f);
      REQUIRE(data[o + flap] >= 0.0f);

      const int p = static_cast<int>(data[o + fpos]);
      REQUIRE(p >= 1);
      REQUIRE(p <= n_cars);
      ++seen[p];
    }
    // Every position is held by exactly one car in every frame. A leaderboard
    // that shows two cars in P3 is the sort of thing nobody notices until the
    // race is being watched.
    for (int p = 1; p <= n_cars; ++p) REQUIRE(seen[p] == 1);
  }
}

TEST_CASE("cars move between consecutive frames", "[racing][feed]") {
  // A feed that captures the same instant repeatedly passes every structural
  // check above and renders a car park.
  EnvConfig c = feed_config();
  auto env = make(c);
  auto feed = run_race(env.get(), 60.0);
  feed->write(temp_dir(), *env);

  std::ifstream in(temp_dir() + "/episode.json");
  json j;
  in >> j;
  const int n_cars = j.at("n_cars").get<int>();
  const int n_frames = j.at("n_frames").get<int>();
  const int stride = j.at("stride").get<int>();

  std::ifstream fb(temp_dir() + "/frames.f32", std::ios::binary);
  std::vector<float> data(static_cast<size_t>(n_frames) * n_cars * stride);
  fb.read(reinterpret_cast<char*>(data.data()),
          static_cast<std::streamsize>(data.size() * sizeof(float)));

  double total = 0.0;
  for (int f = 1; f < n_frames; ++f) {
    const size_t a = (static_cast<size_t>(f - 1) * n_cars) * stride;
    const size_t b = (static_cast<size_t>(f) * n_cars) * stride;
    const double dx = data[b + FRAME_X] - data[a + FRAME_X];
    const double dy = data[b + FRAME_Y] - data[a + FRAME_Y];
    const double step = std::sqrt(dx * dx + dy * dy);
    // At 60 Hz and 100 m/s a car covers under two metres per frame. Anything
    // much larger means a teleport, which is what a contact response that
    // corrects position in one go looks like.
    REQUIRE(step < 5.0);
    total += step;
  }
  REQUIRE(total > 100.0);
}

TEST_CASE("events land on real frames and name a real car", "[racing][feed]") {
  EnvConfig c = feed_config();
  auto env = make(c);
  auto feed = run_race(env.get(), 60.0);
  feed->write(temp_dir(), *env);

  std::ifstream in(temp_dir() + "/episode.json");
  json j;
  in >> j;
  const int n_frames = j.at("n_frames").get<int>();
  const int n_cars = j.at("n_cars").get<int>();

  int last_frame = -1;
  bool saw_finish = false;
  for (const auto& e : j.at("events")) {
    const int frame = e.at("frame").get<int>();
    REQUIRE(frame >= 0);
    REQUIRE(frame < n_frames);
    // Monotonic, so a reader can walk the event list alongside the frames
    // instead of sorting it.
    REQUIRE(frame >= last_frame);
    last_frame = frame;

    const int car = e.at("car").get<int>();
    REQUIRE(car >= 0);
    REQUIRE(car < n_cars);
    REQUIRE_FALSE(e.at("type").get<std::string>().empty());
    if (e.at("type") == "finish") saw_finish = true;
  }
  REQUIRE(saw_finish);

  // And the result block is complete.
  const auto& res = j.at("result");
  REQUIRE(static_cast<int>(res.at("finish_order").size()) == n_cars);
  REQUIRE(static_cast<int>(res.at("classification").size()) == n_cars);
}

TEST_CASE("the live stream is one JSON object per line", "[racing][feed]") {
  EnvConfig c = feed_config();
  c.track.episode_distance = 300.0;
  auto env = make(c);

  Feed feed(*env, 30.0);
  feed.attach(env.get());
  env->reset(0);

  const std::string path = temp_dir() + "/stream_test.jsonl";
  feed.open_stream(path, *env);
  REQUIRE(feed.streaming());

  std::vector<float> act(env->n_cars() * 2, 0.0f);
  for (int i = 0; i < env->n_cars(); ++i) act[i * 2 + 1] = 0.4f;
  std::vector<float> rew(env->n_cars());
  for (int t = 0; t < 3000 && !env->done(); ++t) {
    env->step(act.data(), rew.data(), nullptr);
    feed.collect_events(*env);
  }
  feed.close_stream(*env);
  REQUIRE_FALSE(feed.streaming());

  std::ifstream in(path);
  REQUIRE(in.good());
  std::string line;
  int headers = 0, frames = 0, results = 0, events = 0;
  bool header_first = false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const json o = json::parse(line);  // throws if a line is not self-contained
    const std::string type = o.at("type").get<std::string>();
    if (type == "header") {
      if (headers == 0 && frames == 0) header_first = true;
      ++headers;
      REQUIRE(o.at("stride").get<int>() == static_cast<int>(FRAME_FIELDS));
    } else if (type == "frame") {
      ++frames;
      REQUIRE(static_cast<int>(o.at("cars").size()) == env->n_cars());
      for (const auto& row : o.at("cars")) {
        REQUIRE(static_cast<int>(row.size()) == static_cast<int>(FRAME_FIELDS));
      }
    } else if (type == "event") {
      ++events;
    } else if (type == "result") {
      ++results;
    }
  }
  // Exactly one header, written before anything else so a reader that attaches
  // late still knows the layout; exactly one result, at the end.
  REQUIRE(headers == 1);
  REQUIRE(header_first);
  REQUIRE(results == 1);
  REQUIRE(frames > 10);
  REQUIRE(events > 0);
  REQUIRE(frames == feed.n_frames());

  std::remove(path.c_str());
}

TEST_CASE("track.json carries enough to draw the circuit", "[racing][feed]") {
  const Track t = Track::load(std::string(RACING_DATA_DIR) +
                              "/tracks/bahrain.json");
  Feed::write_track(temp_dir(), t);

  std::ifstream in(temp_dir() + "/track.json");
  REQUIRE(in.good());
  json j;
  in >> j;

  REQUIRE(j.at("n").get<int>() == t.n());
  REQUIRE(static_cast<int>(j.at("line").size()) == t.n());
  REQUIRE(static_cast<int>(j.at("half_width_left").size()) == t.n());
  REQUIRE(j.at("closed").get<bool>());
  // Three coordinates per point: the elevation is there for a renderer that
  // wants it and is harmless for one that does not.
  for (const auto& p : j.at("line")) REQUIRE(p.size() == 3);
  REQUIRE(j.at("sector_indices").size() >= 1);
}
