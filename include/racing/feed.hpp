// The visualizer feed: where every car is, often enough to draw it.
//
// This is the seam between the engine and whatever renders the race, and it is
// the one interface in the repository that other people build against. Two
// things follow from that.
//
// First, it is self-describing. `episode.json` carries a `fields` array naming
// every float in a car's frame record, in order, and a `stride`. A reader that
// indexes by name rather than by a hardcoded offset keeps working when a field
// is added, and fields WILL be added. Never hardcode 16.
//
// Second, it comes in two shapes for two jobs:
//
//   * A replay. `track.json` + `episode.json` + `frames.f32`. The binary blob
//     is a flat little-endian float32 array laid out frame -> car -> field, so
//     car `c` at frame `f` starts at (f * n_cars + c) * stride. Compact enough
//     to scrub through a whole race in a browser.
//   * A live stream. One JSON object per line: a header, then a frame per
//     sample, then a result. Tail it while the race runs and draw as it
//     arrives. Slower to parse and far larger, which is why it is not the
//     replay format.
//
// Sampling rate. The physics runs at 100 Hz and the policy acts at 25 Hz. 25 Hz
// is too coarse to render from -- a car covers three metres between policy
// steps at the end of the straight -- so the feed hangs off the physics
// observer instead and decimates to whatever `frame_rate` asks for. 60 Hz is
// the default and is what the format was sized around. Frames are sampled
// against elapsed simulated time rather than by dropping every Nth step,
// because 100/60 is not an integer and dropping would drift against the clock.

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "racing/race.hpp"
#include "racing/track.hpp"

namespace racing {

// Field order is part of the published format. Append only -- never reorder,
// never remove, and bump kFeedVersion if you do either anyway.
enum FrameField : int {
  FRAME_X = 0,             // world metres
  FRAME_Y,                 // world metres
  FRAME_Z,                 // world metres; elevation, from the source telemetry
  FRAME_HEADING,           // radians, NOT wrapped -- it accumulates over a lap
  FRAME_SPEED,             // m/s
  FRAME_STEER,             // -1 to 1, fraction of full lock
  FRAME_THROTTLE,          // -1 to 1; negative is braking
  FRAME_DOWNFORCE,         // 1.0 in clean air, lower in dirty air
  FRAME_DRAG,              // 1.0 in clean air, lower in a tow
  FRAME_WAKE,              // 0 to 1, how deep in another car's wake this is
  FRAME_LATERAL_OFFSET,    // metres left of the reference line
  FRAME_POSITION,          // 1-based race position
  FRAME_LAP,               // laps completed
  FRAME_LAP_FRACTION,      // 0 to 1 around the current lap
  FRAME_GAP_AHEAD,         // seconds to the car classified in front
  FRAME_FLAGS,             // bit field, see FrameFlag

  // --- appended when the full physics arrived ----------------------------
  // Everything below was added after the format shipped, which is exactly the
  // case the `fields` array in the header exists for: a reader that looks
  // fields up by name kept working, one that hardcoded a stride of 16 did not.
  FRAME_TYRE_TEMP_FRONT,   // degrees C, core
  FRAME_TYRE_TEMP_REAR,
  FRAME_TYRE_WEAR_FRONT,   // 0 fresh, 1 worn out
  FRAME_TYRE_WEAR_REAR,
  FRAME_TYRE_GRIP,         // combined multiplier, 1.0 is a tyre at its best
  FRAME_FUEL_KG,
  FRAME_GEAR,              // 1-based, so it reads like a dashboard
  FRAME_RPM,
  FRAME_ERS_CHARGE,        // 0 to 1 of the store
  FRAME_LATERAL_G,
  FRAME_LONGITUDINAL_G,

  // --- appended when damage arrived --------------------------------------
  FRAME_DAMAGE,            // 0 undamaged, 1 terminal. Never decreases
  FRAME_FIELDS
};

enum FrameFlag : int {
  FLAG_OFF_TRACK = 1,
  FLAG_CONTACT = 2,
  FLAG_FINISHED = 4,
  FLAG_RETIRED = 8,
  FLAG_DRS_OPEN = 16,
  FLAG_WHEELSPIN = 32,
  FLAG_LOCKUP = 64,
  FLAG_ERS_DEPLOYING = 128,
  // Set for the frames in which the car is against a barrier. Brief -- an
  // impact is over in a step or two -- so a renderer should latch it rather
  // than expecting it to persist.
  FLAG_BARRIER = 256,
};

// 4: `damage` appended, FLAG_BARRIER added, and RETIRE events now carry a
// `reason`. Additive, as every version bump here has been -- a reader that
// looks fields up by name and ignores flags it does not know keeps working.
constexpr int kFeedVersion = 4;

struct TeamInfo {
  std::string name;
  std::string color;  // "#rrggbb"
};

class Feed {
 public:
  // `frame_rate` is the sampling rate in simulated hertz. It is capped by the
  // physics rate; asking for more than that just gives you the physics rate.
  explicit Feed(const RaceEnv& env, double frame_rate = 60.0);

  // Attach to a race so frames are captured automatically as it steps. This
  // installs the environment's physics observer, so only one Feed can be
  // attached to a race at a time.
  void attach(RaceEnv* env);

  // Sample one frame if enough simulated time has passed since the last. Called
  // for you when attached; exposed for a caller that would rather drive it.
  void capture(const RaceEnv& env);

  // Drain a step's events into the feed, tagging each with the frame that was
  // current when it happened. Called for you when attached.
  void collect_events(const RaceEnv& env);

  // --- live stream --------------------------------------------------------
  // Opens `path` and writes the header immediately, so a reader that starts
  // late still knows the layout. Every captured frame is appended as one line.
  void open_stream(const std::string& path, const RaceEnv& env);
  void close_stream(const RaceEnv& env);
  bool streaming() const { return stream_ && stream_->is_open(); }

  // --- replay -------------------------------------------------------------
  // Writes `<dir>/track.json`, `<dir>/episode.json` and `<dir>/frames.f32`.
  void write(const std::string& dir, const RaceEnv& env);

  // The circuit, in the shape the renderer wants: a polyline with widths.
  // Static per track, so a visualizer can cache it across episodes.
  static void write_track(const std::string& dir, const Track& track);

  // --- presentation -------------------------------------------------------
  // Optional. Sensible defaults are generated if these are never called.
  void set_teams(const std::vector<TeamInfo>& teams);
  void set_car_names(const std::vector<std::string>& names);

  int n_frames() const { return n_frames_; }
  int n_cars() const { return n_cars_; }
  double frame_rate() const { return frame_rate_; }

 private:
  struct Event {
    int frame;
    double t;
    int type;
    int car;
    int other;
    double value;
    int lap;
    int reason;
  };

  void fill_frame(const RaceEnv& env, float* out) const;
  std::string header_json(const RaceEnv& env) const;
  std::string result_json(const RaceEnv& env) const;
  void default_presentation(const RaceEnv& env);

  int n_cars_ = 0;
  double frame_rate_ = 60.0;
  double next_capture_ = 0.0;
  int n_frames_ = 0;

  std::vector<float> frames_;
  std::vector<Event> events_;
  std::vector<TeamInfo> teams_;
  std::vector<std::string> car_names_;
  std::vector<int> team_of_;

  std::unique_ptr<std::ofstream> stream_;
  std::vector<float> scratch_;
};

// Names for the RaceEvent types, as they appear in the feed. Kept next to the
// writer so the two cannot drift apart.
const char* event_type_name(int type);

}  // namespace racing
