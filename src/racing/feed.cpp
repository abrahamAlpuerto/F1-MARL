#include "racing/feed.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "racing/tyres.hpp"

namespace racing {

namespace {

using nlohmann::json;

// The field names, in the order FrameField declares them. This array IS the
// published contract -- it is what a reader indexes by -- so it must stay in
// step with the enum. The static_assert below is what enforces that.
const char* const kFieldNames[] = {
    "x", "y", "z", "heading", "speed", "steer", "throttle",
    "downforce", "drag", "wake", "lateral_offset", "position",
    "lap", "lap_fraction", "gap_ahead", "flags",
    "tyre_temp_front", "tyre_temp_rear", "tyre_wear_front", "tyre_wear_rear",
    "tyre_grip", "fuel_kg", "gear", "rpm", "ers_charge",
    "lateral_g", "longitudinal_g",
};
static_assert(sizeof(kFieldNames) / sizeof(kFieldNames[0]) == FRAME_FIELDS,
              "kFieldNames and the FrameField enum have drifted apart");

// Deliberately invented names and colours rather than real teams'. They are
// there so eight cars on a screen can be told apart, which is a rendering
// problem, not a licensing one.
//
// Every initial is distinct, and that is not an accident: car names are built
// from the team's first letter, so a palette with both "Vermilion" and
// "Verdant" in it produces two different cars both called V1 -- which is
// exactly as confusing on a timing screen as it sounds.
const TeamInfo kDefaultTeams[] = {
    {"Vermilion", "#e2412c"},
    {"Cobalt", "#2f6fe0"},
    {"Jade", "#25a55f"},
    {"Amber", "#e2a32c"},
    {"Indigo", "#8a4fd4"},
    {"Slate", "#7b8794"},
    {"Rose", "#e8617f"},
    {"Teal", "#1fa2a6"},
    {"Ochre", "#b07d2b"},
    {"Fuchsia", "#c8399b"},
};
constexpr int kDefaultTeamCount =
    static_cast<int>(sizeof(kDefaultTeams) / sizeof(kDefaultTeams[0]));

json event_json(int frame, double t, int type, int car, int other, double value,
                int lap) {
  json o;
  o["frame"] = frame;
  o["t"] = t;
  o["type"] = event_type_name(type);
  o["car"] = car;
  if (other >= 0) o["other"] = other;
  if (value != 0.0) o["value"] = value;
  o["lap"] = lap;
  return o;
}

}  // namespace

const char* event_type_name(int type) {
  switch (type) {
    case RaceEvent::OVERTAKE: return "overtake";
    case RaceEvent::CONTACT: return "contact";
    case RaceEvent::OFF_TRACK: return "off_track";
    case RaceEvent::REJOIN: return "rejoin";
    case RaceEvent::LAP: return "lap";
    case RaceEvent::FINISH: return "finish";
    case RaceEvent::RETIRE: return "retire";
    default: return "unknown";
  }
}

Feed::Feed(const RaceEnv& env, double frame_rate)
    : n_cars_(env.n_cars()), frame_rate_(frame_rate) {
  if (frame_rate_ <= 0.0) frame_rate_ = 60.0;
  scratch_.resize(static_cast<size_t>(n_cars_) * FRAME_FIELDS);
  team_of_ = env.teams();
  default_presentation(env);
}

void Feed::default_presentation(const RaceEnv& env) {
  const int n_teams = env.config().field.n_teams;
  if (teams_.empty()) {
    teams_.resize(n_teams);
    for (int t = 0; t < n_teams; ++t) {
      if (t < kDefaultTeamCount) {
        teams_[t] = kDefaultTeams[t];
      } else {
        teams_[t].name = "Team " + std::to_string(t + 1);
        teams_[t].color = "#888888";
      }
    }
  }
  if (car_names_.empty()) {
    const int per_team = std::max(env.config().field.cars_per_team, 1);
    car_names_.resize(n_cars_);
    for (int i = 0; i < n_cars_; ++i) {
      const int t = team_of_.empty() ? 0 : team_of_[i];
      const std::string& tn = teams_[t].name;
      car_names_[i] = tn.substr(0, 1) + std::to_string(i % per_team + 1);
    }
  }
}

void Feed::set_teams(const std::vector<TeamInfo>& teams) { teams_ = teams; }
void Feed::set_car_names(const std::vector<std::string>& names) {
  car_names_ = names;
}

void Feed::attach(RaceEnv* env) {
  env->set_observer([this](const RaceEnv& e) { this->capture(e); });
}

// --- capture ---------------------------------------------------------------

void Feed::fill_frame(const RaceEnv& env, float* out) const {
  const Track& track = env.track();
  const double L = track.length();
  // Reported as a fraction rather than in megajoules, so a renderer can draw a
  // bar without needing to know the regulation store size.
  const double cap = env.config().powertrain.ers_capacity_mj;

  for (int i = 0; i < n_cars_; ++i) {
    const CarState& c = env.cars()[i];
    float* o = out + static_cast<size_t>(i) * FRAME_FIELDS;

    o[FRAME_X] = static_cast<float>(c.v.x);
    o[FRAME_Y] = static_cast<float>(c.v.y);
    o[FRAME_Z] = static_cast<float>(track.z_at(c.f.s));
    o[FRAME_HEADING] = static_cast<float>(c.v.psi);
    o[FRAME_SPEED] = static_cast<float>(c.v.speed());
    o[FRAME_STEER] = static_cast<float>(c.steer);
    o[FRAME_THROTTLE] = static_cast<float>(c.throttle);
    o[FRAME_DOWNFORCE] = static_cast<float>(c.downforce_factor);
    o[FRAME_DRAG] = static_cast<float>(c.drag_factor);
    o[FRAME_WAKE] = static_cast<float>(c.wake);
    o[FRAME_LATERAL_OFFSET] = static_cast<float>(c.f.e_y);
    o[FRAME_POSITION] = static_cast<float>(c.position);
    o[FRAME_LAP] = static_cast<float>(c.lap);
    o[FRAME_LAP_FRACTION] = static_cast<float>(L > 0.0 ? c.f.s / L : 0.0);
    o[FRAME_GAP_AHEAD] = static_cast<float>(c.gap_ahead);

    int flags = 0;
    if (c.off_track) flags |= FLAG_OFF_TRACK;
    if (c.contact) flags |= FLAG_CONTACT;
    if (c.finished) flags |= FLAG_FINISHED;
    if (c.retired) flags |= FLAG_RETIRED;
    if (c.drs_open) flags |= FLAG_DRS_OPEN;
    if (c.wheelspin) flags |= FLAG_WHEELSPIN;
    if (c.lockup) flags |= FLAG_LOCKUP;
    if (c.powertrain.deploying) flags |= FLAG_ERS_DEPLOYING;
    o[FRAME_FLAGS] = static_cast<float>(flags);

    o[FRAME_TYRE_TEMP_FRONT] = static_cast<float>(c.tyre_front.temperature_c);
    o[FRAME_TYRE_TEMP_REAR] = static_cast<float>(c.tyre_rear.temperature_c);
    o[FRAME_TYRE_WEAR_FRONT] = static_cast<float>(c.tyre_front.wear);
    o[FRAME_TYRE_WEAR_REAR] = static_cast<float>(c.tyre_rear.wear);
    o[FRAME_TYRE_GRIP] =
        static_cast<float>(0.5 * (c.tyre_front.grip + c.tyre_rear.grip));
    o[FRAME_FUEL_KG] = static_cast<float>(c.fuel_kg);
    // 1-based, so it reads like the number on a dashboard rather than an index.
    o[FRAME_GEAR] = static_cast<float>(c.powertrain.gear + 1);
    o[FRAME_RPM] = static_cast<float>(c.powertrain.rpm);
    o[FRAME_ERS_CHARGE] = static_cast<float>(
        cap > 0.0 ? c.powertrain.ers_charge_mj / cap : 0.0);
    o[FRAME_LATERAL_G] = static_cast<float>(c.lateral_g);
    o[FRAME_LONGITUDINAL_G] = static_cast<float>(c.longitudinal_g);
  }
}

void Feed::capture(const RaceEnv& env) {
  // Sample against elapsed simulated time rather than a step counter: physics
  // at 100 Hz into playback at 60 Hz is not an integer ratio, so dropping every
  // other step would run playback 20% fast.
  const double t = env.race_time();
  if (n_frames_ > 0 && t + 1e-12 < next_capture_) return;

  fill_frame(env, scratch_.data());
  frames_.insert(frames_.end(), scratch_.begin(), scratch_.end());

  if (stream_ && stream_->is_open()) {
    json line;
    line["type"] = "frame";
    line["f"] = n_frames_;
    line["t"] = t;
    json cars = json::array();
    for (int i = 0; i < n_cars_; ++i) {
      json row = json::array();
      const float* o = scratch_.data() + static_cast<size_t>(i) * FRAME_FIELDS;
      for (int k = 0; k < FRAME_FIELDS; ++k) row.push_back(o[k]);
      cars.push_back(std::move(row));
    }
    line["cars"] = std::move(cars);
    (*stream_) << line.dump() << "\n";
  }

  ++n_frames_;

  // Advance the schedule by exactly one frame interval rather than setting it
  // relative to the time this frame actually landed. The difference is not
  // cosmetic: physics ticks land on a 10 ms grid, so `next = t + 1/60` always
  // rounds the next capture up to the following tick and the feed comes out at
  // 50 Hz instead of 60. Accumulating the ideal interval keeps the long-run
  // rate exact and lets the individual frames sit where the grid allows.
  next_capture_ += 1.0 / frame_rate_;
}

void Feed::collect_events(const RaceEnv& env) {
  // Events are worked out once per policy step, after the physics has run, so
  // they are attached to the most recent frame. That is at worst one policy
  // step -- 40 ms -- ahead of where they actually happened, which no viewer can
  // see and which keeps the event list monotonic in frame number.
  const int frame = std::max(0, n_frames_ - 1);
  for (const RaceEvent& e : env.events()) {
    events_.push_back(Event{frame, e.time, e.type, e.car, e.other, e.value, e.lap});
    if (stream_ && stream_->is_open()) {
      json line;
      line["type"] = "event";
      line["f"] = frame;
      line["event"] = event_json(frame, e.time, e.type, e.car, e.other, e.value,
                                 e.lap);
      (*stream_) << line.dump() << "\n";
    }
  }
}

// --- headers and results ---------------------------------------------------

std::string Feed::header_json(const RaceEnv& env) const {
  json j;
  j["type"] = "header";
  j["format"] = "racing-feed";
  j["version"] = kFeedVersion;
  j["n_cars"] = n_cars_;
  j["frame_rate"] = frame_rate_;
  j["stride"] = static_cast<int>(FRAME_FIELDS);

  json fields = json::array();
  for (int k = 0; k < FRAME_FIELDS; ++k) fields.push_back(kFieldNames[k]);
  j["fields"] = fields;

  j["track"] = env.track().name();
  // Conditions. A viewer comparing two races wants to know whether the second
  // one was quicker because the driving was better or because the air was cold.
  json cond;
  cond["air_temperature_c"] = env.config().atmosphere.air_temperature_c;
  cond["track_temperature_c"] = env.config().atmosphere.track_temperature_c;
  cond["air_density"] = env.air_density();
  cond["wind_speed"] = env.config().atmosphere.wind_speed;
  cond["tyre_compound"] = compound(env.config().tyre).name;
  cond["fuel_start_kg"] = env.config().fuel.enabled
                              ? env.config().fuel.start_kg : 0.0;
  j["conditions"] = cond;

  // Where the wing may be opened. Worth drawing on the map: a viewer who can
  // see the zones can see why a pass happened where it did.
  json zones = json::array();
  for (const DrsZone& z : env.drs_zones()) {
    json o;
    o["detection_s"] = z.detection_s;
    o["start_s"] = z.start_s;
    o["end_s"] = z.end_s;
    zones.push_back(o);
  }
  j["drs_zones"] = zones;
  j["lap_length_m"] = env.track().length();
  j["laps"] = env.config().track.laps;
  j["race_distance_m"] = env.race_distance();

  json teams = json::array();
  for (size_t t = 0; t < teams_.size(); ++t) {
    json o;
    o["index"] = static_cast<int>(t);
    o["name"] = teams_[t].name;
    o["color"] = teams_[t].color;
    teams.push_back(o);
  }
  j["teams"] = teams;

  json cars = json::array();
  for (int i = 0; i < n_cars_; ++i) {
    json o;
    o["index"] = i;
    o["name"] = i < static_cast<int>(car_names_.size()) ? car_names_[i]
                                                        : std::to_string(i);
    o["team"] = i < static_cast<int>(team_of_.size()) ? team_of_[i] : 0;
    cars.push_back(o);
  }
  j["cars"] = cars;

  return j.dump();
}

std::string Feed::result_json(const RaceEnv& env) const {
  json j;
  j["type"] = "result";
  j["finish_order"] = env.finish_order();
  j["team_scores"] = env.team_scores();

  json cls = json::array();
  for (int i = 0; i < n_cars_; ++i) {
    const CarState& c = env.cars()[i];
    json o;
    o["car"] = i;
    o["position"] = c.position;
    o["laps"] = c.lap;
    o["race_time"] = c.finished ? c.finish_time : c.race_time;
    o["best_lap"] = c.best_lap_time;
    o["finished"] = c.finished;
    o["retired"] = c.retired;
    cls.push_back(o);
  }
  j["classification"] = cls;
  return j.dump();
}

// --- live stream -----------------------------------------------------------

void Feed::open_stream(const std::string& path, const RaceEnv& env) {
  stream_ = std::make_unique<std::ofstream>(path);
  if (!stream_->is_open()) {
    throw std::runtime_error("cannot open stream for writing: " + path);
  }
  (*stream_) << header_json(env) << "\n";
  stream_->flush();
}

void Feed::close_stream(const RaceEnv& env) {
  if (!stream_ || !stream_->is_open()) return;
  (*stream_) << result_json(env) << "\n";
  stream_->flush();
  stream_->close();
  stream_.reset();
}

// --- replay ----------------------------------------------------------------

void Feed::write(const std::string& dir, const RaceEnv& env) {
  write_track(dir, env.track());

  const std::string frames_path = dir + "/frames.f32";
  std::ofstream fb(frames_path, std::ios::binary);
  if (!fb) throw std::runtime_error("cannot write " + frames_path);
  fb.write(reinterpret_cast<const char*>(frames_.data()),
           static_cast<std::streamsize>(frames_.size() * sizeof(float)));
  fb.close();

  json j = json::parse(header_json(env));
  j.erase("type");
  j["n_frames"] = n_frames_;
  j["frames_file"] = "frames.f32";

  json evs = json::array();
  for (const Event& e : events_) {
    evs.push_back(event_json(e.frame, e.t, e.type, e.car, e.other, e.value, e.lap));
  }
  j["events"] = evs;

  json res = json::parse(result_json(env));
  res.erase("type");
  j["result"] = res;

  const std::string ep_path = dir + "/episode.json";
  std::ofstream fj(ep_path);
  if (!fj) throw std::runtime_error("cannot write " + ep_path);
  fj << j.dump(1);
}

void Feed::write_track(const std::string& dir, const Track& track) {
  json j;
  j["name"] = track.name();
  j["lap_length_m"] = track.length();
  j["ds_m"] = track.ds();
  j["n"] = track.n();

  // A polyline with per-point widths. Deliberately not the engine's own track
  // file: that one carries normals, curvature and a telemetry speed trace which
  // a renderer has no use for, and it is three times the size.
  const auto& xs = track.xs();
  const auto& ys = track.ys();
  const auto& zs = track.zs();
  json line = json::array();
  json wl = json::array();
  json wr = json::array();
  for (size_t i = 0; i < xs.size(); ++i) {
    const double s = static_cast<double>(i) * track.ds();
    line.push_back({xs[i], ys[i], i < zs.size() ? zs[i] : 0.0});
    const double w = track.half_width_at(s);
    wl.push_back(w);
    wr.push_back(w);
  }
  j["line"] = line;
  j["half_width_left"] = wl;
  j["half_width_right"] = wr;
  j["start_finish_index"] = 0;
  j["closed"] = true;

  std::vector<int> sectors;
  sectors.push_back(0);
  for (double s : track.sector_s()) {
    const int idx = static_cast<int>(s / track.ds());
    if (idx > 0 && idx < track.n()) sectors.push_back(idx);
  }
  j["sector_indices"] = sectors;

  const std::string path = dir + "/track.json";
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << j.dump(1);
}

}  // namespace racing
