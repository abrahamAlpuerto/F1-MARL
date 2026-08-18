#include "racing/config.hpp"

#include <fstream>
#include <iterator>
#include <set>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"

namespace racing {
namespace {

using nlohmann::json;

// Read `key` into `dst` and record that it was consumed. Anything left
// unconsumed at the end is a typo -- and a typo in a swept parameter name would
// otherwise quietly run the wrong race while looking like it worked.
template <typename T>
void take(const json& j, const char* key, T* dst, std::set<std::string>* seen) {
  seen->insert(key);
  if (j.contains(key)) *dst = j.at(key).get<T>();
}

void reject_unknown(const json& j, const std::set<std::string>& known,
                    const char* section) {
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (!known.count(it.key())) {
      throw std::runtime_error(std::string("unknown key '") + it.key() +
                               "' in config section '" + section + "'");
    }
  }
}

void load_vehicle(const json& j, VehicleParams* v) {
  std::set<std::string> seen;
  take(j, "mass", &v->mass, &seen);
  take(j, "wheelbase", &v->wheelbase, &seen);
  take(j, "front_weight_frac", &v->front_weight_frac, &seen);
  take(j, "cg_height", &v->cg_height, &seen);
  take(j, "yaw_inertia", &v->yaw_inertia, &seen);
  take(j, "cl_a", &v->cl_a, &seen);
  take(j, "cd_a", &v->cd_a, &seen);
  take(j, "aero_balance_front", &v->aero_balance_front, &seen);
  take(j, "air_density", &v->air_density, &seen);
  take(j, "mu_peak", &v->mu_peak, &seen);
  take(j, "mu_ref_load", &v->mu_ref_load, &seen);
  take(j, "mu_load_sensitivity", &v->mu_load_sensitivity, &seen);
  take(j, "cornering_stiffness_front_per_n", &v->cornering_stiffness_front_per_n, &seen);
  take(j, "cornering_stiffness_rear_per_n", &v->cornering_stiffness_rear_per_n, &seen);
  take(j, "max_lateral_g", &v->max_lateral_g, &seen);
  take(j, "max_power", &v->max_power, &seen);
  take(j, "max_brake_force", &v->max_brake_force, &seen);
  take(j, "drivetrain_rear_frac", &v->drivetrain_rear_frac, &seen);
  take(j, "brake_bias_front", &v->brake_bias_front, &seen);
  take(j, "max_steer", &v->max_steer, &seen);
  take(j, "rolling_resistance", &v->rolling_resistance, &seen);
  take(j, "blend_speed_lo", &v->blend_speed_lo, &seen);
  take(j, "blend_speed_hi", &v->blend_speed_hi, &seen);
  take(j, "length", &v->length, &seen);
  take(j, "width", &v->width, &seen);
  take(j, "track_width", &v->track_width, &seen);
  take(j, "roll_stiffness_front", &v->roll_stiffness_front, &seen);
  take(j, "wheel_radius", &v->wheel_radius, &seen);
  reject_unknown(j, seen, "vehicle");
}

void load_sim(const json& j, SimConfig* s) {
  std::set<std::string> seen;
  take(j, "physics_dt", &s->physics_dt, &seen);
  take(j, "action_repeat", &s->action_repeat, &seen);
  reject_unknown(j, seen, "sim");
}

void load_track(const json& j, TrackConfig* t) {
  std::set<std::string> seen;
  take(j, "path", &t->path, &seen);
  take(j, "laps", &t->laps, &seen);
  take(j, "episode_distance", &t->episode_distance, &seen);
  take(j, "start_s", &t->start_s, &seen);
  take(j, "grid_spacing", &t->grid_spacing, &seen);
  take(j, "grid_stagger", &t->grid_stagger, &seen);
  take(j, "rolling_start", &t->rolling_start, &seen);
  take(j, "rolling_start_speed", &t->rolling_start_speed, &seen);
  take(j, "randomize_start", &t->randomize_start, &seen);
  take(j, "start_jitter_lateral", &t->start_jitter_lateral, &seen);
  take(j, "start_jitter_heading", &t->start_jitter_heading, &seen);
  take(j, "start_speed_lo", &t->start_speed_lo, &seen);
  take(j, "start_speed_hi", &t->start_speed_hi, &seen);
  reject_unknown(j, seen, "track");
}

void load_field(const json& j, FieldConfig* fc) {
  std::set<std::string> seen;
  take(j, "n_teams", &fc->n_teams, &seen);
  take(j, "cars_per_team", &fc->cars_per_team, &seen);
  take(j, "grid_order", &fc->grid_order, &seen);
  reject_unknown(j, seen, "field");
}

void load_aero(const json& j, AeroConfig* a) {
  std::set<std::string> seen;
  take(j, "enabled", &a->enabled, &seen);
  take(j, "tow_max", &a->tow_max, &seen);
  take(j, "wash_max", &a->wash_max, &seen);
  take(j, "decay_length", &a->decay_length, &seen);
  take(j, "width", &a->width, &seen);
  take(j, "range", &a->range, &seen);
  reject_unknown(j, seen, "aero");
}

void load_contact(const json& j, ContactConfig* c) {
  std::set<std::string> seen;
  take(j, "enabled", &c->enabled, &seen);
  take(j, "restitution", &c->restitution, &seen);
  take(j, "speed_loss", &c->speed_loss, &seen);
  take(j, "yaw_kick", &c->yaw_kick, &seen);
  take(j, "separation_gain", &c->separation_gain, &seen);
  reject_unknown(j, seen, "contact");
}

void load_damage(const json& j, DamageConfig* d) {
  std::set<std::string> seen;
  take(j, "enabled", &d->enabled, &seen);
  take(j, "contact_threshold", &d->contact_threshold, &seen);
  take(j, "contact_rate", &d->contact_rate, &seen);
  take(j, "run_off_width", &d->run_off_width, &seen);
  take(j, "impact_speed_full", &d->impact_speed_full, &seen);
  take(j, "barrier_threshold", &d->barrier_threshold, &seen);
  take(j, "barrier_per_hit", &d->barrier_per_hit, &seen);
  take(j, "barrier_restitution", &d->barrier_restitution, &seen);
  take(j, "barrier_speed_loss", &d->barrier_speed_loss, &seen);
  take(j, "retire_threshold", &d->retire_threshold, &seen);
  take(j, "downforce_loss", &d->downforce_loss, &seen);
  take(j, "drag_penalty", &d->drag_penalty, &seen);
  reject_unknown(j, seen, "damage");
}

void load_reward(const json& j, RewardConfig* r) {
  std::set<std::string> seen;
  take(j, "gamma", &r->gamma, &seen);
  take(j, "progress_weight", &r->progress_weight, &seen);
  take(j, "shaping_gamma", &r->shaping_gamma, &seen);
  take(j, "position_weight", &r->position_weight, &seen);
  take(j, "finish_weight", &r->finish_weight, &seen);
  take(j, "team_weight", &r->team_weight, &seen);
  take(j, "off_track_penalty", &r->off_track_penalty, &seen);
  take(j, "contact_penalty", &r->contact_penalty, &seen);
  take(j, "time_penalty", &r->time_penalty, &seen);
  take(j, "off_track_margin", &r->off_track_margin, &seen);
  take(j, "off_track_grip", &r->off_track_grip, &seen);
  take(j, "terminate_off_track", &r->terminate_off_track, &seen);
  take(j, "retire_penalty", &r->retire_penalty, &seen);
  reject_unknown(j, seen, "reward");
}

void load_race(const json& j, RaceConfig* r) {
  std::set<std::string> seen;
  take(j, "recover_after_s", &r->recover_after_s, &seen);
  take(j, "recover_speed", &r->recover_speed, &seen);
  take(j, "recover_distance", &r->recover_distance, &seen);
  take(j, "n_neighbours", &r->n_neighbours, &seen);
  reject_unknown(j, seen, "race");
}


void load_atmosphere(const json& j, AtmosphereConfig* a) {
  std::set<std::string> seen;
  take(j, "enabled", &a->enabled, &seen);
  take(j, "air_temperature_c", &a->air_temperature_c, &seen);
  take(j, "pressure_pa", &a->pressure_pa, &seen);
  take(j, "humidity", &a->humidity, &seen);
  take(j, "track_temperature_c", &a->track_temperature_c, &seen);
  take(j, "wind_speed", &a->wind_speed, &seen);
  take(j, "wind_direction", &a->wind_direction, &seen);
  reject_unknown(j, seen, "atmosphere");
}

void load_fuel(const json& j, FuelConfig* fc) {
  std::set<std::string> seen;
  take(j, "enabled", &fc->enabled, &seen);
  take(j, "start_kg", &fc->start_kg, &seen);
  take(j, "burn_kg_per_km", &fc->burn_kg_per_km, &seen);
  reject_unknown(j, seen, "fuel");
}

void load_tyre(const json& j, TyreConfig* t) {
  std::set<std::string> seen;
  take(j, "enabled", &t->enabled, &seen);
  take(j, "compound", &t->compound, &seen);
  take(j, "start_temperature_c", &t->start_temperature_c, &seen);
  take(j, "cooling_rate", &t->cooling_rate, &seen);
  take(j, "cooling_speed_factor", &t->cooling_speed_factor, &seen);
  take(j, "grip_cold", &t->grip_cold, &seen);
  take(j, "grip_hot", &t->grip_hot, &seen);
  take(j, "grip_worn", &t->grip_worn, &seen);
  take(j, "wear_optimum_shift_c", &t->wear_optimum_shift_c, &seen);
  reject_unknown(j, seen, "tyre");
}

void load_powertrain(const json& j, PowertrainConfig* pt) {
  std::set<std::string> seen;
  take(j, "enabled", &pt->enabled, &seen);
  take(j, "idle_rpm", &pt->idle_rpm, &seen);
  take(j, "redline_rpm", &pt->redline_rpm, &seen);
  take(j, "shift_time_s", &pt->shift_time_s, &seen);
  take(j, "shift_up_frac", &pt->shift_up_frac, &seen);
  take(j, "shift_down_frac", &pt->shift_down_frac, &seen);
  take(j, "final_drive", &pt->final_drive, &seen);
  take(j, "ers_capacity_mj", &pt->ers_capacity_mj, &seen);
  take(j, "ers_deploy_kw", &pt->ers_deploy_kw, &seen);
  take(j, "ers_harvest_kw", &pt->ers_harvest_kw, &seen);
  take(j, "ers_max_deploy_per_lap_mj", &pt->ers_max_deploy_per_lap_mj, &seen);
  take(j, "ers_start_charge", &pt->ers_start_charge, &seen);
  // Arrays are taken wholesale or not at all: a partially overridden gearset is
  // never what anyone means.
  seen.insert("gear_ratios");
  if (j.contains("gear_ratios")) {
    const auto v = j.at("gear_ratios").get<std::vector<double>>();
    if (v.size() != 8) throw std::runtime_error("gear_ratios must have 8 entries");
    for (size_t i = 0; i < 8; ++i) pt->gear_ratios[i] = v[i];
  }
  seen.insert("torque_curve");
  if (j.contains("torque_curve")) {
    const auto v = j.at("torque_curve").get<std::vector<double>>();
    if (v.size() != 7) throw std::runtime_error("torque_curve must have 7 entries");
    for (size_t i = 0; i < 7; ++i) pt->torque_curve[i] = v[i];
  }
  reject_unknown(j, seen, "powertrain");
}

void load_drs(const json& j, DrsConfig* d) {
  std::set<std::string> seen;
  take(j, "enabled", &d->enabled, &seen);
  take(j, "detection_gap_s", &d->detection_gap_s, &seen);
  take(j, "drag_reduction", &d->drag_reduction, &seen);
  take(j, "downforce_loss", &d->downforce_loss, &seen);
  take(j, "zone_min_length_m", &d->zone_min_length_m, &seen);
  take(j, "zone_max_curvature", &d->zone_max_curvature, &seen);
  take(j, "detection_before_m", &d->detection_before_m, &seen);
  reject_unknown(j, seen, "drs");
}

}  // namespace

EnvConfig EnvConfig::from_json_string(const std::string& text) {
  const json j = json::parse(text);
  EnvConfig c;
  std::set<std::string> seen;

  if (j.contains("vehicle")) load_vehicle(j.at("vehicle"), &c.vehicle);
  if (j.contains("sim")) load_sim(j.at("sim"), &c.sim);
  if (j.contains("track")) load_track(j.at("track"), &c.track);
  if (j.contains("field")) load_field(j.at("field"), &c.field);
  if (j.contains("aero")) load_aero(j.at("aero"), &c.aero);
  if (j.contains("contact")) load_contact(j.at("contact"), &c.contact);
  if (j.contains("damage")) load_damage(j.at("damage"), &c.damage);
  if (j.contains("reward")) load_reward(j.at("reward"), &c.reward);
  if (j.contains("race")) load_race(j.at("race"), &c.race);
  if (j.contains("atmosphere")) load_atmosphere(j.at("atmosphere"), &c.atmosphere);
  if (j.contains("fuel")) load_fuel(j.at("fuel"), &c.fuel);
  if (j.contains("tyre")) load_tyre(j.at("tyre"), &c.tyre);
  if (j.contains("powertrain")) load_powertrain(j.at("powertrain"), &c.powertrain);
  if (j.contains("drs")) load_drs(j.at("drs"), &c.drs);
  for (const char* k : {"vehicle", "sim", "track", "field", "aero", "contact",
                        "damage", "reward", "race", "atmosphere", "fuel", "tyre",
                        "powertrain", "drs"}) {
    seen.insert(k);
  }

  take(j, "seed", &c.seed, &seen);
  take(j, "curvature_lookahead", &c.curvature_lookahead, &seen);
  take(j, "lookahead_spacing", &c.lookahead_spacing, &seen);
  reject_unknown(j, seen, "root");
  return c;
}

EnvConfig EnvConfig::from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return from_json_string(ss.str());
}

std::string EnvConfig::to_json_string() const {
  json j;
  j["seed"] = seed;
  j["curvature_lookahead"] = curvature_lookahead;
  j["lookahead_spacing"] = lookahead_spacing;

  auto& v = j["vehicle"];
  v["mass"] = vehicle.mass;
  v["wheelbase"] = vehicle.wheelbase;
  v["front_weight_frac"] = vehicle.front_weight_frac;
  v["cg_height"] = vehicle.cg_height;
  v["yaw_inertia"] = vehicle.yaw_inertia;
  v["cl_a"] = vehicle.cl_a;
  v["cd_a"] = vehicle.cd_a;
  v["aero_balance_front"] = vehicle.aero_balance_front;
  v["air_density"] = vehicle.air_density;
  v["mu_peak"] = vehicle.mu_peak;
  v["mu_ref_load"] = vehicle.mu_ref_load;
  v["mu_load_sensitivity"] = vehicle.mu_load_sensitivity;
  v["cornering_stiffness_front_per_n"] = vehicle.cornering_stiffness_front_per_n;
  v["cornering_stiffness_rear_per_n"] = vehicle.cornering_stiffness_rear_per_n;
  v["max_lateral_g"] = vehicle.max_lateral_g;
  v["max_power"] = vehicle.max_power;
  v["max_brake_force"] = vehicle.max_brake_force;
  v["drivetrain_rear_frac"] = vehicle.drivetrain_rear_frac;
  v["brake_bias_front"] = vehicle.brake_bias_front;
  v["max_steer"] = vehicle.max_steer;
  v["rolling_resistance"] = vehicle.rolling_resistance;
  v["blend_speed_lo"] = vehicle.blend_speed_lo;
  v["blend_speed_hi"] = vehicle.blend_speed_hi;
  v["length"] = vehicle.length;
  v["width"] = vehicle.width;
  v["track_width"] = vehicle.track_width;
  v["roll_stiffness_front"] = vehicle.roll_stiffness_front;
  v["wheel_radius"] = vehicle.wheel_radius;

  auto& s = j["sim"];
  s["physics_dt"] = sim.physics_dt;
  s["action_repeat"] = sim.action_repeat;

  auto& t = j["track"];
  t["path"] = track.path;
  t["laps"] = track.laps;
  t["episode_distance"] = track.episode_distance;
  t["start_s"] = track.start_s;
  t["grid_spacing"] = track.grid_spacing;
  t["grid_stagger"] = track.grid_stagger;
  t["rolling_start"] = track.rolling_start;
  t["rolling_start_speed"] = track.rolling_start_speed;
  t["randomize_start"] = track.randomize_start;
  t["start_jitter_lateral"] = track.start_jitter_lateral;
  t["start_jitter_heading"] = track.start_jitter_heading;
  t["start_speed_lo"] = track.start_speed_lo;
  t["start_speed_hi"] = track.start_speed_hi;

  auto& fc = j["field"];
  fc["n_teams"] = field.n_teams;
  fc["cars_per_team"] = field.cars_per_team;
  fc["grid_order"] = field.grid_order;

  auto& a = j["aero"];
  a["enabled"] = aero.enabled;
  a["tow_max"] = aero.tow_max;
  a["wash_max"] = aero.wash_max;
  a["decay_length"] = aero.decay_length;
  a["width"] = aero.width;
  a["range"] = aero.range;

  auto& ct = j["contact"];
  ct["enabled"] = contact.enabled;
  ct["restitution"] = contact.restitution;
  ct["speed_loss"] = contact.speed_loss;
  ct["yaw_kick"] = contact.yaw_kick;
  ct["separation_gain"] = contact.separation_gain;

  auto& dm = j["damage"];
  dm["enabled"] = damage.enabled;
  dm["contact_threshold"] = damage.contact_threshold;
  dm["contact_rate"] = damage.contact_rate;
  dm["run_off_width"] = damage.run_off_width;
  dm["impact_speed_full"] = damage.impact_speed_full;
  dm["barrier_threshold"] = damage.barrier_threshold;
  dm["barrier_per_hit"] = damage.barrier_per_hit;
  dm["barrier_restitution"] = damage.barrier_restitution;
  dm["barrier_speed_loss"] = damage.barrier_speed_loss;
  dm["retire_threshold"] = damage.retire_threshold;
  dm["downforce_loss"] = damage.downforce_loss;
  dm["drag_penalty"] = damage.drag_penalty;

  auto& r = j["reward"];
  r["gamma"] = reward.gamma;
  r["progress_weight"] = reward.progress_weight;
  r["shaping_gamma"] = reward.shaping_gamma;
  r["position_weight"] = reward.position_weight;
  r["finish_weight"] = reward.finish_weight;
  r["team_weight"] = reward.team_weight;
  r["off_track_penalty"] = reward.off_track_penalty;
  r["contact_penalty"] = reward.contact_penalty;
  r["time_penalty"] = reward.time_penalty;
  r["off_track_margin"] = reward.off_track_margin;
  r["off_track_grip"] = reward.off_track_grip;
  r["terminate_off_track"] = reward.terminate_off_track;
  r["retire_penalty"] = reward.retire_penalty;

  auto& rc = j["race"];
  rc["recover_after_s"] = race.recover_after_s;
  rc["recover_speed"] = race.recover_speed;
  rc["recover_distance"] = race.recover_distance;
  rc["n_neighbours"] = race.n_neighbours;

  auto& at = j["atmosphere"];
  at["enabled"] = atmosphere.enabled;
  at["air_temperature_c"] = atmosphere.air_temperature_c;
  at["pressure_pa"] = atmosphere.pressure_pa;
  at["humidity"] = atmosphere.humidity;
  at["track_temperature_c"] = atmosphere.track_temperature_c;
  at["wind_speed"] = atmosphere.wind_speed;
  at["wind_direction"] = atmosphere.wind_direction;

  auto& fu = j["fuel"];
  fu["enabled"] = fuel.enabled;
  fu["start_kg"] = fuel.start_kg;
  fu["burn_kg_per_km"] = fuel.burn_kg_per_km;

  auto& ty = j["tyre"];
  ty["enabled"] = tyre.enabled;
  ty["compound"] = tyre.compound;
  ty["start_temperature_c"] = tyre.start_temperature_c;
  ty["cooling_rate"] = tyre.cooling_rate;
  ty["cooling_speed_factor"] = tyre.cooling_speed_factor;
  ty["grip_cold"] = tyre.grip_cold;
  ty["grip_hot"] = tyre.grip_hot;
  ty["grip_worn"] = tyre.grip_worn;
  ty["wear_optimum_shift_c"] = tyre.wear_optimum_shift_c;

  auto& pt = j["powertrain"];
  pt["enabled"] = powertrain.enabled;
  pt["idle_rpm"] = powertrain.idle_rpm;
  pt["redline_rpm"] = powertrain.redline_rpm;
  pt["shift_time_s"] = powertrain.shift_time_s;
  pt["shift_up_frac"] = powertrain.shift_up_frac;
  pt["shift_down_frac"] = powertrain.shift_down_frac;
  pt["final_drive"] = powertrain.final_drive;
  pt["gear_ratios"] = std::vector<double>(std::begin(powertrain.gear_ratios),
                                          std::end(powertrain.gear_ratios));
  pt["torque_curve"] = std::vector<double>(std::begin(powertrain.torque_curve),
                                           std::end(powertrain.torque_curve));
  pt["ers_capacity_mj"] = powertrain.ers_capacity_mj;
  pt["ers_deploy_kw"] = powertrain.ers_deploy_kw;
  pt["ers_harvest_kw"] = powertrain.ers_harvest_kw;
  pt["ers_max_deploy_per_lap_mj"] = powertrain.ers_max_deploy_per_lap_mj;
  pt["ers_start_charge"] = powertrain.ers_start_charge;

  auto& dr = j["drs"];
  dr["enabled"] = drs.enabled;
  dr["detection_gap_s"] = drs.detection_gap_s;
  dr["drag_reduction"] = drs.drag_reduction;
  dr["downforce_loss"] = drs.downforce_loss;
  dr["zone_min_length_m"] = drs.zone_min_length_m;
  dr["zone_max_curvature"] = drs.zone_max_curvature;
  dr["detection_before_m"] = drs.detection_before_m;

  return j.dump(2);
}

}  // namespace racing
