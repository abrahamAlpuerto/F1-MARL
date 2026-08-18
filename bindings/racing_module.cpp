// Python bindings for the racing engine.
//
// Two entry points matter, and they are for different jobs.
//
//   VecRaceEnv  training. Takes and fills numpy arrays in place, so a learner
//               crosses the language boundary once per policy step no matter
//               how many races are running.
//   RaceEnv     watching. One race, with the car states and the event list
//               readable from Python, which is what a recorder or a scripted
//               driver needs.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

#include "racing/car.hpp"
#include "racing/config.hpp"
#include "racing/feed.hpp"
#include "racing/atmosphere.hpp"
#include "racing/interaction.hpp"
#include "racing/powertrain.hpp"
#include "racing/qss.hpp"
#include "racing/tyres.hpp"
#include "racing/race.hpp"
#include "racing/track.hpp"
#include "racing/vehicle.hpp"

namespace py = pybind11;
using namespace racing;

namespace {

// Templated on the array type rather than its scalar, because the caller-side
// flag combinations differ and would otherwise fail to deduce.
template <typename Arr>
auto* mutable_ptr(Arr& a, size_t expect, const char* name) {
  if (static_cast<size_t>(a.size()) != expect) {
    throw std::runtime_error(std::string(name) + " has the wrong size: got " +
                             std::to_string(a.size()) + ", expected " +
                             std::to_string(expect));
  }
  return a.mutable_data();
}

}  // namespace

PYBIND11_MODULE(_racing, m) {
  m.doc() = "Teams of AI cars racing each other";

  // --- config --------------------------------------------------------------
  py::class_<VehicleParams>(m, "VehicleParams")
      .def(py::init<>())
      .def_readwrite("mass", &VehicleParams::mass)
      .def_readwrite("wheelbase", &VehicleParams::wheelbase)
      .def_readwrite("cl_a", &VehicleParams::cl_a)
      .def_readwrite("cd_a", &VehicleParams::cd_a)
      .def_readwrite("mu_peak", &VehicleParams::mu_peak)
      .def_readwrite("mu_load_sensitivity", &VehicleParams::mu_load_sensitivity)
      .def_readwrite("max_power", &VehicleParams::max_power)
      .def_readwrite("max_steer", &VehicleParams::max_steer)
      .def_readwrite("max_lateral_g", &VehicleParams::max_lateral_g)
      .def_readwrite("length", &VehicleParams::length)
      .def_readwrite("width", &VehicleParams::width)
      .def_readwrite("track_width", &VehicleParams::track_width)
      .def_readwrite("roll_stiffness_front", &VehicleParams::roll_stiffness_front)
      .def_readwrite("wheel_radius", &VehicleParams::wheel_radius)
      .def_readwrite("mu_ref_load", &VehicleParams::mu_ref_load);

  py::class_<SimConfig>(m, "SimConfig")
      .def(py::init<>())
      .def_readwrite("physics_dt", &SimConfig::physics_dt)
      .def_readwrite("action_repeat", &SimConfig::action_repeat);

  py::class_<TrackConfig>(m, "TrackConfig")
      .def(py::init<>())
      .def_readwrite("path", &TrackConfig::path)
      .def_readwrite("laps", &TrackConfig::laps)
      .def_readwrite("episode_distance", &TrackConfig::episode_distance)
      .def_readwrite("start_s", &TrackConfig::start_s)
      .def_readwrite("grid_spacing", &TrackConfig::grid_spacing)
      .def_readwrite("grid_stagger", &TrackConfig::grid_stagger)
      .def_readwrite("rolling_start", &TrackConfig::rolling_start)
      .def_readwrite("rolling_start_speed", &TrackConfig::rolling_start_speed)
      .def_readwrite("randomize_start", &TrackConfig::randomize_start)
      .def_readwrite("start_jitter_lateral", &TrackConfig::start_jitter_lateral)
      .def_readwrite("start_jitter_heading", &TrackConfig::start_jitter_heading)
      .def_readwrite("start_speed_lo", &TrackConfig::start_speed_lo)
      .def_readwrite("start_speed_hi", &TrackConfig::start_speed_hi);

  py::class_<FieldConfig> field(m, "FieldConfig");
  field.def(py::init<>())
      .def_readwrite("n_teams", &FieldConfig::n_teams)
      .def_readwrite("cars_per_team", &FieldConfig::cars_per_team)
      .def_readwrite("grid_order", &FieldConfig::grid_order)
      .def_property_readonly("n_cars", &FieldConfig::n_cars);
  field.attr("GRID_INTERLEAVE") = int(FieldConfig::GRID_INTERLEAVE);
  field.attr("GRID_BLOCKED") = int(FieldConfig::GRID_BLOCKED);

  py::class_<AeroConfig>(m, "AeroConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &AeroConfig::enabled)
      .def_readwrite("tow_max", &AeroConfig::tow_max)
      .def_readwrite("wash_max", &AeroConfig::wash_max)
      .def_readwrite("decay_length", &AeroConfig::decay_length)
      .def_readwrite("width", &AeroConfig::width)
      .def_readwrite("range", &AeroConfig::range);

  py::class_<ContactConfig>(m, "ContactConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &ContactConfig::enabled)
      .def_readwrite("restitution", &ContactConfig::restitution)
      .def_readwrite("speed_loss", &ContactConfig::speed_loss)
      .def_readwrite("yaw_kick", &ContactConfig::yaw_kick)
      .def_readwrite("separation_gain", &ContactConfig::separation_gain);

  py::class_<DamageConfig>(m, "DamageConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &DamageConfig::enabled)
      .def_readwrite("contact_threshold", &DamageConfig::contact_threshold)
      .def_readwrite("contact_rate", &DamageConfig::contact_rate)
      .def_readwrite("run_off_width", &DamageConfig::run_off_width)
      .def_readwrite("impact_speed_full", &DamageConfig::impact_speed_full)
      .def_readwrite("barrier_threshold", &DamageConfig::barrier_threshold)
      .def_readwrite("barrier_per_hit", &DamageConfig::barrier_per_hit)
      .def_readwrite("barrier_restitution", &DamageConfig::barrier_restitution)
      .def_readwrite("barrier_speed_loss", &DamageConfig::barrier_speed_loss)
      .def_readwrite("retire_threshold", &DamageConfig::retire_threshold)
      .def_readwrite("downforce_loss", &DamageConfig::downforce_loss)
      .def_readwrite("drag_penalty", &DamageConfig::drag_penalty);

  py::class_<RewardConfig>(m, "RewardConfig")
      .def(py::init<>())
      .def_readwrite("gamma", &RewardConfig::gamma)
      .def_readwrite("progress_weight", &RewardConfig::progress_weight)
      .def_readwrite("shaping_gamma", &RewardConfig::shaping_gamma)
      .def_readwrite("position_weight", &RewardConfig::position_weight)
      .def_readwrite("finish_weight", &RewardConfig::finish_weight)
      .def_readwrite("team_weight", &RewardConfig::team_weight)
      .def_readwrite("off_track_penalty", &RewardConfig::off_track_penalty)
      .def_readwrite("contact_penalty", &RewardConfig::contact_penalty)
      .def_readwrite("time_penalty", &RewardConfig::time_penalty)
      .def_readwrite("off_track_margin", &RewardConfig::off_track_margin)
      .def_readwrite("off_track_grip", &RewardConfig::off_track_grip)
      .def_readwrite("terminate_off_track", &RewardConfig::terminate_off_track)
      .def_readwrite("retire_penalty", &RewardConfig::retire_penalty);

  py::class_<RaceConfig>(m, "RaceConfig")
      .def(py::init<>())
      .def_readwrite("recover_after_s", &RaceConfig::recover_after_s)
      .def_readwrite("recover_speed", &RaceConfig::recover_speed)
      .def_readwrite("recover_distance", &RaceConfig::recover_distance)
      .def_readwrite("n_neighbours", &RaceConfig::n_neighbours);


  py::class_<AtmosphereConfig>(m, "AtmosphereConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &AtmosphereConfig::enabled)
      .def_readwrite("air_temperature_c", &AtmosphereConfig::air_temperature_c)
      .def_readwrite("pressure_pa", &AtmosphereConfig::pressure_pa)
      .def_readwrite("humidity", &AtmosphereConfig::humidity)
      .def_readwrite("track_temperature_c", &AtmosphereConfig::track_temperature_c)
      .def_readwrite("wind_speed", &AtmosphereConfig::wind_speed)
      .def_readwrite("wind_direction", &AtmosphereConfig::wind_direction);

  py::class_<FuelConfig>(m, "FuelConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &FuelConfig::enabled)
      .def_readwrite("start_kg", &FuelConfig::start_kg)
      .def_readwrite("burn_kg_per_km", &FuelConfig::burn_kg_per_km);

  py::class_<TyreCompound>(m, "TyreCompound")
      .def_readonly("name", &TyreCompound::name)
      .def_readonly("grip", &TyreCompound::grip)
      .def_readonly("optimum_c", &TyreCompound::optimum_c)
      .def_readonly("window_c", &TyreCompound::window_c)
      .def_readonly("wear_rate", &TyreCompound::wear_rate);

  py::class_<TyreConfig>(m, "TyreConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &TyreConfig::enabled)
      .def_readwrite("compound", &TyreConfig::compound)
      .def_readwrite("start_temperature_c", &TyreConfig::start_temperature_c)
      .def_readwrite("cooling_rate", &TyreConfig::cooling_rate)
      .def_readwrite("cooling_speed_factor", &TyreConfig::cooling_speed_factor)
      .def_readwrite("grip_cold", &TyreConfig::grip_cold)
      .def_readwrite("grip_hot", &TyreConfig::grip_hot)
      .def_readwrite("grip_worn", &TyreConfig::grip_worn);

  py::class_<PowertrainConfig>(m, "PowertrainConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &PowertrainConfig::enabled)
      .def_readwrite("idle_rpm", &PowertrainConfig::idle_rpm)
      .def_readwrite("redline_rpm", &PowertrainConfig::redline_rpm)
      .def_readwrite("shift_time_s", &PowertrainConfig::shift_time_s)
      .def_readwrite("final_drive", &PowertrainConfig::final_drive)
      .def_readwrite("ers_capacity_mj", &PowertrainConfig::ers_capacity_mj)
      .def_readwrite("ers_deploy_kw", &PowertrainConfig::ers_deploy_kw)
      .def_readwrite("ers_harvest_kw", &PowertrainConfig::ers_harvest_kw)
      .def_readwrite("ers_max_deploy_per_lap_mj",
                     &PowertrainConfig::ers_max_deploy_per_lap_mj)
      .def_readwrite("ers_start_charge", &PowertrainConfig::ers_start_charge);

  py::class_<DrsConfig>(m, "DrsConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &DrsConfig::enabled)
      .def_readwrite("detection_gap_s", &DrsConfig::detection_gap_s)
      .def_readwrite("drag_reduction", &DrsConfig::drag_reduction)
      .def_readwrite("downforce_loss", &DrsConfig::downforce_loss)
      .def_readwrite("zone_min_length_m", &DrsConfig::zone_min_length_m)
      .def_readwrite("zone_max_curvature", &DrsConfig::zone_max_curvature)
      .def_readwrite("detection_before_m", &DrsConfig::detection_before_m);

  py::class_<DrsZone>(m, "DrsZone")
      .def_readonly("detection_s", &DrsZone::detection_s)
      .def_readonly("start_s", &DrsZone::start_s)
      .def_readonly("end_s", &DrsZone::end_s);

  py::class_<TyreState>(m, "TyreState")
      .def_readonly("temperature_c", &TyreState::temperature_c)
      .def_readonly("wear", &TyreState::wear)
      .def_readonly("grip", &TyreState::grip)
      .def_readonly("slip_power_w", &TyreState::slip_power_w);

  py::class_<PowertrainState>(m, "PowertrainState")
      .def_readonly("gear", &PowertrainState::gear)
      .def_readonly("rpm", &PowertrainState::rpm)
      .def_readonly("ers_charge_mj", &PowertrainState::ers_charge_mj)
      .def_readonly("ers_deployed_lap_mj", &PowertrainState::ers_deployed_lap_mj)
      .def_readonly("deploying", &PowertrainState::deploying)
      .def_readonly("ice_power_w", &PowertrainState::ice_power_w)
      .def_readonly("ers_power_w", &PowertrainState::ers_power_w);

  m.def("air_density", &air_density, py::arg("atmosphere"),
        "Density of moist air, kg/m^3, from temperature, pressure and humidity.");
  m.def("tyre_compound", &compound_by_index, py::arg("index"),
        py::return_value_policy::reference,
        "0 soft, 1 medium, 2 hard.");

  py::class_<EnvConfig>(m, "EnvConfig")
      .def(py::init<>())
      .def_readwrite("vehicle", &EnvConfig::vehicle)
      .def_readwrite("sim", &EnvConfig::sim)
      .def_readwrite("track", &EnvConfig::track)
      .def_readwrite("field", &EnvConfig::field)
      .def_readwrite("aero", &EnvConfig::aero)
      .def_readwrite("contact", &EnvConfig::contact)
      .def_readwrite("damage", &EnvConfig::damage)
      .def_readwrite("reward", &EnvConfig::reward)
      .def_readwrite("race", &EnvConfig::race)
      .def_readwrite("atmosphere", &EnvConfig::atmosphere)
      .def_readwrite("fuel", &EnvConfig::fuel)
      .def_readwrite("tyre", &EnvConfig::tyre)
      .def_readwrite("powertrain", &EnvConfig::powertrain)
      .def_readwrite("drs", &EnvConfig::drs)
      .def_readwrite("seed", &EnvConfig::seed)
      .def_readwrite("curvature_lookahead", &EnvConfig::curvature_lookahead)
      .def_readwrite("lookahead_spacing", &EnvConfig::lookahead_spacing)
      .def_property_readonly("n_cars", &EnvConfig::n_cars)
      .def_static("from_json_file", &EnvConfig::from_json_file)
      .def_static("from_json_string", &EnvConfig::from_json_string)
      .def("to_json_string", &EnvConfig::to_json_string);

  // --- track and vehicle ---------------------------------------------------
  py::class_<Frenet>(m, "Frenet")
      .def_readonly("s", &Frenet::s)
      .def_readonly("e_y", &Frenet::e_y)
      .def_readonly("e_psi", &Frenet::e_psi);

  py::class_<Track, std::shared_ptr<Track>>(m, "Track")
      .def_static("load", &Track::load)
      .def_property_readonly("length", &Track::length)
      .def_property_readonly("ds", &Track::ds)
      .def_property_readonly("n", &Track::n)
      .def_property_readonly("name", &Track::name)
      .def_property_readonly("reference_lap_time", &Track::reference_lap_time)
      .def("kappa_at", &Track::kappa_at)
      .def("half_width_at", &Track::half_width_at)
      .def("z_at", &Track::z_at)
      .def("telemetry_speed_at", &Track::telemetry_speed_at)
      .def("wrap_s", &Track::wrap_s)
      .def("delta_s", &Track::delta_s)
      .def("project_global", &Track::project_global)
      .def("project", &Track::project, py::arg("x"), py::arg("y"),
           py::arg("heading"), py::arg("s_hint"))
      .def("pose_at",
           [](const Track& t, double s) {
             double x, y, th;
             t.pose_at(s, &x, &y, &th);
             return py::make_tuple(x, y, th);
           },
           py::arg("s"), "Point and tangent heading on the reference line.")
      .def("to_world",
           [](const Track& t, double s, double e_y) {
             double x, y;
             t.to_world(s, e_y, &x, &y);
             return py::make_tuple(x, y);
           },
           py::arg("s"), py::arg("e_y"),
           "Frenet to world: a point `e_y` metres left of the line at `s`.")
      .def("xy",
           [](const Track& t) {
             const int n = t.n();
             py::array_t<double> out({n, 2});
             auto r = out.mutable_unchecked<2>();
             for (int i = 0; i < n; ++i) {
               r(i, 0) = t.xs()[i];
               r(i, 1) = t.ys()[i];
             }
             return out;
           })
      .def("kappas", [](const Track& t) {
        return py::array_t<double>(static_cast<py::ssize_t>(t.kappas().size()),
                                   t.kappas().data());
      });

  py::class_<VehicleState>(m, "VehicleState")
      .def(py::init<>())
      .def_readwrite("x", &VehicleState::x)
      .def_readwrite("y", &VehicleState::y)
      .def_readwrite("psi", &VehicleState::psi)
      .def_readwrite("vx", &VehicleState::vx)
      .def_readwrite("vy", &VehicleState::vy)
      .def_readwrite("r", &VehicleState::r)
      .def_readwrite("ax", &VehicleState::ax)
      .def_property_readonly("speed", &VehicleState::speed);

  py::class_<VehicleTelemetry>(m, "VehicleTelemetry")
      .def_readonly("fz_front", &VehicleTelemetry::fz_front)
      .def_readonly("fz_rear", &VehicleTelemetry::fz_rear)
      .def_readonly("alpha_front", &VehicleTelemetry::alpha_front)
      .def_readonly("alpha_rear", &VehicleTelemetry::alpha_rear)
      .def_readonly("fy_front", &VehicleTelemetry::fy_front)
      .def_readonly("fy_rear", &VehicleTelemetry::fy_rear)
      .def_readonly("downforce", &VehicleTelemetry::downforce)
      .def_readonly("drag", &VehicleTelemetry::drag)
      .def_readonly("front_saturated", &VehicleTelemetry::front_saturated)
      .def_readonly("rear_saturated", &VehicleTelemetry::rear_saturated)
      .def_readonly("fz_fl", &VehicleTelemetry::fz_fl)
      .def_readonly("fz_fr", &VehicleTelemetry::fz_fr)
      .def_readonly("fz_rl", &VehicleTelemetry::fz_rl)
      .def_readonly("fz_rr", &VehicleTelemetry::fz_rr)
      .def_readonly("slip_ratio", &VehicleTelemetry::slip_ratio)
      .def_readonly("wheelspin", &VehicleTelemetry::wheelspin)
      .def_readonly("lockup", &VehicleTelemetry::lockup)
      .def_readonly("slip_power_front", &VehicleTelemetry::slip_power_front)
      .def_readonly("slip_power_rear", &VehicleTelemetry::slip_power_rear)
      .def_readonly("lateral_g", &VehicleTelemetry::lateral_g)
      .def_readonly("longitudinal_g", &VehicleTelemetry::longitudinal_g)
      .def_readonly("mass", &VehicleTelemetry::mass);

  py::class_<Vehicle>(m, "Vehicle")
      .def(py::init<const VehicleParams&>())
      .def("step",
           [](const Vehicle& v, VehicleState& s, double steer, double throttle,
              double dt, double downforce_scale, double drag_scale,
              double grip_scale) {
             VehicleInput u;
             u.steer = steer;
             u.throttle = throttle;
             u.downforce_scale = downforce_scale;
             u.drag_scale = drag_scale;
             u.grip_scale = grip_scale;
             v.step(&s, u, dt);
           },
           py::arg("state"), py::arg("steer"), py::arg("throttle"),
           py::arg("dt"), py::arg("downforce_scale") = 1.0,
           py::arg("drag_scale") = 1.0, py::arg("grip_scale") = 1.0)
      .def("telemetry",
           [](const Vehicle& v, const VehicleState& s, double steer,
              double throttle) {
             VehicleInput u;
             u.steer = steer;
             u.throttle = throttle;
             return v.telemetry(s, u);
           })
      .def_property_readonly("params", &Vehicle::params,
                             py::return_value_policy::reference_internal)
      .def("axle_grip", &Vehicle::axle_grip, py::arg("fz_axle"),
           py::arg("lateral_transfer"))
      .def("max_corner_speed", &Vehicle::max_corner_speed)
      .def("max_long_accel", &Vehicle::max_long_accel)
      .def("max_long_decel", &Vehicle::max_long_decel)
      .def("downforce_at", &Vehicle::downforce_at)
      .def("drag_at", &Vehicle::drag_at);

  py::class_<QssResult>(m, "QssResult")
      .def_readonly("lap_time", &QssResult::lap_time)
      .def_readonly("converged", &QssResult::converged)
      .def_readonly("iterations", &QssResult::iterations)
      .def_property_readonly("speed", [](const QssResult& q) {
        return py::array_t<double>(static_cast<py::ssize_t>(q.speed.size()),
                                   q.speed.data());
      });

  m.def("solve_qss", &solve_qss, py::arg("track"), py::arg("vehicle"),
        py::arg("max_iterations") = 12);

  m.def("wake_strength", &wake_strength, py::arg("aero"), py::arg("gap"),
        py::arg("lateral_offset"),
        "Wake strength a follower `gap` metres behind and `lateral_offset` "
        "metres to the side sees. 1 is directly behind and touching, 0 is "
        "clean air.");

  // --- race ----------------------------------------------------------------
  py::class_<CarState>(m, "CarState")
      .def_readonly("index", &CarState::index)
      .def_readonly("team", &CarState::team)
      .def_readonly("state", &CarState::v)
      .def_readonly("frenet", &CarState::f)
      .def_readonly("distance", &CarState::distance)
      .def_readonly("race_time", &CarState::race_time)
      .def_readonly("lap", &CarState::lap)
      .def_readonly("last_lap_time", &CarState::last_lap_time)
      .def_readonly("best_lap_time", &CarState::best_lap_time)
      .def_readonly("position", &CarState::position)
      .def_readonly("finished", &CarState::finished)
      .def_readonly("finish_time", &CarState::finish_time)
      .def_readonly("off_track", &CarState::off_track)
      .def_readonly("retired", &CarState::retired)
      .def_readonly("downforce_factor", &CarState::downforce_factor)
      .def_readonly("drag_factor", &CarState::drag_factor)
      .def_readonly("wake", &CarState::wake)
      .def_readonly("wake_source", &CarState::wake_source)
      .def_readonly("gap_ahead", &CarState::gap_ahead)
      .def_readonly("gap_behind", &CarState::gap_behind)
      .def_readonly("contact", &CarState::contact)
      .def_readonly("contact_severity", &CarState::contact_severity)
      .def_readonly("damage", &CarState::damage)
      .def_readonly("barrier_impact", &CarState::barrier_impact)
      .def_readonly("off_track_time", &CarState::off_track_time)
      .def_property_readonly(
          "retire_reason",
          [](const CarState& c) { return retire_reason_name(c.retire_reason); })
      .def_readonly("steer", &CarState::steer)
      .def_readonly("throttle", &CarState::throttle)
      .def_readonly("fuel_kg", &CarState::fuel_kg)
      .def_readonly("tyre_front", &CarState::tyre_front)
      .def_readonly("tyre_rear", &CarState::tyre_rear)
      .def_readonly("powertrain", &CarState::powertrain)
      .def_readonly("drs_open", &CarState::drs_open)
      .def_readonly("drs_armed", &CarState::drs_armed)
      .def_readonly("drs_zone", &CarState::drs_zone)
      .def_readonly("slip_ratio", &CarState::slip_ratio)
      .def_readonly("wheelspin", &CarState::wheelspin)
      .def_readonly("lockup", &CarState::lockup)
      .def_readonly("lateral_g", &CarState::lateral_g)
      .def_readonly("longitudinal_g", &CarState::longitudinal_g)
      .def_property_readonly("speed",
                             [](const CarState& c) { return c.v.speed(); });

  py::class_<RaceEvent> ev(m, "RaceEvent");
  ev.def_readonly("type", &RaceEvent::type)
      .def_readonly("time", &RaceEvent::time)
      .def_readonly("car", &RaceEvent::car)
      .def_readonly("other", &RaceEvent::other)
      .def_readonly("value", &RaceEvent::value)
      .def_readonly("lap", &RaceEvent::lap)
      .def_readonly("reason_code", &RaceEvent::reason)
      .def_property_readonly(
          "name", [](const RaceEvent& e) { return event_type_name(e.type); })
      .def_property_readonly(
          "reason",
          [](const RaceEvent& e) { return retire_reason_name(e.reason); });
  ev.attr("OVERTAKE") = int(RaceEvent::OVERTAKE);
  ev.attr("CONTACT") = int(RaceEvent::CONTACT);
  ev.attr("OFF_TRACK") = int(RaceEvent::OFF_TRACK);
  ev.attr("REJOIN") = int(RaceEvent::REJOIN);
  ev.attr("LAP") = int(RaceEvent::LAP);
  ev.attr("FINISH") = int(RaceEvent::FINISH);
  ev.attr("RETIRE") = int(RaceEvent::RETIRE);

  py::class_<StepInfo>(m, "StepInfo")
      .def_readonly("race_time", &StepInfo::race_time)
      .def_readonly("leader_distance", &StepInfo::leader_distance)
      .def_readonly("mean_speed", &StepInfo::mean_speed)
      .def_readonly("n_contacts", &StepInfo::n_contacts)
      .def_readonly("n_overtakes", &StepInfo::n_overtakes)
      .def_readonly("n_off_track", &StepInfo::n_off_track)
      .def_readonly("n_finished", &StepInfo::n_finished)
      .def_readonly("n_retired", &StepInfo::n_retired)
      .def_readonly("mean_damage", &StepInfo::mean_damage)
      .def_readonly("done", &StepInfo::done);

  py::class_<RaceEnv>(m, "RaceEnv")
      .def(py::init([](const EnvConfig& cfg, uint32_t env_idx) {
             auto track = std::make_shared<const Track>(Track::load(cfg.track.path));
             return std::make_unique<RaceEnv>(cfg, track, env_idx);
           }),
           py::arg("config"), py::arg("env_idx") = 0)
      .def_property_readonly("n_cars", &RaceEnv::n_cars)
      .def_property_readonly("obs_dim", &RaceEnv::obs_dim)
      .def_property_readonly("done", &RaceEnv::done)
      .def_property_readonly("race_time", &RaceEnv::race_time)
      .def_property_readonly("race_distance", &RaceEnv::race_distance)
      .def_property_readonly("step_count", &RaceEnv::step_count)
      .def_property_readonly("cars", &RaceEnv::cars,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("teams", &RaceEnv::teams)
      .def_property_readonly("drs_zones", &RaceEnv::drs_zones)
      .def_property_readonly("air_density", &RaceEnv::air_density)
      .def_property_readonly("events", &RaceEnv::events,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("track", &RaceEnv::track,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("config", &RaceEnv::config,
                             py::return_value_policy::reference_internal)
      .def("finish_order", &RaceEnv::finish_order)
      .def("team_scores", &RaceEnv::team_scores)
      .def("reset", &RaceEnv::reset, py::arg("episode") = 0)
      .def("step",
           [](RaceEnv& e,
              py::array_t<float, py::array::c_style | py::array::forcecast> actions) {
             const int nc = e.n_cars();
             if (static_cast<size_t>(actions.size()) != size_t(nc) * 2) {
               throw std::runtime_error("actions must be [n_cars, 2]");
             }
             py::array_t<float> rewards(nc);
             StepInfo info;
             e.step(actions.data(), rewards.mutable_data(), &info);
             return py::make_tuple(rewards, info);
           },
           py::arg("actions"),
           "Step the whole field. Returns (rewards[n_cars], StepInfo).")
      .def("observe", [](const RaceEnv& e) {
        py::array_t<float> out({e.n_cars(), e.obs_dim()});
        e.observe(out.mutable_data());
        return out;
      });

  py::class_<VecRaceEnv> vec(m, "VecRaceEnv");
  vec.def(py::init<const EnvConfig&, int, int>(), py::arg("config"),
          py::arg("n_envs"), py::arg("n_threads") = 1)
      .def_property_readonly("n_envs", &VecRaceEnv::n_envs)
      .def_property_readonly("n_cars", &VecRaceEnv::n_cars)
      .def_property_readonly("obs_dim", &VecRaceEnv::obs_dim)
      .def("reset_all", &VecRaceEnv::reset_all, py::arg("episode") = 0)
      .def("env", py::overload_cast<int>(&VecRaceEnv::env),
           py::return_value_policy::reference_internal)
      .def("observe",
           [](VecRaceEnv& e, py::array_t<float, py::array::c_style> obs) {
             e.observe(mutable_ptr(
                 obs,
                 size_t(e.n_envs()) * e.n_cars() * e.obs_dim(), "obs"));
           })
      .def("states",
           [](const VecRaceEnv& e) {
             py::array_t<float> out(
                 {e.n_envs(), e.n_cars(), int(VecRaceEnv::STATE_FIELDS)});
             e.states(out.mutable_data());
             return out;
           },
           "Privileged per-car state [n_envs, n_cars, 16]. Never part of an "
           "observation -- see the note in race.hpp.")
      .def("step",
           [](VecRaceEnv& e,
              py::array_t<float, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<float, py::array::c_style> rewards,
              py::array_t<uint8_t, py::array::c_style> dones) {
             const size_t n = size_t(e.n_envs());
             const size_t nc = size_t(e.n_cars());
             if (static_cast<size_t>(actions.size()) != n * nc * 2) {
               throw std::runtime_error("actions must be [n_envs, n_cars, 2]");
             }
             float* o = mutable_ptr(obs, n * nc * e.obs_dim(), "obs");
             float* r = mutable_ptr(rewards, n * nc, "rewards");
             uint8_t* d = mutable_ptr(dones, n, "dones");
             const float* a = actions.data();
             // Release the GIL: this is the whole point of batching, and with
             // n_threads > 1 the races step concurrently.
             py::gil_scoped_release release;
             e.step(a, o, r, d, nullptr);
           })
      .def("step_with_info",
           [](VecRaceEnv& e,
              py::array_t<float, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<float, py::array::c_style> rewards,
              py::array_t<uint8_t, py::array::c_style> dones) {
             const size_t n = size_t(e.n_envs());
             const size_t nc = size_t(e.n_cars());
             std::vector<StepInfo> infos(n);
             float* o = mutable_ptr(obs, n * nc * e.obs_dim(), "obs");
             float* r = mutable_ptr(rewards, n * nc, "rewards");
             uint8_t* d = mutable_ptr(dones, n, "dones");
             const float* a = actions.data();
             {
               py::gil_scoped_release release;
               e.step(a, o, r, d, infos.data());
             }
             return infos;
           });
  vec.attr("STATE_FIELDS") = int(VecRaceEnv::STATE_FIELDS);

  // --- feed ----------------------------------------------------------------
  py::class_<TeamInfo>(m, "TeamInfo")
      .def(py::init<>())
      .def(py::init([](std::string name, std::string color) {
             TeamInfo t;
             t.name = std::move(name);
             t.color = std::move(color);
             return t;
           }),
           py::arg("name"), py::arg("color"))
      .def_readwrite("name", &TeamInfo::name)
      .def_readwrite("color", &TeamInfo::color);

  py::class_<Feed> feed(m, "Feed");
  feed.def(py::init<const RaceEnv&, double>(), py::arg("env"),
           py::arg("frame_rate") = 60.0)
      .def("attach", &Feed::attach, py::arg("env"), py::keep_alive<1, 2>(),
           "Capture a frame automatically at every physics step, decimated to "
           "the feed's frame rate.")
      .def("capture", &Feed::capture, py::arg("env"))
      .def("collect_events", &Feed::collect_events, py::arg("env"))
      .def("open_stream", &Feed::open_stream, py::arg("path"), py::arg("env"))
      .def("close_stream", &Feed::close_stream, py::arg("env"))
      .def("write", &Feed::write, py::arg("dir"), py::arg("env"))
      .def_static("write_track", &Feed::write_track, py::arg("dir"),
                  py::arg("track"))
      .def("set_teams", &Feed::set_teams, py::arg("teams"))
      .def("set_car_names", &Feed::set_car_names, py::arg("names"))
      .def_property_readonly("n_frames", &Feed::n_frames)
      .def_property_readonly("n_cars", &Feed::n_cars)
      .def_property_readonly("frame_rate", &Feed::frame_rate)
      .def_property_readonly("streaming", &Feed::streaming);
  feed.attr("FRAME_FIELDS") = int(FRAME_FIELDS);
  feed.attr("VERSION") = int(kFeedVersion);
}
