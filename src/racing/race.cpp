#include "racing/race.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

#include "racing/atmosphere.hpp"
#include "racing/rng.hpp"

namespace racing {

namespace {

// Observation scalings. The point is to hand the network numbers of order one;
// raw SI units span five orders of magnitude here and make the first layer's
// job unnecessarily hard.
constexpr double kSpeedScale = 100.0;     // m/s
constexpr double kLatVelScale = 10.0;     // m/s
constexpr double kYawRateScale = 2.0;     // rad/s
constexpr double kSlipScale = 0.5;        // rad
constexpr double kHeadingScale = 0.7854;  // rad (pi/4)
constexpr double kCurvatureScale = 0.05;  // 1/m, near the tightest corner here
constexpr double kGapScale = 50.0;        // m, for neighbour separations
constexpr double kRelSpeedScale = 20.0;   // m/s, for closing speeds

inline float f(double v) { return static_cast<float>(v); }

inline double wrap_pi(double a) {
  constexpr double kTwoPi = 6.283185307179586;
  a = std::fmod(a + 3.141592653589793, kTwoPi);
  if (a < 0.0) a += kTwoPi;
  return a - 3.141592653589793;
}

}  // namespace

RaceEnv::RaceEnv(EnvConfig cfg, std::shared_ptr<const Track> track,
                 uint32_t env_idx)
    : cfg_(std::move(cfg)),
      track_(std::move(track)),
      vehicle_(cfg_.vehicle),
      powertrain_(cfg_.powertrain, cfg_.vehicle),
      env_idx_(env_idx),
      n_cars_(cfg_.n_cars()) {
  // Both are fixed for the whole race, so they are worked out once here rather
  // than per car per step.
  // Qualified, because RaceEnv has an accessor of the same name and unqualified
  // lookup would find that instead of the free function in atmosphere.hpp.
  air_density_ = cfg_.atmosphere.enabled
                     ? ::racing::air_density(cfg_.atmosphere)
                     : cfg_.vehicle.air_density;
  drs_zones_ = find_drs_zones(*track_, cfg_.drs);

  race_distance_ = cfg_.track.episode_distance > 0.0
                       ? cfg_.track.episode_distance
                       : cfg_.track.laps * track_->length();

  cars_.resize(n_cars_);
  team_of_.resize(n_cars_);
  prev_position_.resize(n_cars_);
  prev_potential_.resize(n_cars_);
  own_reward_.resize(n_cars_);

  for (int i = 0; i < n_cars_; ++i) {
    team_of_[i] = i / std::max(cfg_.field.cars_per_team, 1);
  }

  // A generous ceiling: enough steps to cover the distance at a crawl, so a
  // slow-but-improving field is not cut off, but bounded so a race where
  // everyone has stopped cannot stall a batch forever.
  const double policy_dt = cfg_.sim.physics_dt * cfg_.sim.action_repeat;
  const double crawl_speed = 12.0;  // m/s
  const double longest = race_distance_ +
                         n_cars_ * std::max(cfg_.track.grid_spacing, 0.0);
  max_steps_ = static_cast<int>(longest / (crawl_speed * policy_dt)) + 100;
}

int RaceEnv::obs_dim() const {
  // own dynamics (4) + track relative (2) + curvature lookahead (K)
  // + race context (3) + aero (2) + neighbours (5 each)
  return 4 + 2 + cfg_.curvature_lookahead + 3 + 2 + 5 * cfg_.race.n_neighbours;
}

double RaceEnv::potential(double distance) const {
  return cfg_.reward.progress_weight * distance;
}

void RaceEnv::emit(int type, int car, int other, double value, int lap) {
  RaceEvent e;
  e.type = type;
  e.time = race_time_;
  e.car = car;
  e.other = other;
  e.value = value;
  e.lap = lap;
  events_.push_back(e);
}

// --- retirement ------------------------------------------------------------

void RaceEnv::retire(int car, int reason, int other) {
  CarState& c = cars_[car];
  if (c.retired) return;

  c.retired = true;
  c.retire_reason = reason;

  // The car stops where it stopped. Deliberately not moved out of the way:
  // wherever it came to rest is where a marshal would find it, and teleporting
  // it to the edge of the circuit would put a car in the replay somewhere it
  // never drove. Everything else already skips a retired car -- the wake,
  // contact, the neighbour observation, the DRS gap -- so it is out of the race
  // without needing to be out of the way.
  c.v.vx = 0.0;
  c.v.vy = 0.0;
  c.v.r = 0.0;
  c.drs_open = false;
  c.drs_armed = false;
  c.off_track_time = 0.0;

  just_retired_.push_back(car);
  emit(RaceEvent::RETIRE, car, other, c.damage, c.lap);
  events_.back().reason = reason;
}

// --- the grid --------------------------------------------------------------

void RaceEnv::place_on_grid(uint32_t ep) {
  const double L = track_->length();

  double line_s = track_->wrap_s(cfg_.track.start_s);
  if (cfg_.track.randomize_start) {
    // Move the whole grid, not each car independently: the field has to start
    // as a field, or the first corner is a lottery rather than a race.
    line_s = track_->wrap_s(
        uniform(cfg_.seed, env_idx_, STREAM_RESET_POSE, ep, 0, 0) * L);
  }

  for (int slot = 0; slot < n_cars_; ++slot) {
    // Which car is in this grid slot.
    int idx;
    if (cfg_.field.grid_order == FieldConfig::GRID_BLOCKED) {
      idx = slot;
    } else {
      const int team = slot % cfg_.field.n_teams;
      const int member = slot / cfg_.field.n_teams;
      idx = team * cfg_.field.cars_per_team + member;
    }

    CarState& c = cars_[idx];
    c = CarState{};
    c.index = idx;
    c.team = team_of_[idx];

    const double back = slot * cfg_.track.grid_spacing;
    const double s0 = track_->wrap_s(line_s - back);
    // Alternating sides, the way a real grid staggers.
    double lat = (slot % 2 == 0 ? 1.0 : -1.0) * cfg_.track.grid_stagger;

    double v0 = std::max(10.0, track_->telemetry_speed_at(s0));
    v0 *= cfg_.track.rolling_start ? cfg_.track.rolling_start_speed : 0.12;

    double dpsi = 0.0;
    if (cfg_.track.randomize_start) {
      v0 = std::max(10.0, track_->telemetry_speed_at(s0)) *
           uniform_range(cfg_.track.start_speed_lo, cfg_.track.start_speed_hi,
                         cfg_.seed, env_idx_, STREAM_RESET_SPEED, ep,
                         static_cast<uint32_t>(idx), 0);
      lat += uniform_range(-cfg_.track.start_jitter_lateral,
                           cfg_.track.start_jitter_lateral, cfg_.seed, env_idx_,
                           STREAM_RESET_POSE, ep, static_cast<uint32_t>(idx), 1);
      dpsi = uniform_range(-cfg_.track.start_jitter_heading,
                           cfg_.track.start_jitter_heading, cfg_.seed, env_idx_,
                           STREAM_RESET_POSE, ep, static_cast<uint32_t>(idx), 2);
    }

    const double half_w = track_->half_width_at(s0);
    lat = std::clamp(lat, -0.75 * half_w, 0.75 * half_w);

    double x, y, th;
    track_->pose_at(s0, &x, &y, &th);
    track_->to_world(s0, lat, &x, &y);

    c.v.x = x;
    c.v.y = y;
    c.v.psi = th + dpsi;
    c.v.vx = v0;
    c.f = track_->project_global(c.v.x, c.v.y, c.v.psi);

    // --- what the car sets off carrying ------------------------------------
    c.fuel_kg = cfg_.fuel.enabled ? cfg_.fuel.start_kg : 0.0;
    c.tyre_front = TyreState{};
    c.tyre_rear = TyreState{};
    c.tyre_front.temperature_c = cfg_.tyre.start_temperature_c;
    c.tyre_rear.temperature_c = cfg_.tyre.start_temperature_c;
    c.tyre_front.grip = tyre_grip(cfg_.tyre, c.tyre_front);
    c.tyre_rear.grip = tyre_grip(cfg_.tyre, c.tyre_rear);
    powertrain_.reset(&c.powertrain);
    c.powertrain.gear = powertrain_.select_gear(c.v.vx, 2);

    // Distance is measured from the start/finish line, so a car on the back
    // of the grid starts this far behind it and everyone finishes at the same
    // place having covered the same distance PAST THE LINE. Measuring from
    // where each car happened to be parked instead would drop the flag in
    // eight different places on the circuit.
    c.distance = -back;
  }
}

void RaceEnv::reset(uint64_t episode) {
  episode_ = episode;
  done_ = false;
  step_count_ = 0;
  race_time_ = 0.0;
  n_finished_ = 0;
  events_.clear();
  contacts_.clear();
  prev_touching_.clear();

  place_on_grid(static_cast<uint32_t>(episode));

  apply_wake(cfg_.aero, *track_, &cars_);
  classify(&cars_);

  // Seed the pairwise order from the grid, so the first move each pair makes is
  // judged against where they started rather than reported as a pass. Seeded
  // from grid position, not from distance: every car has covered exactly zero
  // metres at this point, so distance cannot tell them apart.
  order_sign_.assign(static_cast<size_t>(n_cars_) * n_cars_, 0);
  for (int i = 0; i < n_cars_; ++i) {
    for (int j = i + 1; j < n_cars_; ++j) {
      order_sign_[static_cast<size_t>(i) * n_cars_ + j] =
          cars_[i].position < cars_[j].position ? static_cast<int8_t>(1)
                                                : static_cast<int8_t>(-1);
    }
  }

  for (int i = 0; i < n_cars_; ++i) {
    prev_position_[i] = cars_[i].position;
    prev_potential_[i] = potential(0.0);
  }
}

// --- physics ---------------------------------------------------------------

void RaceEnv::physics_step(double dt) {
  // The wake is recomputed before every integration step rather than once per
  // policy step. It changes fast -- a car pulling out of a tow at 80 m/s moves
  // most of a car's width in four physics steps -- and holding it fixed across
  // the policy step made the moment of pulling out feel laggy.
  apply_wake(cfg_.aero, *track_, &cars_);

  for (int i = 0; i < n_cars_; ++i) {
    CarState& c = cars_[i];
    if (c.retired) continue;

    const double speed = c.v.speed();

    VehicleInput u;
    u.steer = c.steer;
    u.throttle = c.throttle;

    // --- the air ----------------------------------------------------------
    // The wake, then DRS on top of it. They multiply rather than replace each
    // other: a car in a tow that also has the wing open gets both, which is
    // exactly the combination that completes an overtake.
    u.downforce_scale = c.downforce_factor;
    u.drag_scale = c.drag_factor;
    if (c.drs_open) {
      u.drag_scale *= 1.0 - cfg_.drs.drag_reduction;
      u.downforce_scale *= 1.0 - cfg_.drs.downforce_loss;
    }
    // Damage, on top of both. A car that has lost bodywork is slower in the
    // corners and draggier down the straight, which is the whole reason partial
    // damage is worth modelling rather than just counting down to a DNF.
    if (cfg_.damage.enabled && c.damage > 0.0) {
      u.downforce_scale *= 1.0 - cfg_.damage.downforce_loss * c.damage;
      u.drag_scale *= 1.0 + cfg_.damage.drag_penalty * c.damage;
    }
    u.air_density = air_density_;
    u.headwind = headwind_component(cfg_.atmosphere, c.v.psi);

    // --- the road ---------------------------------------------------------
    u.grip_scale = c.off_track ? cfg_.reward.off_track_grip : 1.0;
    u.grade = track_->grade_at(c.f.s);

    // --- what the car is carrying -----------------------------------------
    u.extra_mass = c.fuel_kg;
    u.grip_front = c.tyre_front.grip;
    u.grip_rear = c.tyre_rear.grip;

    // --- what the powertrain can give -------------------------------------
    // Worked out from last step's state, which is the same one-step lag the
    // load transfer already runs on. At 100 Hz it is far below the model's own
    // fidelity, and it keeps the step a single pass.
    u.drive_force = cfg_.powertrain.enabled
                        ? powertrain_.drive_force(c.powertrain, speed)
                        : -1.0;

    const double s_before = c.f.s;
    VehicleTelemetry tel;
    vehicle_.step(&c.v, u, dt, &tel);

    const Frenet prev = c.f;
    c.f = track_->project(c.v.x, c.v.y, c.v.psi, prev.s);

    // Progress is the signed advance around the loop, so crossing the
    // start/finish line does not read as five kilometres of instant backward
    // travel.
    const double advance = track_->delta_s(c.f.s, s_before);
    c.distance += advance;
    c.race_time += dt;

    // --- consumables -------------------------------------------------------
    // Tyres are heated and worn by the frictional work the contact patches are
    // actually doing, which the force solve already computed. A car that is
    // sliding to keep up in dirty air pays for it here.
    update_tyre(cfg_.tyre, cfg_.atmosphere, &c.tyre_front, tel.slip_power_front,
                speed, dt, /*is_front=*/true);
    update_tyre(cfg_.tyre, cfg_.atmosphere, &c.tyre_rear, tel.slip_power_rear,
                speed, dt, /*is_front=*/false);

    // The brakes are what there is to harvest from, so the hybrid is fed the
    // power they are dissipating.
    const double braking_w =
        c.throttle < 0.0
            ? std::abs(tel.fx_front + tel.fx_rear) * speed
            : 0.0;
    powertrain_.update(&c.powertrain, speed, std::max(c.throttle, 0.0),
                       braking_w, dt);

    if (cfg_.fuel.enabled) {
      c.fuel_kg = std::max(
          0.0, c.fuel_kg - cfg_.fuel.burn_kg_per_km * std::max(advance, 0.0) * 1e-3);
    }

    c.slip_ratio = tel.slip_ratio;
    c.wheelspin = tel.wheelspin;
    c.lockup = tel.lockup;
    c.lateral_g = tel.lateral_g;
    c.longitudinal_g = tel.longitudinal_g;

    update_drs(&c, s_before);
  }

  resolve_contacts(cfg_.contact, cfg_.damage, cfg_.vehicle, *track_, dt, &cars_,
                   &contacts_);

  // The wall, at the physics rate. A car crossing the run-off at 80 m/s covers
  // three metres per physics step and twelve per policy step, so this is the
  // slowest it can run and still catch the impact near where it happened.
  resolve_barriers(cfg_.damage, *track_, &cars_, &barrier_hits_);

  // Retirement is decided here, in the same step the damage was done, so a car
  // that has just been written off does not get another physics step of driving
  // out of it. Both sources feed one threshold: it does not matter to a broken
  // car whether the wall or another car broke it.
  if (cfg_.damage.enabled) {
    for (int i = 0; i < n_cars_; ++i) {
      CarState& c = cars_[i];
      if (c.retired || c.finished) continue;
      if (c.damage < cfg_.damage.retire_threshold) continue;
      // The wall gets the blame if it was involved this step, because a hit
      // hard enough to retire a car is the thing a viewer just watched.
      retire(i, c.barrier_impact > 0.0 ? RETIRE_BARRIER : RETIRE_COLLISION, -1);
    }
  }

  // Positions and gaps are refreshed at the physics rate rather than the policy
  // rate. It costs almost nothing at this field size, and it means the timing
  // information in the feed moves as smoothly as the cars do.
  classify(&cars_);

  race_time_ += dt;
  if (observer_) observer_(*this);
}

// --- DRS -------------------------------------------------------------------

void RaceEnv::update_drs(CarState* c, double prev_s) {
  if (!cfg_.drs.enabled || drs_zones_.empty()) {
    c->drs_open = false;
    c->drs_zone = -1;
    return;
  }

  // Arm the wing if the car crossed a detection point this step while close
  // enough to whoever is in front of it ON THE ROAD.
  //
  // Detection is deliberately a point rather than a state: being within a
  // second somewhere down the straight is too late, and a rule written that way
  // would let a car that has already been passed re-open its wing. Following
  // closely through the corner BEFORE the straight is what earns it.
  for (const DrsZone& z : drs_zones_) {
    const double before = track_->delta_s(z.detection_s, prev_s);
    const double after = track_->delta_s(z.detection_s, c->f.s);
    // Crossed if the detection point went from ahead of us to behind us.
    if (before >= 0.0 && after < 0.0) {
      const double gap = gap_to_car_ahead(*track_, cars_, c->index);
      c->drs_armed = gap <= cfg_.drs.detection_gap_s;
    }
  }

  const int was_in = c->drs_zone;
  c->drs_zone = zone_at(drs_zones_, *track_, c->f.s);
  c->drs_open = c->drs_armed && c->drs_zone >= 0;

  // Disarm on LEAVING a zone, not merely on being outside one. The detection
  // point sits before the zone starts, so a car is out of any zone for the
  // stretch between the two -- disarming there would close the wing before it
  // ever opened, and DRS would do nothing at all.
  if (was_in >= 0 && c->drs_zone < 0) c->drs_armed = false;
}

// --- one policy step -------------------------------------------------------

void RaceEnv::step(const float* actions, float* rewards, StepInfo* info) {
  events_.clear();
  contacts_.clear();
  barrier_hits_.clear();
  just_retired_.clear();

  for (int i = 0; i < n_cars_; ++i) {
    cars_[i].steer = std::clamp(static_cast<double>(actions[i * 2 + 0]), -1.0, 1.0);
    cars_[i].throttle = std::clamp(static_cast<double>(actions[i * 2 + 1]), -1.0, 1.0);
  }

  const double dt = cfg_.sim.physics_dt;

  std::vector<char> off_before(n_cars_);
  for (int i = 0; i < n_cars_; ++i) off_before[i] = cars_[i].off_track ? 1 : 0;
  std::vector<int> just_finished;

  for (int k = 0; k < cfg_.sim.action_repeat; ++k) physics_step(dt);

  const double policy_dt = dt * cfg_.sim.action_repeat;

  // --- track limits, recovery, laps, finishing -----------------------------
  for (int i = 0; i < n_cars_; ++i) {
    CarState& c = cars_[i];
    if (c.retired) continue;

    const double half_w = track_->half_width_at(c.f.s);
    const double excess = std::abs(c.f.e_y) - half_w;
    c.off_track = excess > cfg_.reward.off_track_margin;
    c.off_track_time = c.off_track ? c.off_track_time + policy_dt : 0.0;

    if (c.off_track && !off_before[i]) emit(RaceEvent::OFF_TRACK, i, -1, excess, c.lap);
    if (!c.off_track && off_before[i]) emit(RaceEvent::REJOIN, i, -1, 0.0, c.lap);

    // A car pointing the wrong way is not recoverable in any interesting sense.
    const bool wrong_way = std::abs(wrap_pi(c.f.e_psi)) > 1.9;

    if (cfg_.reward.terminate_off_track && (c.off_track || wrong_way)) {
      // Training mode: a car that leaves the circuit is done. This is what
      // stops a policy learning to cut corners, and stops it spending samples
      // driving through the desert.
      retire(i, RETIRE_OFF_TRACK, -1);
      continue;
    }

    // Racing mode: put a stranded car back on the circuit rather than ending
    // anyone's afternoon over it.
    const bool stranded = excess > cfg_.race.recover_distance ||
                          (wrong_way && c.off_track);
    if (stranded && c.off_track_time > cfg_.race.recover_after_s) {
      double x, y, th;
      track_->pose_at(c.f.s, &x, &y, &th);
      const double lat = std::clamp(c.f.e_y, -0.5 * half_w, 0.5 * half_w);
      track_->to_world(c.f.s, lat, &x, &y);
      c.v = VehicleState{};
      c.v.x = x;
      c.v.y = y;
      c.v.psi = th;
      c.v.vx = cfg_.race.recover_speed;
      c.f = track_->project_global(x, y, th);
      c.off_track = false;
      c.off_track_time = 0.0;
      emit(RaceEvent::REJOIN, i, -1, 1.0, c.lap);
    }

    // --- laps -------------------------------------------------------------
    const double L = track_->length();
    const int laps_done =
        c.distance >= 0.0 ? static_cast<int>(c.distance / L) : 0;
    while (c.lap < laps_done) {
      ++c.lap;
      c.last_lap_time = c.race_time - c.lap_start_time;
      c.lap_start_time = c.race_time;
      if (c.best_lap_time <= 0.0 || c.last_lap_time < c.best_lap_time) {
        c.best_lap_time = c.last_lap_time;
      }
      powertrain_.new_lap(&c.powertrain);
      emit(RaceEvent::LAP, i, -1, c.last_lap_time, c.lap);
    }

    if (!c.finished && c.distance >= race_distance_) {
      c.finished = true;
      c.finish_time = c.race_time;
      ++n_finished_;
      just_finished.push_back(i);
    }
  }

  classify(&cars_);

  // --- events that need the new classification -----------------------------
  // The flag, at the moment the car takes it and with the position it took it
  // in. Emitted here rather than at the end of the race so a viewer sees each
  // car finish as it happens.
  for (int i : just_finished) {
    emit(RaceEvent::FINISH, i, -1, cars_[i].position, cars_[i].lap);
  }
  // Overtakes, with hysteresis.
  //
  // Reporting every position swap sounds right and is not: two cars running
  // nose to tail cross each other's distance dozens of times a lap as they
  // breathe in and out of each other's wake, and a raw comparison called all of
  // them a pass. One lap of eight cars produced 152 "overtakes". A viewer
  // watching that leaderboard sees a slot machine.
  //
  // So the order of each pair is only considered to have changed once one car
  // is clear of the other by `kOvertakeMargin`, and it has to get that far the
  // other way before it can flip back. A Schmitt trigger, in other words. The
  // margin is well under a car length, so a genuine pass still reports the
  // moment it is real.
  constexpr double kOvertakeMargin = 1.5;  // m
  for (int i = 0; i < n_cars_; ++i) {
    for (int j = i + 1; j < n_cars_; ++j) {
      const size_t k = static_cast<size_t>(i) * n_cars_ + j;
      const double d = cars_[i].distance - cars_[j].distance;
      if (d > kOvertakeMargin && order_sign_[k] <= 0) {
        if (!cars_[i].retired && !cars_[j].retired) {
          emit(RaceEvent::OVERTAKE, i, j, 0.0, cars_[i].lap);
        }
        order_sign_[k] = 1;
      } else if (d < -kOvertakeMargin && order_sign_[k] >= 0) {
        if (!cars_[i].retired && !cars_[j].retired) {
          emit(RaceEvent::OVERTAKE, j, i, 0.0, cars_[j].lap);
        }
        order_sign_[k] = -1;
      }
    }
  }
  // Contact is reported once per touch, not once per physics step. Two cars
  // rubbing along each other for half a second overlap on fifty consecutive
  // steps; reporting each one buries the event log and makes the replay look
  // like a fifty-car pileup. Collapse them per pair, keep the worst severity,
  // and only report a pair that was not already touching last step.
  std::vector<uint64_t> touching;
  touching.reserve(contacts_.size());
  for (const Contact& ct : contacts_) {
    const int lo = std::min(ct.a, ct.b), hi = std::max(ct.a, ct.b);
    const uint64_t key = uint64_t(lo) * 1024u + uint64_t(hi);
    if (std::find(touching.begin(), touching.end(), key) != touching.end()) {
      continue;
    }
    touching.push_back(key);

    double worst = 0.0;
    for (const Contact& other : contacts_) {
      const int a = std::min(other.a, other.b), b = std::max(other.a, other.b);
      if (uint64_t(a) * 1024u + uint64_t(b) == key) {
        worst = std::max(worst, other.severity);
      }
    }
    const bool was = std::find(prev_touching_.begin(), prev_touching_.end(),
                               key) != prev_touching_.end();
    if (!was) emit(RaceEvent::CONTACT, ct.a, ct.b, worst, cars_[ct.a].lap);
  }
  prev_touching_ = std::move(touching);

  // --- reward ---------------------------------------------------------------
  const bool race_over =
      n_finished_ + static_cast<int>(std::count_if(
                        cars_.begin(), cars_.end(),
                        [](const CarState& c) { return c.retired; })) >= n_cars_;
  const bool timed_out = ++step_count_ >= max_steps_;
  done_ = race_over || timed_out;

  for (int i = 0; i < n_cars_; ++i) {
    const CarState& c = cars_[i];
    double r = 0.0;

    if (!c.retired && !c.finished) {
      // Progress: potential-based shaping. With shaping_gamma = 1 this
      // telescopes to progress_weight * metres covered.
      const double phi = potential(c.distance);
      r += cfg_.reward.shaping_gamma * phi - prev_potential_[i];

      // Position: the term that makes this a race rather than eight parallel
      // time trials. Paid on the change, so a place gained is worth exactly
      // what a place lost costs.
      r += cfg_.reward.position_weight * (prev_position_[i] - c.position);

      if (c.off_track) r -= cfg_.reward.off_track_penalty;
      if (c.contact) r -= cfg_.reward.contact_penalty * c.contact_severity;
      r -= cfg_.reward.time_penalty;
    }
    // Charged on the step the car went out, and only then -- which is why it is
    // keyed off `just_retired_` rather than off `c.retired`, a flag that stays
    // true for the rest of the race and would bill every remaining step.
    if (std::find(just_retired_.begin(), just_retired_.end(), i) !=
        just_retired_.end()) {
      r -= cfg_.reward.retire_penalty;
    }
    prev_potential_[i] = potential(c.distance);

    // The flag. Paid once, on the step the race ends, so that the last lap is
    // worth as much as it should be.
    if (done_) {
      r += cfg_.reward.finish_weight * (n_cars_ - c.position);
    }
    own_reward_[i] = static_cast<float>(r);
  }

  // --- team mixing ---------------------------------------------------------
  // A car receives a blend of its own reward and its team's mean. This is the
  // whole cooperative mechanism: at team_weight 0.5 a car that gives up a place
  // to let a team mate through loses one unit of position reward and gains half
  // of what the team mate gained, so the move is worth making exactly when the
  // team mate had more to gain than it did.
  const double lam = std::clamp(cfg_.reward.team_weight, 0.0, 1.0);
  const int per_team = std::max(cfg_.field.cars_per_team, 1);
  std::vector<double> team_sum(cfg_.field.n_teams, 0.0);
  for (int i = 0; i < n_cars_; ++i) team_sum[team_of_[i]] += own_reward_[i];

  for (int i = 0; i < n_cars_; ++i) {
    const double team_mean = team_sum[team_of_[i]] / per_team;
    rewards[i] = static_cast<float>((1.0 - lam) * own_reward_[i] + lam * team_mean);
  }

  for (int i = 0; i < n_cars_; ++i) prev_position_[i] = cars_[i].position;

  if (info) {
    double speed_sum = 0.0, leader = 0.0, damage_sum = 0.0;
    int n_off = 0, n_out = 0;
    for (const CarState& c : cars_) {
      speed_sum += c.v.speed();
      leader = std::max(leader, c.distance);
      damage_sum += c.damage;
      if (c.off_track) ++n_off;
      if (c.retired) ++n_out;
    }
    info->race_time = race_time_;
    info->leader_distance = leader;
    info->mean_speed = n_cars_ > 0 ? speed_sum / n_cars_ : 0.0;
    info->n_contacts = static_cast<int>(contacts_.size());
    info->n_overtakes = static_cast<int>(std::count_if(
        events_.begin(), events_.end(),
        [](const RaceEvent& e) { return e.type == RaceEvent::OVERTAKE; }));
    info->n_off_track = n_off;
    info->n_finished = n_finished_;
    info->n_retired = n_out;
    info->mean_damage = n_cars_ > 0 ? damage_sum / n_cars_ : 0.0;
    info->done = done_;
  }
}

// --- observation -----------------------------------------------------------

void RaceEnv::observe(float* out) const {
  const int d = obs_dim();
  const int nn = cfg_.race.n_neighbours;

  for (int i = 0; i < n_cars_; ++i) {
    const CarState& c = cars_[i];
    float* o = out + static_cast<size_t>(i) * d;
    int k = 0;

    // --- own dynamics ------------------------------------------------------
    const double v = c.v.speed();
    const double slip =
        v > 1.0 ? std::atan2(c.v.vy, std::max(c.v.vx, 0.1)) : 0.0;
    o[k++] = f(c.v.vx / kSpeedScale);
    o[k++] = f(c.v.vy / kLatVelScale);
    o[k++] = f(c.v.r / kYawRateScale);
    o[k++] = f(slip / kSlipScale);

    // --- track relative ----------------------------------------------------
    const double half_w = std::max(track_->half_width_at(c.f.s), 0.1);
    o[k++] = f(c.f.e_y / half_w);  // +/-1 at the corridor edge
    o[k++] = f(c.f.e_psi / kHeadingScale);

    // --- curvature ahead ---------------------------------------------------
    // The single most important thing the policy sees: without it, braking for
    // a corner can only ever be reactive, and by then the corner has happened.
    // Starts at i = 0, the curvature the car is in right now, because the
    // policy needs to know what it is already committed to as well as what is
    // coming.
    for (int q = 0; q < cfg_.curvature_lookahead; ++q) {
      o[k++] = f(track_->kappa_at(c.f.s + q * cfg_.lookahead_spacing) /
                 kCurvatureScale);
    }

    // --- race context ------------------------------------------------------
    const double remaining = std::max(0.0, race_distance_ - c.distance);
    o[k++] = f(remaining / std::max(race_distance_, 1.0));
    o[k++] = f(n_cars_ > 1 ? double(c.position - 1) / (n_cars_ - 1) : 0.0);
    o[k++] = f(c.f.s / track_->length());

    // --- what the air is doing ---------------------------------------------
    // Given to the policy directly rather than left to be inferred. A car can
    // feel the grip it has; withholding it would only make the problem
    // partially observable for no reason.
    o[k++] = f(c.downforce_factor);
    o[k++] = f(c.drag_factor);

    // --- neighbours --------------------------------------------------------
    // The nearest few cars by along-track gap, closest first. Signed, so the
    // policy can tell the car it is chasing from the one chasing it, and
    // flagged by team, so it knows whose race it is helping.
    std::vector<std::pair<double, int>> near;
    near.reserve(n_cars_);
    for (int j = 0; j < n_cars_; ++j) {
      if (j == i || cars_[j].retired) continue;
      near.emplace_back(track_->delta_s(cars_[j].f.s, c.f.s), j);
    }
    std::sort(near.begin(), near.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                if (std::abs(a.first) != std::abs(b.first)) {
                  return std::abs(a.first) < std::abs(b.first);
                }
                return a.second < b.second;  // deterministic on an exact tie
              });

    for (int q = 0; q < nn; ++q) {
      if (q < static_cast<int>(near.size())) {
        const int j = near[q].second;
        const CarState& other = cars_[j];
        o[k++] = f(std::clamp(near[q].first / kGapScale, -4.0, 4.0));
        o[k++] = f((other.f.e_y - c.f.e_y) / half_w);
        o[k++] = f(std::clamp((other.v.speed() - v) / kRelSpeedScale, -4.0, 4.0));
        o[k++] = f(other.team == c.team ? 1.0 : 0.0);
        o[k++] = 1.0f;  // this slot holds a car
      } else {
        o[k++] = 0.0f;
        o[k++] = 0.0f;
        o[k++] = 0.0f;
        o[k++] = 0.0f;
        o[k++] = 0.0f;  // nobody there
      }
    }
  }
}

// --- results ---------------------------------------------------------------

std::vector<int> RaceEnv::finish_order() const {
  std::vector<int> order(n_cars_);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return cars_[a].position < cars_[b].position;
  });
  return order;
}

std::vector<int> RaceEnv::team_scores() const {
  std::vector<int> score(cfg_.field.n_teams, 0);
  for (int i = 0; i < n_cars_; ++i) {
    score[team_of_[i]] += n_cars_ - cars_[i].position;
  }
  return score;
}

// --- batched ---------------------------------------------------------------

VecRaceEnv::VecRaceEnv(const EnvConfig& cfg, int n_envs, int n_threads)
    : n_threads_(std::max(1, n_threads)) {
  track_ = std::make_shared<const Track>(Track::load(cfg.track.path));
  envs_.reserve(n_envs);
  for (int i = 0; i < n_envs; ++i) {
    envs_.push_back(
        std::make_unique<RaceEnv>(cfg, track_, static_cast<uint32_t>(i)));
  }
  episode_.assign(n_envs, 0);
}

void VecRaceEnv::reset_all(uint64_t episode) {
  for (size_t i = 0; i < envs_.size(); ++i) {
    episode_[i] = episode;
    envs_[i]->reset(episode);
  }
}

void VecRaceEnv::observe(float* obs) const {
  const size_t stride = static_cast<size_t>(n_cars()) * obs_dim();
  for (size_t i = 0; i < envs_.size(); ++i) envs_[i]->observe(obs + i * stride);
}

void VecRaceEnv::states(float* out) const {
  const int nc = n_cars();
  for (size_t e = 0; e < envs_.size(); ++e) {
    const RaceEnv& env = *envs_[e];
    for (int i = 0; i < nc; ++i) {
      const CarState& c = env.cars()[i];
      float* o = out + (e * nc + i) * STATE_FIELDS;
      o[STATE_X] = f(c.v.x);
      o[STATE_Y] = f(c.v.y);
      o[STATE_Z] = f(track_->z_at(c.f.s));
      o[STATE_PSI] = f(c.v.psi);
      o[STATE_VX] = f(c.v.vx);
      o[STATE_VY] = f(c.v.vy);
      o[STATE_R] = f(c.v.r);
      o[STATE_S] = f(c.f.s);
      o[STATE_E_Y] = f(c.f.e_y);
      o[STATE_DISTANCE] = f(c.distance);
      o[STATE_LAP] = f(c.lap);
      o[STATE_POSITION] = f(c.position);
      o[STATE_DOWNFORCE] = f(c.downforce_factor);
      o[STATE_DRAG] = f(c.drag_factor);
      o[STATE_GAP_AHEAD] = f(c.gap_ahead);
      o[STATE_TEAM] = f(c.team);
    }
  }
}

void VecRaceEnv::step(const float* actions, float* obs, float* rewards,
                      uint8_t* dones, StepInfo* infos) {
  const int n = static_cast<int>(envs_.size());
  const int nc = n_cars();
  const size_t obs_stride = static_cast<size_t>(nc) * obs_dim();
  const size_t act_stride = static_cast<size_t>(nc) * 2;

  auto run = [&](int lo, int hi) {
    for (int i = lo; i < hi; ++i) {
      StepInfo info;
      envs_[i]->step(actions + i * act_stride, rewards + i * nc, &info);
      dones[i] = envs_[i]->done() ? 1u : 0u;
      if (infos) infos[i] = info;

      // Auto-reset, so the caller never has to special-case a finished race
      // mid-batch. Every race advances its own episode counter, which keys its
      // draws, so this stays reproducible regardless of which races happened to
      // finish when.
      if (dones[i]) {
        ++episode_[i];
        envs_[i]->reset(episode_[i]);
      }
      envs_[i]->observe(obs + i * obs_stride);
    }
  };

  if (n_threads_ <= 1 || n < 2 * n_threads_) {
    run(0, n);
    return;
  }

  std::vector<std::thread> pool;
  pool.reserve(n_threads_);
  const int chunk = (n + n_threads_ - 1) / n_threads_;
  for (int t = 0; t < n_threads_; ++t) {
    const int lo = t * chunk;
    const int hi = std::min(n, lo + chunk);
    if (lo >= hi) break;
    pool.emplace_back(run, lo, hi);
  }
  for (auto& th : pool) th.join();
}

}  // namespace racing
