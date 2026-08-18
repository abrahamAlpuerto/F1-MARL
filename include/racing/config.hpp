// Engine configuration: the car, the race, and everything that decides how the
// racing turns out.
//
// Every physical constant in VehicleParams is either a published regulation
// figure or a value fitted to real telemetry, and each one says which. Nothing
// there is "a number that made the lap time come out right" unless it is
// explicitly labelled as fitted -- see tools/calibrate_vehicle.cpp, which does
// the fitting against a real 92.608 s Bahrain lap.
//
// The racing settings below (AeroConfig, ContactConfig, RewardConfig) are the
// opposite: they are dials, and they are meant to be turned. They decide
// whether following is hard, whether a tow is worth having, and whether a car
// would rather beat its team mate or help it. Those three choices are most of
// what makes a race worth watching.

#pragma once

#include <cstdint>
#include <string>

namespace racing {

// --- vehicle ---------------------------------------------------------------
//
// A 2022-onwards ground-effect car: a dynamic bicycle model with a
// load-sensitive linear tire and a friction-ellipse clamp. Deliberately not
// Pacejka or multibody -- those cost throughput and buy nothing here, because
// what this project cares about is what happens *between* cars.
struct VehicleParams {
  // -- mass and geometry, from the FIA Technical Regulations ---------------
  double mass = 798.0;          // kg, 2024 minimum incl. driver (2025: 800)
  double wheelbase = 3.600;     // m, regulation maximum; every team runs it
  double front_weight_frac = 0.455;  // static front axle share; FIA window is
                                     // 44.6-46.1%, so this sits mid-range
  double cg_height = 0.260;     // m. Low by road-car standards, and directly
                                // sets how much load moves fore/aft under power
                                // and braking -- which is what decides whether
                                // the car stays stable when it is slowed hard
  double yaw_inertia = 1000.0;  // kg m^2, typical for a car this size and mass

  // -- aerodynamics --------------------------------------------------------
  // Lift and drag area, i.e. Cl*A and Cd*A, so force = 0.5 * rho * CxA * v^2.
  // Quoted as areas because that is how they are measured and reported; it also
  // avoids having to name a reference area that nobody agrees on.
  double cl_a = 6.2012;         // fitted; ~2.3x car weight in downforce at 250 kph
  double cd_a = 1.7500;         // fitted against the straight-line speed trace
  double aero_balance_front = 0.45;   // share of downforce carried by the front
  // kg/m^3. The density the aero coefficients were fitted at; AtmosphereConfig
  // reproduces it from its default conditions, and overrides it when they
  // change. Left here as the reference so the fit has something to be relative
  // to.
  double air_density = 1.1918;

  // -- tires ---------------------------------------------------------------
  // Peak friction coefficient at the reference load. High compared to a road
  // tire because these are slicks, and because a single coefficient here is
  // also absorbing effects a linear model does not represent separately.
  double mu_peak = 1.8175;      // fitted
  // N per WHEEL, near the static load. It was 4000 when grip was computed once
  // per axle; four-corner loads evaluate mu per contact patch, so the reference
  // has to be per contact patch too. Leaving it at 4000 evaluates mu at half
  // the load it was fitted at, and since mu falls with load that hands the car
  // free grip -- it made the quasi-steady-state lap four seconds quicker than
  // the real one it is supposed to reproduce.
  double mu_ref_load = 2000.0;
  double mu_load_sensitivity = 0.1874;  // fitted. mu falls as (Fz/ref)^-this.
                                        // Real tires lose grip as they are
                                        // pushed harder; without it, downforce
                                        // buys unlimited cornering speed

  // Axle cornering stiffness per newton of normal load, 1/rad. These put peak
  // slip near 6-7 degrees, about right for a slick.
  //
  // The front and rear values differ, and they have to. A single shared value
  // makes the understeer gradient identically zero: it is W_f/C_f - W_r/C_r,
  // and with C proportional to load both terms are 1/k and cancel exactly. The
  // car is then neutrally stable by construction, so the moment weight
  // transfers under braking the rear is the softer axle and the car diverges.
  // It did exactly that -- yaw rate ran away from 0.11 to 4.8 rad/s in about
  // 40 m, with the rear at 49 degrees of slip while the front was still at 6.
  //
  // The fix is not a fudge, it is the actual car: F1 rear tires are far wider
  // than the fronts (405 mm against 305 mm in the current regulations), so the
  // rear axle really is the stiffer one. The ratio here is that width ratio.
  double cornering_stiffness_front_per_n = 14.0;
  double cornering_stiffness_rear_per_n = 18.5;

  // Ceiling on lateral acceleration, and not merely a safety clamp -- without
  // it the model has no cornering limit at all above about 180 m radius.
  //
  // Grip and demand both scale with v^2: demand is m*v^2/R, and grip is
  // mu*(m*g + 0.5*rho*ClA*v^2). Which one wins is decided by a ratio of
  // constants, not by speed, so once 0.5*rho*ClA*mu exceeds m/R the two curves
  // never cross and the solver reports that the car could take the corner at
  // any speed whatever. It duly returned 540 kph and 11.5 g for a 200 m radius.
  //
  // Real cars stop well short of that, for reasons a linear tire and a v^2 aero
  // map do not represent: downforce flattens off at high speed as the floor
  // runs out of ride height, tire construction limits what a contact patch can
  // carry, and there is a driver. 6 g is a little above the peak seen in the
  // Bahrain telemetry (~5 g) and is deliberately not tight enough to bind in
  // ordinary corners -- at 100 m radius the tire limit still governs.
  //
  // This one matters far beyond lap time. Dirty air costs grip *in corners*
  // (see AeroConfig); if fast corners were not grip-limited, losing downforce
  // there would cost nothing at all and following would be free.
  double max_lateral_g = 6.0;

  // -- powertrain ----------------------------------------------------------
  double max_power = 640000.0;  // W (~858 hp), fitted. Below the ~735 kW quoted
                                // peak for a 2024 ICE + ERS, which is the right
                                // direction: peak deployment is not available
                                // for a whole lap
  double max_brake_force = 55000.0;  // N. An actuator ceiling only. The brake
                                     // command is a fraction of the grip
                                     // available at the time (see vehicle.cpp),
                                     // so this rarely binds and the car stays
                                     // grip-limited rather than actuator-limited
  double drivetrain_rear_frac = 1.0;  // F1 is rear-wheel drive
  double brake_bias_front = 0.58;     // Nominal front bias. Retained for
                                      // reference and for later setup work;
                                      // braking is currently split in
                                      // proportion to each axle's available
                                      // grip, which is what a correctly
                                      // balanced system achieves

  // -- steering ------------------------------------------------------------
  // Roadwheel angle at full lock, ~26 degrees.
  //
  // This started at 0.28 rad (16 deg) on the reasoning that F1 steering locks
  // are small, and that was simply wrong -- wrong in a way that quietly made
  // the circuit impossible. Turning at the limit needs the Ackermann angle plus
  // the front slip angle the tire needs to make peak force, and at Bahrain's
  // tightest corners that comes to 17.8 deg:
  //
  //     R = 18 m : 11.3 deg Ackermann + 6.5 deg slip = 17.8 deg
  //     R = 25 m :  8.2 deg           + 6.5 deg      = 14.7 deg
  //
  // So the car could not physically get round Turn 1 or Turn 8, and a policy
  // that stalled there looked like a learning problem and was not.
  double max_steer = 0.45;

  // -- rolling resistance --------------------------------------------------
  double rolling_resistance = 0.014;  // fraction of normal load

  // -- numerical guards ----------------------------------------------------
  // Below this speed the dynamic bicycle model is singular: slip angle divides
  // by longitudinal velocity, so as v -> 0 an infinitesimal lateral velocity
  // implies an enormous slip angle and the tire forces explode. Standard fix is
  // to blend to a kinematic model, which has no such term. See vehicle.cpp.
  double blend_speed_lo = 2.0;  // m/s, fully kinematic below
  double blend_speed_hi = 7.0;  // m/s, fully dynamic above

  // -- track width and roll -------------------------------------------------
  // The distance between the wheels on an axle, and how the car's roll
  // resistance is split front to rear.
  //
  // These exist because lateral load transfer is not a detail. Grip is
  // sub-linear in load, so moving weight from the inside wheel to the outside
  // one LOSES the axle grip -- two wheels at (Fz/2 +/- d) always make less
  // than two at Fz/2. That is the mechanism behind most of what a chassis
  // engineer does, and a model without it corners identically whatever the
  // setup.
  //
  // The front/rear split is the classic balance adjustment: stiffening one end
  // makes it do more of the transferring, so it loses more grip, so the car
  // pushes away from that end.
  double track_width = 1.60;         // m, regulation-ish for a 2m-wide car
  double roll_stiffness_front = 0.52;  // share of roll resisted at the front

  // -- driveline geometry ---------------------------------------------------
  double wheel_radius = 0.36;  // m, loaded radius of a 18" rear

  // -- physical extent, for contact ----------------------------------------
  // Roughly a current F1 car. Used only by the contact model and by the
  // renderer; the dynamics treat the car as a point mass with a wheelbase.
  double length = 5.60;  // m
  double width = 2.00;   // m
};

// --- the air ---------------------------------------------------------------
//
// Air density is not a constant, and it is the term every aerodynamic force is
// multiplied by. A hot afternoon at Bahrain is roughly 8% thinner than a cool
// morning, which is worth several kph of top speed and a noticeable slice of
// downforce -- enough that it is worth being able to say what conditions a race
// was run in rather than baking one number into the car.
struct AtmosphereConfig {
  // The defaults are the conditions the car was CALIBRATED at, and they are not
  // arbitrary: 22 C at 35% humidity and standard pressure works out to
  // 1.1918 kg/m^3, which is the `VehicleParams::air_density` the aero
  // coefficients were fitted against. Change them and the car is genuinely a
  // different car -- which is the point, but it means the 92.6 s reference lap
  // no longer applies.
  //
  // They are also realistic for the race the telemetry came from: Bahrain runs
  // under lights in the evening, not in the afternoon sun.
  double air_temperature_c = 22.0;
  double pressure_pa = 101325.0;
  double humidity = 0.35;  // 0 to 1. Humid air is LESS dense than dry air --
                           // water vapour is lighter than the nitrogen it
                           // displaces, which is the opposite of most people's
                           // intuition

  // Track temperature drives the tyres rather than the aero. Tarmac in the sun
  // runs far hotter than the air above it.
  double track_temperature_c = 28.0;

  // Wind, in the world frame. `direction` is where the wind is blowing TO.
  // A headwind down the main straight costs top speed and adds downforce; the
  // same wind is a tailwind on the other side of the circuit.
  double wind_speed = 0.0;      // m/s
  double wind_direction = 0.0;  // rad

  // Density from the ideal gas law with a humidity correction. Overridden by
  // VehicleParams::air_density if this is disabled.
  bool enabled = true;
};

// --- fuel ------------------------------------------------------------------
//
// A car starts a race heavy and finishes light, and the difference is large:
// 100 kg on a 798 kg car is an eighth of its mass, worth roughly three tenths
// of a second per lap per ten kilograms. It is why the last lap of a stint is
// quicker than the first even on worn tyres, and it is a real strategic axis.
//
// The calibration is run in qualifying trim -- no fuel -- which is what the
// 92.608 s reference lap was set in.
struct FuelConfig {
  bool enabled = true;
  double start_kg = 100.0;      // full race tank
  double burn_kg_per_km = 1.8;  // ~2 kg per Bahrain lap
};

// --- tyres -----------------------------------------------------------------
//
// The single biggest thing missing from a bare friction ellipse. A tyre is not
// a constant: it has a temperature it wants to be at, it takes time to get
// there, it overheats if it is worked too hard, and it wears out.
//
// All three show up as racing rather than as numbers. A cold tyre is why the
// lap after a rejoin is slow. An overheating tyre is why a car that has been
// stuck in dirty air for five laps cannot attack when it finally gets clear.
// Wear is why the order at the end of a stint is not the order at the start.
struct TyreCompound {
  const char* name;
  double grip;          // multiplier on mu_peak
  double optimum_c;     // core temperature it works best at
  double window_c;      // half-width of the band where grip is near peak
  double wear_rate;     // fraction of life per megajoule of friction work
  double heat_rate;     // K per megajoule; how quickly work heats it
};

struct TyreConfig {
  bool enabled = true;

  // 0 soft, 1 medium, 2 hard. Softer is quicker and wears faster, which is the
  // whole trade and the reason more than one exists.
  int compound = 1;

  double start_temperature_c = 80.0;  // off the blankets

  // Newton cooling coefficients, per second per kelvin above ambient. Airflow
  // does most of the work, so the coefficient rises with speed -- which is why
  // tyres cool down the straights and cook through a corner sequence.
  //
  // These set the EQUILIBRIUM temperature against the frictional heating, and
  // they are small numbers for a reason: at 0.55, a tyre 50 K above ambient
  // would shed 27 K every second and nothing could ever warm up. The whole
  // field ran the race at ambient temperature on 55% grip.
  //
  // As set, a car doing ~40 kW of frictional work at 50 m/s settles about 70 K
  // above ambient, which lands in the working window.
  //
  // Scaled together with the compounds' `heat_rate` so the EQUILIBRIUM
  // temperature stays put while the thermal inertia goes up. The first working
  // version settled in the right place but got there in a couple of seconds, so
  // grip swung between 0.55 and 1.00 within a single corner. A real tyre is a
  // lump of rubber with a thermal time constant of the better part of a minute.
  double cooling_rate = 0.0046;
  double cooling_speed_factor = 0.00016;

  // Grip at the edges of the working window, as a fraction of peak. Cold tyres
  // are worse than hot ones up to a point; a badly overheated tyre is worse
  // than either, because the surface goes greasy.
  double grip_cold = 0.88;
  double grip_hot = 0.86;

  // The front tyres get more heat per joule of work than the rears.
  //
  // Measured, not assumed: a lap of Bahrain puts 2.4 MJ through the rear axle
  // and only 1.3 MJ through the front, because the rear is the driven one. With
  // a single heat rate the rears sit in the window and the fronts never leave
  // the cold end, and a car permanently down on front grip understeers off at
  // every corner. Fronts are also narrower -- 305 mm against 405 -- so the same
  // work goes into less rubber and genuinely does raise their temperature
  // faster. This is that ratio, rounded up to close the measured gap.
  double front_heat_scale = 1.7;

  // Grip remaining at 100% wear. Real tyres do not go to zero -- they go slow
  // and unpredictable, and a driver pits long before this.
  double grip_worn = 0.72;

  // Wear also lowers the temperature the tyre wants to run at, because a thin
  // tyre has less rubber to carry the heat.
  double wear_optimum_shift_c = -8.0;
};

// --- powertrain ------------------------------------------------------------
//
// Above the level of a single "max power" figure, three things about an F1
// powertrain are visible from the outside and worth having.
//
// The torque curve and the gearbox, which together mean the car does not
// accelerate smoothly -- it surges, drops on each shift, and runs out of
// puff at the top of a gear.
//
// And the hybrid, which is the interesting one, because it is a resource. A
// car harvests energy under braking and spends it on the straights, and it can
// only spend so much per lap. Choosing WHERE to spend it is a decision, and one
// a team can coordinate: hold station behind a team mate, harvest, then both
// deploy together down the straight.
struct PowertrainConfig {
  bool enabled = true;

  // Normalised torque against normalised rpm, sampled evenly from idle to
  // redline. A modern turbo V6 is far flatter than an old naturally-aspirated
  // engine, which is what these numbers say.
  double torque_curve[7] = {0.62, 0.86, 0.97, 1.00, 0.99, 0.94, 0.84};

  double idle_rpm = 4000.0;
  double redline_rpm = 15000.0;
  double shift_time_s = 0.04;    // torque is cut for this long on every shift
  double shift_up_frac = 0.97;   // shift up at this fraction of redline
  double shift_down_frac = 0.72;

  // Eight forward gears, from a low first to a long eighth.
  //
  // The final drive is set so that top gear reaches the redline at about the
  // car's terminal speed: 15000 rpm is 1571 rad/s, a 0.36 m wheel at 85 m/s
  // turns at 236 rad/s, so the total reduction wanted in eighth is 6.66, and
  // 0.90 * 7.40 is exactly that. Getting it wrong is not subtle -- at 3.40 the
  // car sat in first gear at 170 kph pinned against the rev limiter and never
  // shifted, because every gear was too tall to leave.
  double gear_ratios[8] = {2.90, 2.25, 1.82, 1.52, 1.30, 1.13, 1.00, 0.90};
  double final_drive = 7.40;

  // -- hybrid --------------------------------------------------------------
  double ers_capacity_mj = 4.0;       // usable store
  double ers_deploy_kw = 120.0;       // the regulation figure
  double ers_harvest_kw = 120.0;      // recovered under braking
  // Deployment has to be affordable out of what braking actually recovers, or
  // the battery is flat after two laps and the hybrid stops being a decision.
  // Measured: a Bahrain lap harvests about 2.4 MJ at the 120 kW cap. Spending
  // 4 drained it to zero and left it there for the rest of the race.
  double ers_max_deploy_per_lap_mj = 2.4;
  double ers_start_charge = 0.8;      // fraction of capacity on the grid
};

// --- DRS -------------------------------------------------------------------
//
// The overtaking aid: within a second of the car ahead at a detection point,
// the follower may open its rear wing in the zone that follows, trading
// downforce it does not need on a straight for a large chunk of drag.
//
// It interacts with the wake rather than replacing it. The tow gets you close;
// DRS is what turns close into past. Turn it off (`enabled = false`) to see how
// much harder passing is without it -- which is most of the argument about
// whether it should exist.
struct DrsConfig {
  bool enabled = true;
  double detection_gap_s = 1.0;   // to the car ahead, at the detection point
  double drag_reduction = 0.22;   // fraction of drag removed when open
  double downforce_loss = 0.14;   // and the downforce it costs

  // Zones are found from the circuit rather than authored, so this works on any
  // track without a table of hand-entered arc lengths. A zone is a stretch
  // straight enough and long enough to be worth opening the wing on.
  double zone_min_length_m = 450.0;
  double zone_max_curvature = 0.0022;  // 1/m; ~450 m radius, effectively straight
  double detection_before_m = 120.0;   // detection point ahead of the zone start
};

// How the fitted values above were arrived at, and how good they are
// ------------------------------------------------------------------
// tools/calibrate_vehicle.cpp runs a quasi-steady-state lap over the real
// Bahrain curvature profile and compares it against the lap that geometry came
// from: a 92.608 s pole lap. Only the five genuinely uncertain parameters are
// fitted; mass, wheelbase, weight distribution and steering lock are regulation
// or geometry figures and are held fixed, because letting an optimiser move
// them buys lap time by inventing a car that does not exist.
//
//   lap time      92.759 s vs 92.608 s      +0.16 %
//   top speed     298.5 kph vs 301.0 kph    -0.8 %
//   min speed      67.6 kph vs  66.6 kph    +1.5 %
//   aero efficiency  3.54                   real 3.5-4.5
//   apex speeds   within ~10 kph everywhere except T4 and T6
//
// Fitting to lap time ALONE does not work, and the first attempt proved it: it
// matched 92.6 s exactly with cd_a = 2.00 and load sensitivity pinned at its
// lower bound -- a car that corners far too well at speed paired with one that
// is far too draggy, the two errors cancelling. The fit now also targets top
// speed (which identifies the power-to-drag balance and little else) and
// minimum speed (mechanical grip, where there is no downforce to help).
//
// These values were REFITTED when four-corner loads and lateral load transfer
// were added, and the refit is worth noting for what it says about the model
// rather than the numbers. Aerodynamic efficiency went from 3.26 -- below any
// real car -- to 3.54, which is inside the real 3.5-4.5 range for the first
// time. The old fit was buying lap time with drag it did not have, to pay for
// grip the model was missing. Adding the missing physics let the fit stop
// lying about the aero.

// --- simulation ------------------------------------------------------------
struct SimConfig {
  // 100 Hz integration, with the policy acting at 25 Hz via action repeat. The
  // physics needs the small step to stay stable through a contact; the policy
  // does not, and stepping it four times slower is four times the throughput.
  double physics_dt = 0.01;  // s
  int action_repeat = 4;     // -> 25 Hz policy
};

// --- track and grid --------------------------------------------------------
struct TrackConfig {
  std::string path = "data/tracks/bahrain.json";

  // Race distance. Laps is the natural unit for a race; `episode_distance`
  // overrides it, in metres, when a training episode wants less than a lap.
  int laps = 3;
  double episode_distance = -1.0;  // m; negative means `laps` full laps
  double start_s = 0.0;            // m along the reference line

  // -- grid ---------------------------------------------------------------
  // Cars line up in a staggered column behind `start_s`, alternating sides the
  // way a real grid does. Slot 0 is on pole.
  //
  // Spacing is quoted in metres but the number that matters is the TIME gap it
  // works out to, because that is what a driver has to react in. At the
  // rolling-start speed below, 30 m is about 0.6 s. The first attempt used
  // 14 m, which at 212 kph is 0.24 s -- closer than cars ever run on purpose,
  // and the field simply crashed into itself before the first corner. If you
  // raise `rolling_start_speed`, raise this with it.
  double grid_spacing = 30.0;   // m between consecutive slots, along the line
  double grid_stagger = 2.6;    // m either side of the line, alternating

  // Rolling start. A standing start is a launch problem, and a launch problem
  // is not what anyone is here to watch -- it also spends most of early
  // training on the first hundred metres. Cars are released at this fraction
  // of the speed the reference lap was doing at their slot.
  bool rolling_start = true;
  double rolling_start_speed = 0.60;

  // -- training-only randomisation ----------------------------------------
  // Put the grid somewhere random around the lap. Without this a policy only
  // ever experiences the circuit from wherever it reliably survives to: early
  // on it spins at the first corner every time, so the other fourteen corners
  // generate no experience at all. Turn it off to race -- a race that does not
  // start on the start line means nothing.
  bool randomize_start = false;
  double start_jitter_lateral = 1.2;   // m, either side of the slot
  double start_jitter_heading = 0.04;  // rad
  double start_speed_lo = 0.60;        // fraction of the reference speed
  double start_speed_hi = 0.95;
};

// --- the field -------------------------------------------------------------
struct FieldConfig {
  // Ten teams of two: a full grid, and the F1 arrangement.
  //
  // Two per team is the smallest number where a team can do something a lone
  // car cannot -- tow a team mate, or hold a rival up while the team mate
  // escapes. More cars per team makes the cooperative signal stronger and the
  // racing busier; more teams makes the field longer and the traffic denser.
  int n_teams = 10;
  int cars_per_team = 2;

  // Grid order. GRID_INTERLEAVE puts one car from each team on each row, so no
  // team starts with a free run -- which is what you want if the interesting
  // behaviour is meant to be about racing rather than about the grid.
  // GRID_BLOCKED puts a whole team together, which is the quickest way to see
  // a team work as a unit from lap one.
  enum GridOrder { GRID_INTERLEAVE = 0, GRID_BLOCKED = 1 };
  int grid_order = GRID_INTERLEAVE;

  int n_cars() const { return n_teams * cars_per_team; }
};

// --- aerodynamic interaction -----------------------------------------------
//
// The single most important block in this file for how the racing looks.
//
// A car sitting behind another one is in its wake, and the wake does two
// opposite things. It removes drag, so the follower is faster in a straight
// line -- that is a tow, and it is how overtakes happen. And it removes
// downforce, so the follower has less grip in corners -- that is dirty air,
// and it is why following closely through a corner sequence is punishing.
//
// The tension between those two is the whole game. Get the balance wrong in
// one direction and nobody can ever pass; wrong in the other and passing is
// free and track position means nothing. Both effects use the same wake shape:
//
//     w(d, y) = exp(-d / decay_length) * exp(-(y / width)^2)
//
// with `d` the along-track gap to the car ahead and `y` the lateral offset
// between the two. A follower directly behind and very close sees w -> 1; one
// alongside, or a long way back, sees w -> 0. That the wake is *narrow* is
// what makes moving offline the way out of it, and so what makes a driver pull
// out of the tow to pass rather than sit in it forever.
struct AeroConfig {
  bool enabled = true;
  double tow_max = 0.32;        // fraction of drag removed directly behind.
                                // ~0.3 is the usual quoted figure and is worth
                                // roughly 10-15 kph at the end of a straight
  double wash_max = 0.35;       // fraction of downforce lost directly behind.
                                // Deliberately larger than the tow, because in
                                // a real car it is: this is why a move has to
                                // be finished before the corner, not in it
  double decay_length = 25.0;   // m. Wake strength e-folds over this distance
  double width = 2.6;           // m. Lateral e-folding; about a car's width, so
                                // pulling fully out of line escapes it
  double range = 80.0;          // m. Cars further ahead than this are ignored,
                                // which also bounds the neighbour search
};

// --- contact ---------------------------------------------------------------
//
// Cars cannot drive through each other. The response is deliberately soft -- a
// positional push apart plus some scrubbed speed and a yaw kick -- rather than
// a rigid-body impulse. A hard response at 100 Hz between two cars that are
// both already at the limit of grip is a good way to launch one of them into
// orbit, and a race where every touch ends in a barrel roll is neither
// realistic nor useful to learn from.
struct ContactConfig {
  bool enabled = true;
  double restitution = 0.30;    // share of the closing lateral velocity that is
                                // reversed rather than absorbed
  // The next two are RATES, per second of sustained contact, not amounts per
  // touch. Contact is resolved at the physics rate, so a fixed amount per call
  // is applied a hundred times a second: at 0.05 per step, a one-second scrape
  // removed 99.4% of a car's speed and the field simply stopped. As rates,
  // 1.2 costs a heavy 0.2 s hit about a fifth of its speed, which is roughly
  // what a real one does.
  double speed_loss = 1.2;      // fraction of forward speed scrubbed per second
  double yaw_kick = 2.5;        // rad/s^2 of yaw disturbance at full severity
  double separation_gain = 0.6; // share of the overlap corrected per step.
                                // Correcting all of it at once reads as a
                                // teleport in the replay
};

// --- damage and retirement -------------------------------------------------
//
// What turns a bad moment into a DNF. Two sources, and they are deliberately
// different shapes, because the two accidents are different shapes.
//
// **Car to car is a RATE.** The obvious rule -- damage proportional to the
// severity of a contact event -- does not survive contact with the data. A
// three-lap race of twenty scripted cars produces ~288 contact events, and the
// severity reported for an event is the WORST single physics step of the touch,
// so its distribution is saturated: p90 is 1.0 and a third of all events read as
// a maximum-severity hit. Keyed on that, every car retires on lap one.
//
// The per-step distribution tells the true story -- p50 is 0.021 -- because
// almost all of those peaks are one-step transients as the solver corrects an
// overlap. So damage integrates severity over time, exactly as `speed_loss` and
// `yaw_kick` in ContactConfig already do, and for the same reason. A spike
// costs nothing; staying locked together at high severity is what breaks a car.
//
// The defaults below were fitted to a 20-car scripted race rather than chosen.
// Measured over 3 laps, with the front of the field as the control: the winner
// finishes in 385.6 s against 382.4 s with damage switched off entirely, so the
// model does not tax cars that stay out of trouble. What it does instead is
// stretch the tail -- the last car home goes from 426 s to 509 s -- because the
// cars that had accidents limp to the flag, which is what should happen.
//
// The resulting spread across the field: the median car finishes on 0.116
// damage (barely marked), the 90th percentile on 0.716 (visibly slower and
// unable to defend), and the worst at terminal. Skewed, deliberately -- damage
// should collect on the cars that had incidents rather than spread evenly over
// a field that merely raced closely.
//
// **The barrier is an IMPULSE.** A wall is a single well-defined event with a
// speed attached, so it is charged once, on the speed normal to it. This is the
// realistic path to most DNFs, and the chain runs the way it does in a real
// race: a heavy hit spins a car, the spin puts it off the circuit, and it
// arrives at the wall sideways with enough speed to end its afternoon. In the
// races above, every retirement came this way.
struct DamageConfig {
  bool enabled = true;

  // -- car to car ---------------------------------------------------------
  // Severity below `contact_threshold` is a rub and costs nothing. Rubbing is
  // most of what close racing is, and a model that charges for it teaches a
  // policy to leave a car's width everywhere, which is not racing. At 0.30 the
  // median car finished a race on 0.45 damage, which is a field of wrecks after
  // an afternoon of ordinary wheel-to-wheel running.
  double contact_threshold = 0.55;
  double contact_rate = 1.6;      // damage per severity-second above it

  // -- the barrier --------------------------------------------------------
  // How much run-off there is beyond the edge of the corridor before there is
  // something to hit.
  //
  // This MUST be larger than RaceConfig::recover_distance, and the ordering is
  // load-bearing: a car that trickles off and stops in the gravel is recovered
  // and rejoins, and only a car with enough speed to cross the whole run-off
  // finds the wall. Put the wall inside the recovery distance and a car pinned
  // against it can never satisfy the recovery test, so it sits there forever.
  double run_off_width = 30.0;    // m beyond the corridor edge

  // Severity of a barrier impact is the speed normal to the wall over this.
  // 25 m/s of closing into a wall is a very large accident.
  double impact_speed_full = 25.0;  // m/s -> severity 1
  double barrier_threshold = 0.15;  // brushing the wall costs nothing
  double barrier_per_hit = 1.20;    // damage at severity 1: more than terminal,
                                    // so a full-speed hit ends the race there
  double barrier_restitution = 0.20;  // share of normal speed given back
  double barrier_speed_loss = 0.55;   // share of FORWARD speed lost in a hit

  // -- consequences -------------------------------------------------------
  double retire_threshold = 1.0;  // damage at or above this and the car is out

  // What being damaged costs, quoted at damage = 1. A car retires at 1, so
  // these are approached and never quite reached. Aero damage rather than
  // mechanical because that is what actually falls off a car in a collision,
  // and because losing downforce makes a car slower in a way a viewer can see:
  // it holds up the cars behind it and cannot defend.
  double downforce_loss = 0.40;
  double drag_penalty = 0.20;
};

// --- reward ----------------------------------------------------------------
//
// Three terms, and they do different jobs.
//
//   progress   teaches the car to drive at all. Potential-based, so it can be
//              annealed away later without moving the optimum.
//   position   teaches it to RACE. Progress alone produces a field of cars
//              doing perfect, lonely time trials past each other; paying for
//              the position itself is what makes a car care that someone else
//              is there.
//   team       decides whose result it cares about. See `team_weight` -- this
//              is the dial that decides whether the field behaves like teams.
struct RewardConfig {
  double gamma = 0.997;           // the learner's discount, quoted here so the
                                  // training loop and the shaping agree

  // -- progress -----------------------------------------------------------
  double progress_weight = 0.05;  // per metre

  // The gamma used *inside the shaping term*, which is deliberately not the
  // discount above. Expand F = g*Phi(s') - Phi(s) with Phi = w * progress and a
  // discount of 0.99:
  //
  //     F = w * (0.99 * (p + d) - p) = w * (0.99 d - 0.01 p)
  //
  // The first term is the progress just made, ~2.8 m at racing speed. The
  // second is a drag proportional to total distance covered. They cross at
  // about 280 m, so from a quarter of the way around the first lap onward,
  // driving forward earns *negative* reward and the best available policy is to
  // stop. With shaping_gamma = 1 the term telescopes to w * d, the standard
  // progress reward, and over a completed episode it sums to a constant -- so
  // it cannot change which policy is optimal among those that finish.
  double shaping_gamma = 1.0;

  // -- position -----------------------------------------------------------
  // Paid on the *change* in race position, so gaining a place is worth
  // +position_weight and losing one costs the same. Paying on the level
  // instead would be a constant offset for a car that never passes anyone, and
  // would tell it nothing about what it did this step.
  double position_weight = 1.0;

  // Bonus on the final classification, applied once at the end as
  // (n_cars - position) * this. Large enough that the last lap still matters.
  double finish_weight = 2.0;

  // -- team ---------------------------------------------------------------
  // How much a car cares about its team mates' results rather than its own.
  //
  //   0.0  every car for itself. Team mates race each other as hard as rivals
  //        and will happily take each other out.
  //   0.5  balanced, and where the interesting behaviour lives: a car will give
  //        a place to a faster team mate, sit in front of a rival to hold it
  //        up, or run nose-to-tail with a team mate down a straight so both get
  //        a tow -- all of which pay it back through the team term while
  //        costing it something individually.
  //   1.0  fully cooperative. The team is one agent with several bodies, and
  //        individual position stops mattering, which makes for tidy but
  //        oddly selfless racing.
  //
  // The reward a car actually receives is
  //     (1 - team_weight) * own + team_weight * (mean over its team)
  double team_weight = 0.5;

  // -- penalties ----------------------------------------------------------
  double off_track_penalty = 5.0;   // per step spent outside the corridor
  double contact_penalty = 2.0;     // per contact, scaled by severity
  double time_penalty = 0.0;        // per policy step; 0 while shaping carries it

  // Paid once, on the step a car retires from an ACCIDENT -- a collision or the
  // barrier. A DNF already costs a car everything it would have earned for the
  // rest of the race, which is most of the signal; this is on top so that
  // ending your own race is clearly worse than finishing last rather than
  // merely equal to it.
  //
  // Deliberately NOT charged when the reason is RETIRE_OFF_TRACK, which only
  // happens under `terminate_off_track` -- a training device rather than an
  // accident. Charging it there makes standing still the best available policy
  // and training collapses; the measurement and the reasoning are in race.cpp
  // next to the exception.
  double retire_penalty = 20.0;

  // -- track limits -------------------------------------------------------
  double off_track_margin = 0.5;    // m beyond the corridor edge before it counts

  // Grip available off the racing surface, as a fraction of nominal. A car that
  // runs wide loses time rather than teleporting back, which is both what
  // happens and what keeps the race watchable.
  double off_track_grip = 0.55;

  // End the episode for a car that leaves the track rather than letting it
  // rejoin. Right for the early phase of training -- it stops a policy learning
  // to cut corners, and stops it wasting samples driving through the desert --
  // and wrong for a race, where one car's mistake should not end everyone's
  // afternoon. See RaceConfig::recover_after_s for what happens instead.
  bool terminate_off_track = false;
};

// --- the race --------------------------------------------------------------
struct RaceConfig {
  // A car stranded well off the circuit is put back on it, facing the right
  // way, at a low speed, after this long. Without it a race can end with three
  // cars still running and five parked in the sand, which nobody enjoys
  // watching and which stops generating useful experience.
  double recover_after_s = 3.0;
  double recover_speed = 25.0;      // m/s it rejoins at
  double recover_distance = 25.0;   // m beyond the corridor that counts as
                                    // stranded rather than merely wide

  // How many other cars a policy sees, nearest first by along-track gap. Four
  // covers the car in front, the car behind, and one either side in a fight;
  // it is also small enough to keep the observation cheap.
  int n_neighbours = 4;
};

struct EnvConfig {
  VehicleParams vehicle;
  SimConfig sim;
  TrackConfig track;
  FieldConfig field;
  AeroConfig aero;
  ContactConfig contact;
  DamageConfig damage;
  RewardConfig reward;
  RaceConfig race;
  AtmosphereConfig atmosphere;
  FuelConfig fuel;
  TyreConfig tyre;
  PowertrainConfig powertrain;
  DrsConfig drs;

  uint64_t seed = 0;

  // How far ahead the policy can see, and the single most consequential
  // observation choice here. 20 points at 15 m reaches 300 m down the road.
  //
  // The first attempt used 10 points at 10 m, which is 100 m -- barely more
  // than the ~75 m the car needs just to brake from 300 to 100 kph, and less
  // than the distance covered while deciding to. A policy that cannot see a
  // corner until it is already too late to slow for it can only learn to drive
  // slowly everywhere, which is a much worse local optimum than it sounds,
  // because slow driving is stable and the gradient out of it is weak.
  int curvature_lookahead = 20;
  double lookahead_spacing = 15.0;  // m between those points

  int n_cars() const { return field.n_cars(); }

  // Load an EnvConfig from JSON, so a sweep can vary a parameter without a
  // rebuild. Unknown keys are an error rather than a silent no-op: a typo in a
  // swept parameter name would otherwise quietly run the wrong race.
  static EnvConfig from_json_file(const std::string& path);
  static EnvConfig from_json_string(const std::string& text);
  std::string to_json_string() const;
};

}  // namespace racing
