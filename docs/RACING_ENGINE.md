# How the engine works

The parts that are not obvious from the code, and the mistakes that are worth
not repeating. For the format the visualizer consumes, see
[VISUALIZER_FEED.md](VISUALIZER_FEED.md).

---

## Shape of it

One `RaceEnv` is one race: N cars, T teams, one circuit. `VecRaceEnv` runs many
of them side by side, which is the only arrangement where the throughput works —
a policy step costs far less than a Python round trip, so what matters is how
many races advance per call rather than how fast one of them runs.

Every car is an agent. They share an action space — `(steer, throttle)`, both
−1…1, at 25 Hz — and in the reference training loop they share a policy. What
separates them is what they can see and whose result they are paid for.

The physics runs at 100 Hz under an action repeat of 4. Contact needs the small
step to stay stable; the policy does not, and stepping it four times slower is
four times the throughput.

```
RaceEnv::step
  ├─ for each of 4 physics steps:
  │    apply_wake        every car's downforce and drag, from everyone else
  │    per car:
  │      gather          wake + DRS, air density, wind, gradient, fuel mass,
  │                      tyre grip, and what the powertrain can give
  │      vehicle_.step   integrated independently (RK4), telemetry out free
  │      project         world -> Frenet, and race distance
  │      update_tyre     heat and wear, from the frictional work just done
  │      powertrain      gearbox, and the battery: harvest or deploy
  │      burn fuel, update DRS arming
  │    resolve_contacts  push apart anything overlapping
  │    classify          positions and gaps
  │    observer          -> the feed samples a frame if it is due
  ├─ track limits, recovery, laps, the flag
  ├─ events: overtakes, contacts, laps, finishes
  └─ reward, then team mixing
```

The vehicle knows nothing about other cars, the weather, its own fuel load or
the state of its tyres — and must not, or two cars could not be stepped
independently. All of it reaches the physics through one struct,
`VehicleInput`, whose every field defaults to "nothing": a bare one is the car
the calibration was fitted against, in still sea-level air on fresh tyres with
an empty tank on a flat road.

---

## The observation

51 floats per car:

| what | size | why |
|---|---|---|
| own dynamics | 4 | vx, vy, yaw rate, slip |
| track relative | 2 | lateral offset over half width, heading error |
| curvature ahead | 20 | 20 points at 15 m — 300 m of road |
| race context | 3 | distance remaining, position, lap fraction |
| aero state | 2 | downforce and drag factor |
| neighbours | 20 | nearest 4 cars: gap, lateral offset, closing speed, **is team mate**, present |

Two of these are load-bearing.

**The curvature lookahead is the single most consequential choice here.** The
first attempt used 10 points at 10 m, which is 100 m — barely more than the
~75 m the car needs just to brake from 300 to 100 kph, and less than the
distance covered while deciding to. A policy that cannot see a corner until it
is too late to slow for it can only learn to drive slowly everywhere, which is a
much worse local optimum than it sounds: slow driving is stable, and the
gradient out of it is weak.

**The team-mate flag is what makes a shared policy able to behave like a team.**
One network drives every car, so the only thing that can make it treat a team
mate differently is an input that says which is which.

What is deliberately *not* in there: any absolute position. A policy may know
where a rival is relative to itself, never where either of them is on the
planet, or it learns the circuit by coordinate rather than by what it can see.
`VecRaceEnv::states()` exposes the privileged view for logging and for the feed;
it is never an observation.

---

## The reward

```
own_i  =  progress_weight * metres gained          (learn to drive)
        + position_weight * places gained          (learn to race)
        - off_track_penalty  - contact_penalty
        + finish_weight * (n_cars - final position)

r_i    =  (1 - team_weight) * own_i  +  team_weight * mean(own over team)
```

The position term is paid on the **change**, so a place gained is worth exactly
what a place lost costs. Paying on the level instead is a constant offset for a
car that never passes anybody, and tells it nothing about what it just did.

The team term is the whole cooperative mechanism. At `team_weight` 0.5, a car
that gives up a place to let a team mate through loses one unit of position
reward and gains half of what the team mate gained — so the move is worth making
exactly when the team mate had more to gain than it did. Nothing anywhere
encodes "let your team mate past"; that is a consequence.

### Why shaping_gamma is 1 and not the discount

Potential-based shaping is `F = γΦ(s') − Φ(s)`. Expand it with `Φ = w·progress`
and a discount of 0.99:

```
F = w * (0.99 * (p + d) - p) = w * (0.99 d - 0.01 p)
```

The first term is the progress just made, ~2.8 m at racing speed. The second is
a drag proportional to total distance covered. They cross at about 280 m, so
from a quarter of the way round the first lap onward, driving forward earns
*negative* reward and the best available policy is to stop. That is not a
subtlety of shaping theory — the theory holds for any potential — it is a scale
error: an unbounded potential paired with a discount below 1 puts the useful
signal underneath the drift.

With `shaping_gamma = 1` the term telescopes to `w·d`, the standard progress
reward, and over a completed episode it sums to a constant. It cannot change
which policy is optimal among those that finish.

---

## What the car actually models

The full list, because "it has physics" is not a useful claim. Everything here
is in the loop every step, and every one of them changes how a race turns out.

### Chassis and tyres

- **Dynamic bicycle model**, RK4 at 100 Hz, with a kinematic blend below 7 m/s
  where the slip-angle terms go singular.
- **Load-sensitive tyres.** Grip goes as `(Fz/Fref)^-0.187`, so a harder-loaded
  tyre gives less grip per newton. Everything below depends on this being true.
- **Longitudinal load transfer** under power and braking. This is what makes
  trail-braking work.
- **Lateral load transfer onto four corner loads**, split by roll stiffness.
  Because grip is sub-linear in load, two wheels at `Fz/2 ± d` always make less
  than two at `Fz/2` — so cornering hard costs grip, and the front/rear split of
  that loss is the car's balance.
- **Friction ellipse per axle**, with a **falling tail past the peak**: asking
  for more force than the tyre has gives you *less*. That is what wheelspin and
  a locked wheel are, and the telemetry reports both.
- **Tyre temperature**, per axle, driven by the frictional work the contact
  patches are doing and cooled by airflow. Grip peaks in a window and falls off
  cold or hot.
- **Tyre wear**, from the same frictional work, accelerated when the tyre is
  over its window. Grip falls with the rubber left, and a worn tyre wants to run
  cooler.
- **Three compounds**, trading grip against life.

### Aerodynamics

- **Downforce and drag** from `½ρCA v²`, with the aero balance split front/rear.
- **Air density** from temperature, pressure and humidity, so the same car is a
  different car on a cold night than a hot afternoon.
- **Wind**, resolved per car per step into a headwind component — so the same
  wind helps on one straight and hurts on the other.
- **The wake of the car ahead**: less drag (a tow), less downforce (dirty air).
- **DRS**, in zones found from the circuit's own curvature, armed at a detection
  point only if the car was within a second of the one ahead.

### Powertrain

- **Torque curve** scaled so peak *power* matches the calibrated figure.
- **Eight-speed gearbox** with a torque cut on each shift, so a hairpin exit is
  traction-limited and the end of a straight is power-limited.
- **Hybrid**: a battery that harvests under braking, deploys on throttle, and is
  capped both by a power limit and by a per-lap energy allowance.

### The rest of the world

- **Fuel**, carried as mass and burning off — a full car is genuinely harder to
  stop and turn, not merely slower.
- **Road gradient** from the circuit's elevation profile.
- **Surface**: reduced grip off the racing surface, and a recovery for a car
  that gets stranded.
- **Contact** between cars, resolved on the axis of least penetration.
- **Barriers** at the far edge of the run-off, and **damage** that accumulates
  from hitting them or from hitting other cars. Past a threshold a car retires.

### Damage and the DNF

Two sources, and they are deliberately different shapes because the two
accidents are different shapes.

**Car-to-car damage is a rate**, integrated over the time two cars spend in
contact above a severity threshold. The obvious alternative — damage per contact
*event*, scaled by that event's severity — does not survive contact with the
data. A 3-lap race of twenty scripted cars produces around 288 contact events,
and the severity attached to an event is the worst single physics step of the
whole touch. That distribution is saturated: its 90th percentile is 1.0, and a
third of all events read as a maximum-severity hit. Charge damage against it and
the entire field retires on lap one. The per-*step* distribution tells the true
story — a median of 0.021 — because nearly all of those peaks are one-step
transients while the solver separates two overlapping cars. So the model
integrates, exactly as `speed_loss` and `yaw_kick` already do, and for the same
reason (see *Contact effects are rates, not amounts* below). This is the third
time that lesson has had to be learned in this file.

**Barrier damage is an impulse.** A wall is a single well-defined event with a
speed attached, so it is charged once, on the component of velocity normal to
it. Only impacts count: the collision reverses the normal velocity, so a car
resting against a barrier has nothing left to give and is not billed again, and
a car sliding *along* one has almost no normal component, which is correct —
that is a scrape.

The run-off is 30 m wide, and that number is constrained from below: it must
exceed `RaceConfig::recover_distance` (25 m), or a car pinned against the wall
could never get far enough off the circuit to satisfy the recovery test and
would sit there for the rest of the race. The ordering is asserted in the tests.

The consequence of the split is that the realistic causal chain works: a heavy
hit spins a car, the spin puts it off the circuit, and it arrives at the wall
sideways with enough speed to end its afternoon. In the calibration races every
single retirement came that way. Contact alone retiring a car is possible but
rare, which is also how it looks on a Sunday.

Partial damage is not just a counter ticking toward a DNF — it costs downforce
and adds drag, so a damaged car is slower in the corners and cannot defend. That
is deliberately visible: a viewer should be able to see that a car has been in
the wars without reading a number.

The defaults were fitted against the scripted field rather than chosen. The
winner's time is 385.6 s against 382.4 s with damage switched off entirely, so
the model does not tax cars that stay out of trouble; what it does is stretch
the tail, with the last car home going from 426 s to 509 s. Across the field the
median car finishes on 0.116 damage, the 90th percentile on 0.716, and the worst
at terminal — skewed on purpose, so damage collects on the cars that had
incidents rather than spreading evenly over a field that merely raced closely.

### What is deliberately not modelled

An honest boundary is more useful than a long list. None of these are missing by
oversight; each was considered and left out for a reason.

| | why not |
|---|---|
| Suspension travel, ride height, aero platform | The bicycle model has no vertical degree of freedom. Adding one means springs, dampers, and a different simulator — and the aero-versus-ride-height coupling it would buy is second-order next to the wake |
| Full Pacejka tyres, camber, pressure | A linear tyre with a load-sensitive peak and a falling tail already produces the behaviour that matters. Pacejka costs throughput and needs coefficients nobody publishes |
| Wheel rotational dynamics | Genuinely stiff: at 100 Hz a locked wheel integrates to nonsense, and it would need ~1 kHz or an implicit solver. The *effect* — wheelspin, lockup, the force falling past the peak — is modelled directly instead |
| Differential, driveshaft compliance | A bicycle model has one rear wheel |
| Brake temperature and fade | Real, but it would mostly duplicate what the tyre thermal model already expresses |
| Rain, track evolution, marbles | A weather model existed in the previous engine and was removed with it. Worth adding back if wet races are wanted; it is a day of work, not a rewrite |
| Pit stops and tyre changes | The consumables all exist, so a stop is a short state change. It is deliberately absent because it is a *strategy* layer, and this is a racing engine |
| Mechanical failures | Damage is caused, never rolled for. A random engine blow-up would need the RNG inside the physics step, and the determinism guarantee is worth more than the realism would be |
| Component-level damage | One scalar, not a front wing and a floor and a radiator. The extra fidelity would change which corner a damaged car is slow in; it would not change whether the car is slow |
| Safety cars, red flags, marshals | A retired car stops where it stopped and the race carries on around it. Neutralisation is a race-control layer, and like pit strategy it is a different problem from racecraft |
| ERS deployment as a decision | Currently automatic. Making it a third agent action — "when do I spend my battery" — is the obvious next step and changes the action space everywhere, so it was not taken quietly |

### Where the numbers come from

Five parameters are fitted — `cl_a`, `cd_a`, `mu_peak`, `mu_load_sensitivity`,
`max_power` — against a real 92.608 s Bahrain pole lap, targeting lap time, top
speed and minimum speed together. Everything else is a regulation figure, a
geometry figure, or measured from the simulator itself.

Two of them were measured rather than guessed, and it is worth saying which:

- **Tyre heating.** A lap puts about 2.4 MJ of frictional work through the rear
  axle and 1.3 MJ through the front. The heat and cooling rates are set from
  that measurement so the tyres settle in their window at racing pace. The first
  guesses were out by a factor of five in one direction and then by two in the
  other, and both produced a field that could not race.
- **Hybrid deployment.** A lap harvests about 2.4 MJ at the 120 kW recovery cap,
  so the per-lap deployment allowance is 2.4 MJ. Setting it to the headline 4 MJ
  drained the battery in two laps and left it flat for the rest of the race.

Re-run the calibration with `./build/racing_calibrate --fit`. When four-corner
loads were added it was re-run, and the result is worth knowing: aerodynamic
efficiency moved from 3.26 — below any real car — to 3.54, inside the real
3.5–4.5 range for the first time. The old fit had been buying lap time with drag
it did not have, to pay for grip the model was missing.

---

## Things that cost real time to find

Each of these changes what the engine does, and the reasoning is not recoverable
from the diff.

### 1. Load transfer must be driven by force, not by dvx/dt

The body-frame equations carry rotating-frame terms:

```
dvx/dt = fx/m + vy*r
dvy/dt = fy/m - vx*r
```

Load transfer is caused by force. Feeding `dvx/dt` into it instead meant that as
soon as the car developed sideslip, the `vy*r` term read as enormous braking —
−99 m/s² of a measured −98.6 at 45° of slip — which transferred every newton off
the rear axle until its normal load reached literally zero. The rear then had no
grip, which produced more sideslip, which fed back. The car span from any
moderate steering input above about 180 kph. It presented as an unlearnable
environment.

### 2. A shared cornering stiffness makes the car neutrally stable by construction

The understeer gradient is `W_f/C_f − W_r/C_r`. With `C` proportional to normal
load, both terms are `1/k` and cancel *exactly*, for any weight distribution. The
car is then marginally stable at best, and the moment weight transfers under
braking the rear becomes the softer axle and it diverges.

The fix is the actual car rather than a fudge: F1 rear tyres are far wider than
the fronts — 405 mm against 305 mm — so the rear axle genuinely is stiffer.

Found the same way: steering lock started at 16°, and Bahrain's tightest corners
need 17.8° (11.3° Ackermann plus 6.5° of front slip to make peak force). The car
could not physically get round Turn 1. It is now 26°.

### 3. Contact effects are rates, not amounts

Contact is resolved at the physics rate, so anything applied per call is applied
a hundred times a second. A speed scrub of 0.05 "per contact" removed 99.4% of a
car's speed over a one-second scrape, and a yaw kick of 0.35 rad/s became
35 rad/s². The whole field ground to a halt on the exit of Turn 1 and sat there.

`speed_loss` and `yaw_kick` are now per-second rates multiplied by `dt`. The
positional push-apart stays per-step, because that one is a constraint solver
and converges.

Related: the contact normal is the axis of *least penetration*, the standard
separating-axis choice. Two cars side by side are pushed apart sideways; one
that has run into the back of another is pushed apart along the road. Picking
the wrong axis fires both cars off the circuit sideways.

### 4. Race position is distance past the line, not distance covered

Cars start staggered down a grid, so they do not all begin at the same point.
Sorting the classification by metres-covered puts two cars level while they are
still a straight apart on the road, and the leaderboard flickers between cars
that cannot see each other.

`CarState::distance` is therefore measured from the start/finish line and
**starts negative** for anyone not on pole. Everyone then finishes at the same
place having covered the same distance past the line.

### 5. Every position swap is not an overtake

Two cars running nose to tail cross each other's distance dozens of times a lap
as they breathe in and out of each other's wake. Reporting each one gave 152
"overtakes" in a single lap of eight cars — a leaderboard that reads like a slot
machine. The order of each pair now needs a 1.5 m margin to flip, and the same
margin the other way to flip back. Contact events are debounced the same way:
one per touch, not one per physics step.

### 6. A wake is the strongest one, not the sum

Summing wakes lets a queue of cars stack up an arbitrarily large effect — five
cars nose to tail and the one at the back has no downforce at all. A car takes
the strongest single wake it is sitting in.

### 7. The scripted driver has to budget grip against what the car is doing

`examples/drivers.py` is not the point of the project, but it found a real
distinction. A driver that budgets its throttle against the *reference line's*
curvature believes it has all its grip available on a straight — which is true
right up until the back steps out. Then the car is turning hard, the line still
says straight, the driver keeps full power on, and it spins. Budgeting against
the car's actual path curvature (yaw rate over speed) makes the budget close
itself as the slide develops.

The same file has the other half: opposite lock alone does not catch a power-on
slide, because the thing sustaining it is the throttle.

### 8. Corridor width and racing-line width are different questions

The reference line is a *racing line*, so in a hairpin it is already at the
geometric limit; running three metres off it there means a corner the car cannot
physically take. But shrinking the available room in corners also shrank the
room to get out of another car's way, so two queuing cars at Turn 1 could not
separate by even one car width and ground against each other all the way
through. Choosing a wide line is a luxury; not hitting anyone is not.

---

## Determinism

Every random draw is a pure function of `(seed, env, stream, step, car, sub)`.
There is no generator object, no `next()`, and no mutable state in `rng.hpp`.

Two things depend on it. A race replays bit-exactly under any thread count,
which matters because the visualizer feed is written by a separate pass over the
same seed — if the engine drifted, the replay a viewer watches would not be the
race that was scored. And comparing two policies on "the same race" requires
both runs to draw identical luck, or the difference in finishing order is
confounded by noise.

This cannot be retrofitted, which is why it is tested from the first commit and
runs in CI on every push.

---

## Known limitations

**Turns 4 and 6 read tighter than they are.** The source position channel is
locally bunched there, so Turn 6 comes out at 49 m radius when the real corner is
far more open, and the model is ~29 kph slow through it. This is also why `cd_a`
sits against its upper bound and aero efficiency comes out at 3.26 rather than a
real car's 3.5+: the fit is adding drag to pay for time lost at a corner that is
not really that tight. Repairing those samples — they are detectable, because the
local chord length disagrees sharply with the trusted distance step — is the
single highest-value improvement to the circuit.

**The corridor is authored.** One lap of telemetry cannot recover the track
boundaries, so the drivable corridor is ±7.5 m around the line that was driven,
not the actual kerbs.

**The wake is a two-parameter exponential, not a flow field.** It has the right
shape and the right sign, and its width is what makes moving offline the way out
of it. It is not CFD and does not pretend to be.

**The scripted drivers top out around 86% of the limit.** Above that, pure
pursuit cuts tightening corners and a proportional speed loop arrives at corner
entry over-rotated. That is a statement about the controller, not the car.

**Blame for contact is not assigned.** The engine genuinely cannot tell whose
fault a touch was, so both cars are penalised. A wrong answer would teach a
policy that some contact is free.
