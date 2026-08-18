"""Scripted drivers with some racecraft.

These are not the point of the project -- the point is the learned policies in
train_marl.py -- but they earn their place three times over:

  1. They prove the environment is raceable. If a competent hand-written
     controller cannot get eight cars round without the race falling apart,
     then a failing training run says nothing about the learning algorithm.
  2. They give the visualizer real content on day one, before any policy has
     finished training. A renderer needs something to render.
  3. They are the baseline a learned policy has to beat, and a sparring partner
     it can be trained against.

The control is deliberately plain: pure pursuit for steering, a target speed
taken from the quasi-steady-state profile, and a small set of racecraft rules
on top that decide *which line* to aim at. Everything interesting a driver does
here comes out of that last part, because in a race the hard question is not
how fast to go, it is where to be.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
import racing  # noqa: E402


class RacecraftDriver:
    """One driver. Give each car its own, with its own seed.

    `pace` scales the target speed, so a field of these does not all lap at
    exactly the same speed and there is something to race about. `aggression`
    decides how early it commits to a move and how hard it defends.
    """

    def __init__(self, track, vehicle, index, team, *, pace=0.86,
                 aggression=0.5, seed=0):
        self.track = track
        self.vehicle = vehicle
        self.index = index
        self.team = team
        self.pace = pace
        self.aggression = float(np.clip(aggression, 0.0, 1.0))
        self.max_steer = vehicle.params.max_steer
        self.mass = vehicle.params.mass  # dry, so fuel can be weighed against it

        self.qss = np.asarray(racing.solve_qss(track, vehicle).speed)
        self.L = track.length
        self.ds = track.ds
        self.n = track.n
        self._pts = track.xy()

        # A small standing offset from the reference line, fixed per driver, so
        # the field does not drive nose-to-tail down one identical groove. Real
        # cars do not share a line to the centimetre and a replay where they do
        # looks like a train set.
        rng = np.random.default_rng(seed + 7919 * index)
        self.line_bias = float(rng.uniform(-1.4, 1.4))

        # Which way this driver prefers to go when it decides to pass. Sticking
        # to one side per driver is more readable to a viewer than picking
        # afresh each step, and stops the car dithering across the road.
        self.pass_side = 1.0 if rng.random() < 0.5 else -1.0

        self._target_ey = self.line_bias
        self._intent = "cruise"

    # --- speed ---------------------------------------------------------------

    def condition(self, me):
        """How much of the reference car this car currently is, 0 to ~1.

        The quasi-steady-state profile the target speed comes from was solved
        for a fresh car on optimum tyres with an empty tank. A real one in a
        race is none of those, and a driver that does not notice will keep
        asking for reference pace on cold tyres and put it in the gravel --
        which is exactly what the field did until this existed.

        Three terms, and cornering speed goes as the square root of each:
        the air (dirty air costs downforce), the tyres (cold, hot or worn all
        cost grip), and the fuel (a heavy car needs more force for the same
        corner).
        """
        tyre = 0.5 * (me.tyre_front.grip + me.tyre_rear.grip)
        mass_ratio = self.mass / (self.mass + me.fuel_kg)
        return max(me.downforce_factor, 0.25) * max(tyre, 0.25) * mass_ratio

    def target_speed(self, s, grip=1.0):
        """Slowest thing coming up that we still have room to slow down for.

        Taking the profile value at the current point alone is not enough: the
        car would arrive at a corner still doing straight-line speed. This walks
        forward and asks, for each point ahead, how fast we can be *here* and
        still make it there.

        `grip` is the combined condition from `condition()`.
        """
        scale = self.pace * math.sqrt(max(grip, 0.25))
        best = self.qss[int(s / self.ds) % self.n] * scale
        a = 3.2 * 9.81
        d = 0.0
        while d < 400.0:
            d += self.ds * 2.0
            j = int((s + d) / self.ds) % self.n
            v_there = self.qss[j] * scale
            best = min(best, math.sqrt(v_there * v_there + 2.0 * a * d))
        return best

    # --- racecraft -----------------------------------------------------------

    def choose_line(self, me, cars):
        """Decide where on the road to be. This is where the racing lives.

        Returns a target lateral offset in metres, positive to the left of the
        reference line.
        """
        s = me.frenet.s
        v = me.speed

        # Two different widths, for two different questions.
        #
        # `safe_w` is the road: everything inside it is somewhere a car can be.
        # `race_w` is how far off the reference line this driver will CHOOSE to
        # run, and it shrinks in corners. The reference line is a racing line,
        # so in a hairpin it is already at the geometric limit -- running three
        # metres off it there means a corner the car cannot physically take, and
        # it understeers into the run-off. On a straight, three metres is
        # nothing.
        #
        # Keeping them separate matters. Scaling everything by curvature, which
        # is what this did first, also shrank the room available for getting out
        # of another car's way: at Turn 1 the limit came out at 1.3 m, so two
        # queuing cars could not separate by even one car width and simply
        # ground against each other all the way through the corner. Choosing a
        # wide line is a luxury; not hitting anyone is not.
        kappa = abs(self.track.kappa_at(s))
        safe_w = self.track.half_width_at(s) - 1.2
        half_w = safe_w / (1.0 + 80.0 * kappa)

        ahead, ahead_gap = None, 1e9
        behind, behind_gap = None, 1e9
        for other in cars:
            if other.index == self.index or other.retired:
                continue
            gap = self.track.delta_s(other.frenet.s, s)
            if 0.0 < gap < ahead_gap:
                ahead, ahead_gap = other, gap
            elif -60.0 < gap <= 0.0 and -gap < behind_gap:
                behind, behind_gap = other, -gap

        target = self.line_bias
        self._intent = "cruise"

        # --- is there something to pass? -------------------------------------
        if ahead is not None and ahead_gap < 55.0:
            closing = v - ahead.speed
            teammate = ahead.team == self.team

            # Committing to a move needs an actual speed advantage, not merely
            # proximity. The first version attacked whenever there was a car
            # within 22 m *or* the road ahead was straight, which on a rolling
            # start is every car from the first step: the whole field pulled out
            # of line at once and drove into itself before Turn 1. Being close
            # to someone is not a reason to pass them.
            if ahead_gap < 20.0 and closing > 1.0:
                # Pull out of the wake to the side with room, which both clears
                # the dirty air and puts the car where it needs to be to finish
                # the move.
                side = self.pass_side
                if abs(ahead.frenet.e_y + side * 3.0) > half_w:
                    side = -side
                # Team mates get a wider berth. Taking each other off is the one
                # result that helps nobody.
                offset = 3.4 if teammate else 3.0
                target = float(np.clip(ahead.frenet.e_y + side * offset,
                                       -half_w, half_w))
                self._intent = "attack_teammate" if teammate else "attack"
            elif ahead_gap < 45.0:
                # Too far back to attack, close enough to tow. Sit behind and
                # let the reduced drag pull us onto its gearbox -- but slightly
                # offset, because directly behind leaves nowhere to go when the
                # car in front brakes.
                target = float(np.clip(
                    ahead.frenet.e_y + self.pass_side * 0.8, -half_w, half_w))
                self._intent = "tow"

        # --- is something about to pass us? ----------------------------------
        # Only defend where there is room to. Weaving across a braking zone is
        # how a driver takes two cars out, and it is not what this is for.
        if behind is not None and behind_gap < 18.0 and self._straight_ahead(s) > 0.4:
            attacker = behind
            if attacker.team != self.team and attacker.speed > v - 1.0:
                # Cover the line it is trying to take. `aggression` decides how
                # far across the road this driver is willing to go for it.
                cover = float(np.clip(attacker.frenet.e_y, -half_w, half_w))
                weight = 0.35 + 0.5 * self.aggression
                if self._intent == "cruise":
                    target = (1.0 - weight) * target + weight * cover
                    self._intent = "defend"

        # --- do not drive into anyone ----------------------------------------
        # Only cars actually ALONGSIDE get a lateral push. The window is one car
        # length, not several.
        #
        # This was the single worst bug in the scripted field. With a 12 m
        # window, a car simply queuing behind another counted as needing to be
        # avoided, so the whole train swerved sideways in the braking zone for
        # Turn 1 -- which is exactly where they should all be braking in a
        # straight line. The field arrived at the first corner in a pack, each
        # car steering away from the one in front of it, and stopped. Twenty
        # contacts, eight cars at walking pace, and none of it visible from the
        # rule itself.
        #
        # The response to someone in front of you is to lift, which the speed
        # controller below does. The response to someone next to you is to move
        # over. They are different problems.
        for other in cars:
            if other.index == self.index or other.retired:
                continue
            gap = self.track.delta_s(other.frenet.s, s)
            if abs(gap) > 6.0:  # roughly a car length: genuinely side by side
                continue
            sep = me.frenet.e_y - other.frenet.e_y
            if abs(sep) < 2.8:
                push = math.copysign(2.8 - abs(sep), sep if sep != 0.0 else 1.0)
                # Against the full width of the road, not the racing-line
                # budget: this is the one case where using all the tarmac is
                # worth whatever it costs in lap time.
                target = float(np.clip(me.frenet.e_y + push, -safe_w, safe_w))
                self._intent = "avoid"

        # Move toward the target rather than snapping to it, and no faster than
        # a car can actually change lane. This is per policy step at 25 Hz, so
        # 0.12 m is 3 m/s of lateral movement -- about a second to cross the
        # road. The first version allowed 0.35 m, nearly 9 m/s, and the replay
        # showed cars darting sideways like air-hockey pucks.
        self._target_ey += float(np.clip(target - self._target_ey, -0.12, 0.12))
        return self._target_ey

    def _straight_ahead(self, s):
        """How straight the next 200 m is, 0 to 1."""
        total = 0.0
        for i in range(20):
            total += abs(self.track.kappa_at(s + i * 10.0))
        return float(np.clip(1.0 - total / 0.20, 0.0, 1.0))

    # --- the actual controls -------------------------------------------------

    def act(self, me, cars):
        st = me.state
        v = st.speed
        s = me.frenet.s
        target_ey = self.choose_line(me, cars)

        # --- steering: pure pursuit toward a point on the chosen line --------
        # A short lookahead, because Bahrain's slow corners tighten quickly and
        # pure pursuit aimed far ahead simply cuts across them.
        lookahead = float(np.clip(0.45 * v, 6.0, 35.0))
        tx, ty = self.track.to_world((s + lookahead) % self.L, target_ey)
        alpha = math.atan2(ty - st.y, tx - st.x) - st.psi
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        delta = math.atan2(2.0 * 3.6 * math.sin(alpha), lookahead)

        # Yaw damping. Pure pursuit only aims the car; it has no idea whether
        # the car is already rotating faster than the corner needs, so it cannot
        # catch a slide. The corner needs a yaw rate of v * kappa; anything
        # beyond that is the rear coming round, and gets opposite lock in
        # proportion.
        k = self.track.kappa_at(s)
        delta -= 0.16 * (st.r - v * k)
        steer = float(np.clip(delta / self.max_steer, -1.0, 1.0))

        # --- speed, respecting the friction ellipse --------------------------
        grip = self.condition(me)
        v_t = self.target_speed(s, grip)

        # And respecting the car in front. Arriving at someone's gearbox with a
        # 20 kph overlap is not an overtake, it is a shunt.
        for other in cars:
            if other.index == self.index or other.retired:
                continue
            gap = self.track.delta_s(other.frenet.s, s)
            if not (0.0 < gap < 40.0):
                continue
            if abs(me.frenet.e_y - other.frenet.e_y) > 2.6:
                continue  # alongside, not behind: no need to lift
            # The standard safe-following limit: the fastest we can be here and
            # still slow to the other car's speed within the gap we have.
            #
            #     v^2 = v_other^2 + 2 * a * (gap - safe)
            #
            # A linear "match its speed plus a bit per metre of gap" rule --
            # which is what this was first -- cannot cope with the car ahead
            # braking from 240 to 67 kph for Turn 1. It permits far too much
            # closing speed at 30 m and then demands an impossible deceleration
            # at 10, so the queue concertinas into the back of itself. This form
            # is conservative early and relaxed late, which is the right way
            # round.
            # The gap to hold is a time, not a distance -- 0.12 s of travel plus
            # a car length. A fixed distance is far too little at 240 kph and
            # needlessly timid in a hairpin.
            safe = 8.0 + 0.15 * v
            brake = 3.0 * 9.81
            reachable = math.sqrt(
                max(0.0, other.speed ** 2 + 2.0 * brake * (gap - safe)))
            # The floor is not cosmetic. Inside the safe gap the formula asks
            # for a dead stop, so two cars that ended up 5 m apart both target
            # zero, both stop, and the gap never opens again: a whole queue of
            # cars sat motionless on the exit of Turn 1 waiting for each other.
            # A crawl always being available is what lets a stopped train
            # unstick itself, which is also what happens in traffic.
            v_t = min(v_t, max(reachable, 5.0))

        # How much of the tire is already spoken for sideways. Measured from
        # what the car is ACTUALLY doing -- its yaw rate over its speed is the
        # curvature of the path it is really on -- and not from the curvature of
        # the reference line.
        #
        # That distinction is the difference between a driver and a passenger.
        # Budgeting against the line says a car on a straight has all its grip
        # available for throttle, which is true right up until the moment the
        # back steps out; then the car is turning hard, the line still says
        # straight, and the driver keeps full power on and spins. Cars starting
        # from the back of the grid did exactly this on lap one, alone, with
        # nobody near them: full throttle, rear grip gone, sideways in under two
        # seconds. Using the real path curvature makes the budget close itself
        # as the slide develops.
        k_path = abs(st.r) / max(v, 1.0)
        k_eff = max(abs(k), k_path)
        lat_frac = 0.0
        if k_eff > 1e-5:
            v_max = self.vehicle.max_corner_speed(k_eff) * math.sqrt(grip)
            lat_frac = min(1.0, (v / max(v_max, 1.0)) ** 2)
        long_budget = math.sqrt(max(0.0, 1.0 - lat_frac * lat_frac))

        # And lift outright once the car is rotating faster than the corner
        # asks for. Opposite lock alone does not catch a power-on slide -- the
        # thing sustaining it is the throttle.
        yaw_error = abs(st.r - v * k)
        long_budget *= 1.0 - 0.9 * float(np.clip(yaw_error / 0.5, 0.0, 1.0))

        # The pedal is a fraction of grip available *now*, so asking for -1
        # while cornering means using every last newton longitudinally and
        # having nothing left to turn with.
        throttle = float(np.clip((v_t - v) * 0.35, -1.0, 1.0))
        return steer, float(np.clip(throttle, -long_budget, long_budget))

    @property
    def intent(self):
        """What the driver thinks it is doing. Handy when watching a replay."""
        return self._intent


def build_field(env, seed=0, pace_spread=0.05, base_pace=0.86):
    """One driver per car, with a spread of pace so there is a race on.

    A field of identical drivers produces a procession: nobody is quick enough
    to catch anybody. A spread of a few percent is enough that the order at the
    end is not the order at the start.
    """
    cfg = env.config
    vehicle = racing.Vehicle(cfg.vehicle)
    rng = np.random.default_rng(seed)

    drivers = []
    for car in env.cars:
        pace = base_pace + float(rng.uniform(-pace_spread, pace_spread))
        drivers.append(RacecraftDriver(
            env.track, vehicle, car.index, car.team,
            pace=pace, aggression=float(rng.uniform(0.25, 0.9)),
            seed=seed))
    return drivers


def drive(drivers, env):
    """Actions for the whole field, as a [n_cars, 2] float32 array."""
    cars = env.cars
    out = np.zeros((len(drivers), 2), dtype=np.float32)
    for i, d in enumerate(drivers):
        out[i] = d.act(cars[i], cars)
    return out
