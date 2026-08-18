"""
Pull real circuit geometry and a speed trace out of FastF1.

    python tools/extract_circuit.py --year 2024 --out data/bahrain_track.json

This is the RAW extract, and it is not what the engine loads. The pipeline is:

    extract_circuit.py   timing API  ->  data/bahrain_track.json   (raw)
    build_track.py       raw         ->  data/tracks/bahrain.json  (engine)

`build_track.py` is where the care is: the raw file carries two distance
coordinates that drift against each other, and reading it naively invents
corners nobody drove. Read that script before trusting anything here.

Both outputs are committed, so this only needs running to add a circuit or to
re-pull one. It hits the network; nothing else in the repository does.

WHY THE SECTOR DISTANCES MATTER
-------------------------------
Sector boundaries are published as shares of lap TIME. Bahrain's sector 2 is
~43% of the lap time but a different share of the lap DISTANCE, because the
sectors are not equally fast. Placing a car by treating a time share as a
distance share puts it in the wrong part of the circuit. This script emits the
distance fractions of the real sector boundaries so the visualizer can convert
properly.

SOURCE
------
The line is one real fast lap's position telemetry, i.e. the path a real F1 car
actually drove, not an idealised centreline.
"""

import argparse
import json
import math
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import fastf1


def resample_by_distance(dist, x, y, z, n_points):
    """Even samples along the lap, so the visualizer can index by lap fraction."""
    total = float(dist.max())
    grid = np.linspace(0.0, total, n_points, endpoint=False)
    return (grid,
            np.interp(grid, dist, x),
            np.interp(grid, dist, y),
            np.interp(grid, dist, z),
            total)


def speed_profile(tel, grid, dist):
    """
    Speed and cumulative lap time at each resampled point, from the same real
    lap as the geometry.

    This is what lets a consumer show cars braking into corners. The engine has
    no corner-by-corner speed -- it produces one time per sector -- so without a
    profile a car has to be drawn moving at a constant rate between sector
    boundaries, which looks wrong at every braking zone.

    `time_frac` is the share of the lap TIME elapsed at each point. Inverting it
    against distance gives "where is the car after t seconds", which slows the
    car through slow corners and speeds it up on the straights while still
    hitting the engine's sector times exactly. The car's overall pace still
    comes entirely from the engine; this only shapes how that pace is spent
    around the lap.
    """
    speed = (np.interp(grid, dist, tel["Speed"].to_numpy(dtype=float))
             if "Speed" in tel.columns else np.zeros_like(grid))

    t = tel["Time"].dt.total_seconds().to_numpy()
    t = t - t[0]
    cum = np.interp(grid, dist, t)
    # Force monotonic: telemetry timestamps can tie after resampling, and a
    # non-monotonic curve would make the inversion in the replay layer jump.
    cum = np.maximum.accumulate(cum)
    lap_time = float(t[-1] - t[0]) or 1.0
    return speed, cum, cum / lap_time, lap_time


def sector_boundary_distances(lap, tel):
    """
    Distance along the lap at which sectors 1 and 2 end.

    Sector times are reported as session timestamps, and the telemetry carries
    its own session time, so the boundary is found by interpolating distance at
    those instants rather than assuming anything about where they fall.
    """
    if "SessionTime" not in tel.columns:
        return None
    t = tel["SessionTime"].dt.total_seconds().to_numpy()
    d = tel["Distance"].to_numpy(dtype=float)
    out = []
    for key in ("Sector1SessionTime", "Sector2SessionTime"):
        v = lap[key]
        if v is None or (hasattr(v, "__class__") and str(v) == "NaT"):
            return None
        ts = v.total_seconds()
        if ts < t[0] or ts > t[-1]:
            return None
        out.append(float(np.interp(ts, t, d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--event", default="Bahrain")
    ap.add_argument("--session", default="R")
    ap.add_argument("--points", type=int, default=512)
    ap.add_argument("--out", default="data/bahrain_track.json")
    ap.add_argument("--cache", default=".fastf1_cache")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)

    s = fastf1.get_session(args.year, args.event, args.session)
    s.load(telemetry=True, weather=False, messages=False)

    lap = s.laps.pick_fastest()
    tel = lap.get_telemetry()
    tel = tel[tel["Distance"].notna() & tel["X"].notna() & tel["Y"].notna()]
    tel = tel.sort_values("Distance")

    dist = tel["Distance"].to_numpy(dtype=float)
    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)
    z = (tel["Z"].to_numpy(dtype=float) if "Z" in tel.columns
         else np.zeros_like(x))

    grid, gx, gy, gz, total = resample_by_distance(dist, x, y, z, args.points)
    gspeed, gcum, gtfrac, ref_lap_time = speed_profile(tel, grid, dist)

    # Sector boundaries as fractions of lap distance.
    bounds = sector_boundary_distances(lap, tel)
    if bounds:
        sector_frac = [round(b / total, 6) for b in bounds] + [1.0]
        sector_provenance = "fitted"
    else:
        sector_frac = None
        sector_provenance = "unavailable"

    # Sector TIME shares from the same lap, for comparison. These are what the
    # engine emits, and they are deliberately different numbers.
    s1 = lap["Sector1Time"].total_seconds()
    s2 = lap["Sector2Time"].total_seconds()
    s3 = lap["Sector3Time"].total_seconds()
    tt = s1 + s2 + s3
    sector_time_share = [round(s1 / tt, 6), round(s2 / tt, 6), round(s3 / tt, 6)]

    corners = []
    rotation = 0.0
    try:
        ci = s.get_circuit_info()
        rotation = float(ci.rotation)
        for _, c in ci.corners.iterrows():
            corners.append({
                "number": int(c["Number"]),
                "letter": str(c.get("Letter", "") or ""),
                "x": round(float(c["X"]), 2),
                "y": round(float(c["Y"]), 2),
                "frac": round(float(c["Distance"]) / total, 6),
            })
    except Exception as e:
        print(f"  !! circuit info unavailable: {e}")

    # Unit normal at each point, so a consumer can offset a car sideways from
    # the racing line. Cars do not share one line (CarState::lateral), and
    # without normals there is no way to draw two of them side by side.
    n = len(gx)
    nx = np.zeros(n)
    ny = np.zeros(n)
    for i in range(n):
        # Central difference around the loop gives a smoother tangent than a
        # forward difference, which matters in the tight infield corners.
        dx = gx[(i + 1) % n] - gx[(i - 1) % n]
        dy = gy[(i + 1) % n] - gy[(i - 1) % n]
        mag = math.hypot(dx, dy) or 1.0
        nx[i], ny[i] = -dy / mag, dx / mag        # rotate tangent 90 degrees

    # Telemetry coordinates are not metres; measure the conversion off the
    # polyline so a consumer can turn a track width in metres into units.
    perim = sum(math.hypot(gx[(i + 1) % n] - gx[i], gy[(i + 1) % n] - gy[i])
                for i in range(n))
    units_per_m = perim / total

    spec = {
        "track": "bahrain",
        "display_name": "Bahrain International Circuit",
        "_note": (
            "RAW EXTRACT. The engine does not load this file directly -- run "
            "tools/build_track.py to turn it into data/tracks/bahrain.json, "
            "which is what Track::load reads."
        ),
        "source": (
            f"FastF1 position telemetry, {args.year} {args.event} "
            f"{args.session}, fastest lap ({lap['Driver']}, {lap['LapTime']})"
        ),
        "lap_distance_m": round(total, 2),
        "rotation_deg": rotation,
        "n_points": int(args.points),
        # Real driven line, resampled at even distance intervals. Index i sits
        # at lap fraction i / n_points.
        "line_x": [round(float(v), 2) for v in gx],
        "line_y": [round(float(v), 2) for v in gy],
        "line_z": [round(float(v), 2) for v in gz],
        "elevation_range": [round(float(gz.min()), 2), round(float(gz.max()), 2)],
        "norm_x": [round(float(v), 5) for v in nx],
        "norm_y": [round(float(v), 5) for v in ny],
        # Real speed trace, and the share of lap time elapsed at each point.
        # Inverting time_frac against distance is what makes a car brake into a
        # corner instead of sliding round at a constant rate.
        "speed_kph": [round(float(v), 1) for v in gspeed],
        "time_frac": [round(float(v), 6) for v in gtfrac],
        "reference_lap_time_s": round(ref_lap_time, 3),
        "speed_range_kph": [round(float(gspeed.min()), 1),
                            round(float(gspeed.max()), 1)],
        "units_per_m": round(float(units_per_m), 4),
        "track_width_m": 15.0,
        "track_width_provenance": "tuned -- Bahrain's racing surface is ~15 m",
        "sector_boundaries_frac": sector_frac,
        "sector_boundaries_provenance": sector_provenance,
        "sector_time_share": sector_time_share,
        "corners": corners,
        "start_finish": {"x": round(float(gx[0]), 2),
                         "y": round(float(gy[0]), 2),
                         "z": round(float(gz[0]), 2)},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(spec, f, indent=1)

    print(f"wrote {args.out}")
    print(f"  source            : {spec['source']}")
    print(f"  lap distance      : {total:.1f} m")
    print(f"  line points       : {args.points}")
    print(f"  corners           : {len(corners)}")
    print(f"  sector dist frac  : {sector_frac}")
    print(f"  sector time share : {sector_time_share}")
    if sector_frac:
        print("  -> distance and time shares differ; the visualizer must use "
              "the distance fractions to place cars.")


if __name__ == "__main__":
    main()
