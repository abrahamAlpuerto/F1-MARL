"""Turn raw FastF1 position telemetry into an engine-ready track.

    python tools/build_track.py

Reads `data/bahrain_track.json` (the FastF1 export already in the repo) and writes
`data/tracks/bahrain.json`, which is what the C++ engine actually loads.

Why this is not a one-liner
---------------------------
The vehicle model needs curvature, and curvature is the *second* derivative of
position. Two properties of the source data make that hard, and both were found
by measurement rather than assumed.

**1. The file carries two different distance coordinates, and only one is
trustworthy.** `tools/extract_circuit.py` resamples X/Y evenly against FastF1's
`Distance` channel, which is integrated from ECU speed. So sample `i` sits at
`i * lap_distance / n`, and consecutive samples ought to be one fixed step
apart. They are not: measured chord lengths run from 4.16 m to 19.22 m around a
nominal 10.4766 m. Geometry explains almost none of that -- in a 20 m-radius
corner the chord falls only ~1% short of the arc -- so the spread is error in
the position channel, which is a separate, lower-rate feed interpolated onto the
speed-derived distance.

The consequence is that measuring arc length off the chords folds position error
into the parameterisation itself, and manufactures corners that were never
driven. Doing exactly that put an impossible 10.8 g at T6. This script therefore
parameterises by the trusted distance and treats the chord variation as what it
is: noise to be filtered.

**2. Linear upsampling injects fake curvature.** A polyline has zero curvature
along each segment and a delta spike at every joint, so interpolating the 512
points onto a fine grid and then filtering leaves structure at the original
10.5 m spacing that no filter can distinguish from real corners. Resampling
spectrally never creates it.

What comes out
--------------
Perpendicular distance from the raw telemetry points to the shipped line is
0.05 m median and 0.47 m at p90, with the largest errors (~2.4 m) at the T1, T8
and T10 apexes -- the slowest corners, where the position channel is worst and
any filter must round off the tightest radius. Corner radii land on the real
circuit: T1 21.1 m against a true ~21 m.

A caveat that matters, and is deliberately not hidden
-----------------------------------------------------
This telemetry is Verstappen's fastest lap, so the line is a *racing line*, not
the track centerline. A racing line straightens corners by cutting the apex, so
its curvature is genuinely lower than the centerline's. We do not have track
boundary data to recover the true centerline from a single lap, so the engine
treats this as a *reference line* and builds the drivable corridor around it.

Two consequences, both fine, both worth knowing:

  * The corridor approximates the real track limits -- it is centred on the
    racing line rather than between the kerbs.
  * Calibration gets sharper, not weaker. The real car set 92.608 s on exactly
    this line, so a quasi-steady-state lap sim over this curvature should
    reproduce 92.608 s if the vehicle model is right.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- geometry --------------------------------------------------------------


def spectral_resample_lowpass(sig, n_out, ds_out, cutoff_m, order):
    """Upsample a periodic signal and low-pass it in one frequency-domain step.

    The filter is Butterworth rather than Gaussian, and that choice is load
    bearing. A Gaussian starts rolling off immediately, so every wavelength is
    attenuated a little -- including the ones that *are* the corners. Filtering
    hard enough to reject this data's noise then measurably straightens the
    circuit: at a comparable width the rebuilt line drifted 11.7 m from the real
    one, wider than the track itself, while T1 inflated from its true ~21 m
    radius to 25.6 m. A Butterworth response is flat until the cutoff and then
    falls steeply, so corner wavelengths pass essentially untouched.

    Applied as a real, symmetric multiplier, which makes it exactly zero-phase:
    a corner comes out where it actually is rather than shifted down the road.
    """
    n_in = len(sig)
    f_in = np.fft.rfft(sig)
    if n_in % 2 == 0:                 # split Nyquist so the result stays real
        f_in[-1] *= 0.5
    f_out = np.zeros(n_out // 2 + 1, dtype=complex)
    keep = min(len(f_in), len(f_out))
    f_out[:keep] = f_in[:keep]
    f_out *= float(n_out) / n_in

    f = np.fft.rfftfreq(n_out, d=ds_out)
    f_out *= 1.0 / (1.0 + (f * cutoff_m) ** (2 * order))
    return np.fft.irfft(f_out, n_out)


def build_reference_line(x_raw, y_raw, n_out, cutoff_m, order=4, length=None):
    """Smooth the line, then rebuild it by integrating its heading.

    `length` is the trusted lap distance (see module docstring). Returns
    `(x, y, theta, kappa, ds, geo_length, tau)`, where `tau` gives the trusted
    distance coordinate at each uniform arc-length sample -- the two coordinates
    drift locally against each other, so anything else indexed by trusted
    distance (the speed trace, corner markers) must be mapped through it rather
    than assumed to line up.

    Curvature is exactly consistent with the returned line: the geometry is
    reconstructed by integrating the same heading that curvature differentiates,
    so what the renderer draws and what the physics integrates cannot disagree.
    """
    n_in = len(x_raw)
    if length is None:
        length = float(np.hypot(np.diff(np.append(x_raw, x_raw[0])),
                                np.diff(np.append(y_raw, y_raw[0]))).sum())

    # Smooth in the trusted coordinate: sample j sits at j * length / n_out.
    ds_tau = length / n_out
    xs = spectral_resample_lowpass(x_raw, n_out, ds_tau, cutoff_m, order)
    ys = spectral_resample_lowpass(y_raw, n_out, ds_tau, cutoff_m, order)

    # Re-parameterise onto true uniform arc length, so the engine's `s` really
    # is metres, and keep the map back to the trusted coordinate.
    seg = np.hypot(np.diff(np.append(xs, xs[0])), np.diff(np.append(ys, ys[0])))
    s_cur = np.concatenate([[0.0], np.cumsum(seg)])
    geo_len = float(s_cur[-1])
    ds = geo_len / n_out
    grid = np.arange(n_out) * ds
    xu = np.interp(grid, s_cur, np.append(xs, xs[0]))
    yu = np.interp(grid, s_cur, np.append(ys, ys[0]))
    tau = np.interp(grid, s_cur, np.arange(n_out + 1) * ds_tau)

    # Heading of each segment, unwrapped so it accumulates rather than jumping
    # at +/-pi. Around a closed circuit the total must be one winding of 2*pi.
    dx = np.roll(xu, -1) - xu
    dy = np.roll(yu, -1) - yu
    theta = np.unwrap(np.arctan2(dy, dx))

    turns = round((theta[-1] - theta[0]) / (2 * math.pi))
    if turns == 0:
        raise ValueError("track does not close: net heading change is ~0")

    # Split heading into the linear ramp carrying the winding plus a periodic
    # remainder. The remainder is what can be differentiated spectrally; the
    # ramp contributes the constant term to curvature. The remainder's mean is
    # the circuit's absolute orientation and must be carried through untouched,
    # or the whole track comes out rotated.
    #
    # No second filter here. The line was already band-limited above, and
    # filtering its heading again would attenuate corner wavelengths twice and
    # quietly straighten the circuit.
    s = np.arange(n_out) * ds
    ramp = 2 * math.pi * turns * s / geo_len
    per = theta - ramp
    orientation = per.mean()
    per0 = per - orientation
    theta_s = per0 + orientation + ramp

    freqs = 2 * math.pi * np.fft.fftfreq(n_out, d=ds)
    kappa = np.real(np.fft.ifft(1j * freqs * np.fft.fft(per0))) + (
        2 * math.pi * turns / geo_len
    )

    # Rebuild the line from the heading, so geometry and curvature agree exactly.
    x = np.concatenate([[0.0], np.cumsum(np.cos(theta_s))[:-1]]) * ds
    y = np.concatenate([[0.0], np.cumsum(np.sin(theta_s))[:-1]]) * ds

    # Integration leaves a small closure residual. Spread it linearly along the
    # lap: this perturbs curvature by O(gap / length^2) ~ 1e-7 1/m against
    # curvatures up to 5e-2 1/m, invisible to the physics, but it makes the loop
    # join exactly.
    gap_x = x[-1] + ds * math.cos(theta_s[-1]) - x[0]
    gap_y = y[-1] + ds * math.sin(theta_s[-1]) - y[0]
    frac = s / geo_len
    x -= gap_x * frac
    y -= gap_y * frac

    # Anchor on the real start/finish.
    x += xu[0] - x[0]
    y += yu[0] - y[0]

    return x, y, theta_s, kappa, ds, geo_len, tau


def perpendicular_deviation(x, y, xr, yr):
    """Min distance from each raw telemetry point to the closed polyline.

    Perpendicular rather than same-index, because the two distance coordinates
    drift locally against each other. Comparing point-to-point would charge that
    longitudinal drift as if it were shape error -- it reports ~13.8 m where the
    true transverse error is under half a metre.
    """
    A = np.stack([x, y], 1)
    AB = np.roll(A, -1, axis=0) - A
    L2 = np.maximum((AB**2).sum(1), 1e-12)
    out = np.empty(len(xr))
    for i, (px, py) in enumerate(zip(xr, yr)):
        AP = np.stack([px - A[:, 0], py - A[:, 1]], 1)
        t = np.clip((AP * AB).sum(1) / L2, 0.0, 1.0)
        proj = A + AB * t[:, None]
        out[i] = np.min(np.hypot(proj[:, 0] - px, proj[:, 1] - py))
    return out


# --- main ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(REPO, "data", "bahrain_track.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "tracks", "bahrain.json"))
    ap.add_argument("--n", type=int, default=2048, help="uniform arc-length samples")
    ap.add_argument(
        "--cutoff-m",
        type=float,
        default=90.0,
        help="low-pass cutoff wavelength, in metres of track. Chosen by sweep: "
             "the widest setting that still reproduces the real corner radii, "
             "and the narrowest that keeps implied lateral load physical.",
    )
    ap.add_argument("--order", type=int, default=4, help="Butterworth order")
    ap.add_argument(
        "--half-width",
        type=float,
        default=None,
        help="corridor half width (default: track_width_m/2 from the raw file)",
    )
    args = ap.parse_args()

    raw = json.load(open(args.raw))
    upm = raw["units_per_m"]
    x_raw = np.array(raw["line_x"], dtype=float) / upm
    y_raw = np.array(raw["line_y"], dtype=float) / upm
    z_raw = np.array(raw["line_z"], dtype=float) / upm
    v_raw = np.array(raw["speed_kph"], dtype=float) / 3.6
    trusted_len = float(raw["lap_distance_m"])

    x, y, theta, kappa, ds, length, tau = build_reference_line(
        x_raw, y_raw, args.n, args.cutoff_m, args.order, trusted_len
    )

    # Left-hand normal: the tangent rotated +90 degrees.
    nx, ny = -np.sin(theta), np.cos(theta)

    # Speed and elevation are indexed by the trusted coordinate, so map them
    # through `tau` rather than assuming the two coordinates line up. Speed is
    # the calibration target; elevation is for the renderer only and never
    # reaches the physics (the vehicle model is planar by design). It is
    # carried through for the visualizer, which does want it.
    s_src = np.arange(len(x_raw) + 1) * (trusted_len / len(x_raw))
    v_ref = np.interp(tau, s_src, np.append(v_raw, v_raw[0]))
    z_ref = np.interp(tau, s_src, np.append(z_raw, z_raw[0]))

    half_w = args.half_width
    if half_w is None:
        half_w = float(raw.get("track_width_m", 15.0)) / 2.0

    # --- quality report ----------------------------------------------------
    dev = perpendicular_deviation(x, y, x_raw, y_raw)
    lat_g = v_ref**2 * np.abs(kappa) / 9.81

    print(f"track          : {raw.get('display_name', raw['track'])}")
    print(f"length         : {length:.2f} m over {args.n} samples (ds = {ds:.3f} m)")
    print(f"                 trusted distance {trusted_len:.2f} m "
          f"({100*(length-trusted_len)/trusted_len:+.3f}%)")
    print(f"low-pass       : {args.cutoff_m:.0f} m cutoff, order {args.order}")
    print(f"line deviation : p50 {np.percentile(dev,50):.2f} m, p90 "
          f"{np.percentile(dev,90):.2f} m, max {dev.max():.2f} m "
          f"(corridor half-width {half_w:.1f} m)")
    print(f"curvature      : |k| max {np.abs(kappa).max():.5f} 1/m "
          f"-> R_min {1/np.abs(kappa).max():.1f} m")
    print(f"implied lat g  : max {lat_g.max():.2f} g, p99 "
          f"{np.percentile(lat_g, 99):.2f} g, p95 {np.percentile(lat_g, 95):.2f} g")

    if dev.max() > half_w:
        print("  WARNING: the rebuilt line leaves the corridor somewhere.")

    # Corner markers are given as a fraction of the trusted distance, so they
    # need the same mapping as the speed trace.
    tau_wrapped = np.concatenate([tau, [trusted_len]])
    s_wrapped = np.concatenate([np.arange(args.n) * ds, [length]])
    corners_out = []
    print("\ncorner        s (m)     R (m)   v_real (kph)   lat g")
    for c in raw.get("corners", []):
        s_c = float(np.interp(c["frac"] * trusted_len, tau_wrapped, s_wrapped))
        i = int(round(s_c / ds)) % args.n
        r = 1.0 / max(abs(kappa[i]), 1e-9)
        corners_out.append({**c, "s_m": round(s_c, 2)})
        print(f"  T{c['number']}{c['letter']:<2}      {s_c:8.1f} {r:9.1f}   "
              f"{v_ref[i]*3.6:8.1f}   {lat_g[i]:6.2f}")

    sector_s = [
        float(np.interp(float(f) * trusted_len, tau_wrapped, s_wrapped))
        for f in (raw.get("sector_boundaries_frac") or [])
    ]

    out = {
        "name": raw["track"],
        "display_name": raw.get("display_name", raw["track"]),
        "source": raw.get("source", ""),
        "_provenance": {
            "line": "Verstappen fastest lap, 2024 Bahrain GP (FastF1 position telemetry)",
            "line_is": "racing line, NOT track centerline -- see tools/build_track.py",
            "lowpass_cutoff_m": args.cutoff_m,
            "lowpass_order": args.order,
            "perp_deviation_p50_m": round(float(np.percentile(dev, 50)), 3),
            "perp_deviation_p90_m": round(float(np.percentile(dev, 90)), 3),
            "perp_deviation_max_m": round(float(dev.max()), 3),
            "half_width_m": half_w,
            "half_width_provenance": raw.get("track_width_provenance", "authored"),
        },
        "lap_length_m": length,
        "n": args.n,
        "ds_m": ds,
        "reference_lap_time_s": raw.get("reference_lap_time_s"),
        "x": [round(float(v), 4) for v in x],
        "y": [round(float(v), 4) for v in y],
        "z": [round(float(v), 4) for v in z_ref],
        "theta": [round(float(v), 6) for v in theta],
        "kappa": [float(v) for v in kappa],
        "nx": [round(float(v), 6) for v in nx],
        "ny": [round(float(v), 6) for v in ny],
        "half_width_left": [half_w] * args.n,
        "half_width_right": [half_w] * args.n,
        "sector_s": sector_s,
        "corners": corners_out,
        "telemetry_speed_ms": [round(float(v), 3) for v in v_ref],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nwrote {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
