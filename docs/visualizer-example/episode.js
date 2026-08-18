/**
 * Reading a race out of the engine's feed.
 *
 * Copy this file into your project. It is a plain ES module with JSDoc types,
 * so it runs in a browser with no build step and still type-checks under
 * TypeScript with `allowJs` + `checkJs`.
 *
 * The one rule worth internalising: **index fields by name, never by number.**
 * `episode.json` ships a `fields` array and a `stride`, and fields get appended
 * to it. A loader that hardcodes "speed is index 4" works today and silently
 * renders throttle as lap number the first time the format grows.
 *
 *     const ep = await Episode.load("/episodes/demo");
 *     const car = ep.at(frame, 0);      // { x, y, z, speed, position, ... }
 *     ep.value(frame, 0, "speed");      // one field, no object allocated
 *
 * @module
 */

export class EpisodeFormatError extends Error {}

/** Bits in the per-car `flags` field. */
export const Flag = {
  OffTrack: 1,
  Contact: 2,
  Finished: 4,
  Retired: 8,
  DrsOpen: 16,
  Wheelspin: 32,
  Lockup: 64,
  ErsDeploying: 128,
  // Set only for the frame or two of an impact, so latch it rather than
  // expecting it to persist -- by the time a viewer notices, the car has
  // already bounced off.
  Barrier: 256,
};

export class Episode {
  /**
   * @param {object} header parsed episode.json
   * @param {Float32Array} frames flat frame -> car -> field
   * @param {object|null} track parsed track.json, or null
   */
  constructor(header, frames, track = null) {
    validateHeader(header);

    const expected = header.n_frames * header.n_cars * header.stride;
    if (frames.length !== expected) {
      // A truncated download is the common failure in the field, and it must
      // not render as a blank screen with no explanation.
      throw new EpisodeFormatError(
        `frames.f32 holds ${frames.length} floats, but episode.json declares ` +
          `${header.n_frames} frames x ${header.n_cars} cars x ${header.stride} ` +
          `fields = ${expected}. The file is truncated or the header is stale.`
      );
    }

    this.header = header;
    this.frames = frames;
    this.track = track;

    /** field name -> index, built once. */
    this.fieldIndex = new Map(header.fields.map((name, i) => [name, i]));
  }

  static async load(baseUrl) {
    const base = baseUrl.replace(/\/$/, "");
    const [header, buffer, track] = await Promise.all([
      fetchJson(`${base}/episode.json`),
      fetchBuffer(`${base}/frames.f32`),
      fetchJson(`${base}/track.json`).catch(() => null),
    ]);
    if (buffer.byteLength % 4 !== 0) {
      throw new EpisodeFormatError(
        `frames.f32 is ${buffer.byteLength} bytes, not a whole number of float32 values.`
      );
    }
    // The engine writes little-endian float32. Every platform anyone will run
    // this on is little-endian, but say so rather than assume it silently.
    return new Episode(header, new Float32Array(buffer), track);
  }

  get nFrames() { return this.header.n_frames; }
  get nCars() { return this.header.n_cars; }
  get stride() { return this.header.stride; }
  get frameRate() { return this.header.frame_rate; }
  get duration() { return this.nFrames / this.frameRate; }
  get events() { return this.header.events ?? []; }
  get result() { return this.header.result ?? {}; }

  /** Index of a named field. Throws rather than returning -1. */
  index(name) {
    const i = this.fieldIndex.get(name);
    if (i === undefined) {
      throw new EpisodeFormatError(
        `no field "${name}" in this episode; it has: ${this.header.fields.join(", ")}`
      );
    }
    return i;
  }

  /** True if this feed carries a given field. Use it to degrade gracefully. */
  has(name) { return this.fieldIndex.has(name); }

  /** One field of one car at one frame. No allocation. */
  value(frame, car, name) {
    return this.frames[(frame * this.nCars + car) * this.stride + this.index(name)];
  }

  /** Every field of one car at one frame, as a named object. */
  at(frame, car) {
    const base = (frame * this.nCars + car) * this.stride;
    const out = {};
    this.header.fields.forEach((name, i) => { out[name] = this.frames[base + i]; });
    return out;
  }

  carInfo(car) { return this.header.cars[car]; }
  teamInfo(team) { return this.header.teams[team]; }
  colorOf(car) { return this.teamInfo(this.carInfo(car).team).color; }

  /** Cars at one frame, ordered by race position rather than by car number. */
  leaderboard(frame) {
    const pos = this.index("position");
    const order = [];
    for (let c = 0; c < this.nCars; c++) {
      order.push({ car: c, position: this.frames[(frame * this.nCars + c) * this.stride + pos] });
    }
    order.sort((a, b) => a.position - b.position);
    return order;
  }

  /** Events landing in [from, to). The list is already sorted by frame. */
  eventsInRange(from, to) {
    return this.events.filter((e) => e.frame >= from && e.frame < to);
  }

  /**
   * Track edges as two arrays of [x, y].
   *
   * `track.json` ships a centre line and half widths but no normals, because
   * they are trivially derivable and shipping them would be one more thing that
   * can disagree with the line. Derive them with a CENTRAL difference: the
   * circuit is a closed loop, and a forward difference leaves a visible kink at
   * the start/finish point.
   */
  trackEdges() {
    if (!this.track) throw new EpisodeFormatError("this episode was loaded without track.json");
    const line = this.track.line;
    const wl = this.track.half_width_left;
    const wr = this.track.half_width_right;
    const n = line.length;
    const left = [];
    const right = [];
    for (let i = 0; i < n; i++) {
      const a = line[(i - 1 + n) % n];
      const b = line[(i + 1) % n];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      left.push([line[i][0] + nx * wl[i], line[i][1] + ny * wl[i]]);
      right.push([line[i][0] - nx * wr[i], line[i][1] - ny * wr[i]]);
    }
    return { left, right };
  }
}

/**
 * Parse a live `stream.jsonl` into the same shape.
 *
 * The stream is one self-contained JSON object per line: a `header`, then a
 * `frame` or `event` per line as the race runs, then a `result`. Tail it to
 * draw a race live; this function is for loading a finished one.
 *
 * Unknown line types are ignored on purpose -- more of them are expected, and
 * a reader that throws on the first one it does not recognise will break the
 * next time a field is added.
 */
export function parseStream(text) {
  let header = null;
  const rows = [];
  const events = [];
  let result = {};

  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    switch (obj.type) {
      case "header": header = obj; break;
      case "frame": rows.push(obj.cars); break;
      case "event": events.push(obj.event); break;
      case "result": result = obj; break;
      default: break;
    }
  }
  if (!header) throw new EpisodeFormatError("stream has no header line");

  const stride = header.stride;
  const flat = new Float32Array(rows.length * header.n_cars * stride);
  let k = 0;
  for (const frame of rows) {
    for (const car of frame) {
      for (let i = 0; i < stride; i++) flat[k++] = car[i];
    }
  }
  return new Episode(
    { ...header, n_frames: rows.length, events, result },
    flat
  );
}

// --- internals --------------------------------------------------------------

function validateHeader(header) {
  for (const key of ["n_frames", "n_cars", "stride", "fields", "frame_rate"]) {
    if (header[key] === undefined) {
      throw new EpisodeFormatError(`episode.json is missing "${key}"`);
    }
  }
  if (header.fields.length !== header.stride) {
    throw new EpisodeFormatError(
      `episode.json says stride ${header.stride} but lists ${header.fields.length} field names`
    );
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new EpisodeFormatError(`${url}: HTTP ${res.status}`);
  return res.json();
}

async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new EpisodeFormatError(`${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}
