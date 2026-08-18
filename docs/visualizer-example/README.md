# Reference viewer

A worked example of reading the racing feed. It is **not** the visualizer — it
exists so you can see the data working before deciding how to build the real
thing, and so the parsing details are written down somewhere rather than
rediscovered.

Copy what is useful. Delete the rest.

```
episode.js    the loader. This is the file worth copying. Plain ES module with
              JSDoc types: runs in a browser with no build step, and type-checks
              under TypeScript with allowJs + checkJs.
index.html    a runnable viewer: circuit, playback, scrubbing, wake shading,
              live classification, event ticker.
```

There used to be a separate `episode.ts` carrying the same code with real type
annotations. It is gone: two copies of one loader is a thing that rots, and the
JSDoc version gives a TypeScript project the same types without the duplicate.

## Run it

Episodes are generated rather than committed, so make one first. From the
repository root:

```bash
python examples/race.py --laps 3 --name demo
```

Then serve the repository, because the files are fetched over HTTP and `file://`
is blocked by the browser:

```bash
python -m http.server 8777
```

Open <http://127.0.0.1:8777/docs/visualizer-example/index.html>.

The box at the top is the episode directory, relative to the page. Point it at
any episode you have generated.

No C++ toolchain? Ask for an `episodes/<name>/` folder and drop it anywhere.
Nothing here imports the engine, and nothing should.

## What you are looking at

Eight cars, four teams of two, on real Bahrain geometry. Cars are drawn in their
team colour, dimmed by `downforce` — so a car struggling in someone's dirty air
visibly fades — with a wake cone behind anything that is disturbing the air for
a follower. A red outline is contact this frame; amber is off the circuit.

The right-hand panel is the classification with live gaps, and an event ticker
underneath.

## What it deliberately does not do

Camera work, 3D, a track map minimap, sector timing, side-by-side replay of two
episodes, video export. Those are the actual visualizer and they are yours.

The loader is deliberately just a data object with no global state, so holding
two `Episode` instances at once — two runs of the same seed with different
policies, on one shared scrub bar — is not a problem. That comparison is worth
building; it is the clearest way to show what a trained policy is doing
differently.

## The format

See [`docs/VISUALIZER_FEED.md`](../VISUALIZER_FEED.md) for the full contract.
The short version:

- Three files. `track.json` is static per circuit, `episode.json` is the header,
  `frames.f32` is a flat little-endian float32 array laid out frame → car →
  field. Car `c` at frame `f` starts at `(f * n_cars + c) * stride`.
- **Index fields by name**, using the `fields` array in the header. Never by a
  hardcoded number — fields get appended.
- Read `n_cars` and `n_frames` from the header. Both are allowed to change.

If the engine's output ever disagrees with `episode.js`, that is an engine bug.
Say so rather than working around it, because a workaround silently forks the
format.
