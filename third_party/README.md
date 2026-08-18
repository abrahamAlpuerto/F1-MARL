# Vendored dependencies

These are committed deliberately. `ENGINE_BUILD_BRIEF.md` §2 allows no external
runtime dependencies beyond the standard library and pybind11, and vendoring
these two headers means `git clone && cmake --build build` works with no network
access and no package manager.

Both are single-header and unmodified from upstream. To verify, download the
same version and diff — `.gitattributes` marks this directory `-diff` so the
files stay byte-identical to what upstream shipped.

| File | Project | Version | License |
|---|---|---|---|
| `nlohmann/json.hpp` | [nlohmann/json](https://github.com/nlohmann/json) | v3.11.3 | MIT |
| `catch2/catch.hpp` | [catchorg/Catch2](https://github.com/catchorg/Catch2) | v2.13.10 | BSL-1.0 |

Both licenses permit redistribution with attribution; the full license text is
embedded in the header of each file.

## Updating

```bash
curl -L -o third_party/nlohmann/json.hpp \
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
curl -L -o third_party/catch2/catch.hpp \
  https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp
```

Catch2 is pinned to the v2 line on purpose: v3 is no longer single-header and
would require adding a build dependency.

## What is *not* vendored

**pybind11** comes from pip (`pip install pybind11`), because it has to match
the Python interpreter you are building against. CMake locates it by asking that
interpreter for `python -m pybind11 --cmakedir`.
