cmake_minimum_required(VERSION 3.20)
project(racing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(RACING_BUILD_TESTS    "Build C++ tests"        ON)
option(RACING_BUILD_BINDINGS "Build pybind11 module"  ON)

if(MSVC)
  add_compile_options(/W4 /permissive- /Zc:__cplusplus /bigobj /utf-8)
  # /fp:precise is the MSVC default, but state it: a replay is only reproducible
  # if the compiler does not reassociate float arithmetic behind our backs.
  add_compile_options(/fp:precise)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  # Same reason as /fp:precise above. -ffast-math silently breaks the
  # determinism tests in ways that only show up at some optimisation levels.
  add_compile_options(-fno-fast-math)
endif()

# --- third-party (vendored, header-only) ------------------------------------
add_library(racing_thirdparty INTERFACE)
target_include_directories(racing_thirdparty INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# --- engine -----------------------------------------------------------------
add_library(racing_core STATIC
  src/racing/config.cpp
  src/racing/track.cpp
  src/racing/vehicle.cpp
  src/racing/tyres.cpp
  src/racing/powertrain.cpp
  src/racing/qss.cpp
  src/racing/interaction.cpp
  src/racing/race.cpp
  src/racing/feed.cpp
)
target_include_directories(racing_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(racing_core PUBLIC racing_thirdparty Threads::Threads)
set_target_properties(racing_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- tools ------------------------------------------------------------------
add_executable(racing_calibrate tools/calibrate_vehicle.cpp)
target_link_libraries(racing_calibrate PRIVATE racing_core)

add_executable(racing_bench tools/bench.cpp)
target_link_libraries(racing_bench PRIVATE racing_core)
target_compile_definitions(racing_bench PRIVATE
  RACING_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

# --- tests ------------------------------------------------------------------
if(RACING_BUILD_TESTS)
  enable_testing()
  add_executable(racing_tests
    tests/test_racing_main.cpp
    tests/test_racing_vehicle.cpp
    tests/test_racing_track.cpp
    tests/test_racing_determinism.cpp
    tests/test_racing_interaction.cpp
    tests/test_racing_race.cpp
    tests/test_racing_feed.cpp
    tests/test_racing_physics.cpp
  )
  target_link_libraries(racing_tests PRIVATE racing_core)
  target_include_directories(racing_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_compile_definitions(racing_tests PRIVATE
    RACING_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME racing_tests COMMAND racing_tests)
endif()

# --- python bindings --------------------------------------------------------
if(RACING_BUILD_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development.Module QUIET)
  if(Python_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE PYBIND11_CMAKEDIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(PYBIND11_CMAKEDIR)
      list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
      find_package(pybind11 CONFIG QUIET)
    endif()
  endif()

  if(pybind11_FOUND)
    pybind11_add_module(_racing bindings/racing_module.cpp)
    target_link_libraries(_racing PRIVATE racing_core)
    # Land the module inside the python package so `import racing` works from
    # the source tree with no install step.
    set_target_properties(_racing PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/racing
      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/python/racing
      LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/python/racing
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/racing
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/python/racing
      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/python/racing)
  else()
    message(WARNING "pybind11 not found -- skipping python bindings.")
  endif()
endif()
