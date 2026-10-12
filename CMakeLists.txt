cmake_minimum_required(VERSION 3.16)
project(WhatifIndex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED)
include(CTest)

# Shared execution primitives, not a separate implementation version.
add_library(whatif_core
  src/core/BoundedScopeCache.cpp
  src/core/FullLatticeIndex.cpp
  src/core/FullJoinRollup.cpp
  src/core/JoinSizeGreedyWorkingPath.cpp
  src/core/ScopeOrderedWorkingPath.cpp
)
target_include_directories(whatif_core PUBLIC include)
target_compile_options(whatif_core PRIVATE -Wall -Wextra -Wpedantic)

add_library(whatif_index
  src/CausalAdjustment.cpp
  src/AdjustmentScopePlanner.cpp
  src/BoundaryJoinIndex.cpp
  src/SharedConstruction.cpp
  src/BoundaryRequirementPlanner.cpp
  src/ParentTreePlanner.cpp
  src/Snapshot.cpp
)
target_link_libraries(whatif_index PUBLIC whatif_core)
target_include_directories(whatif_index SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
target_compile_options(whatif_index PRIVATE -Wall -Wextra -Wpedantic)

add_executable(whatif_build benchmarks/boundary_benchmark.cpp)
add_executable(whatif_query benchmarks/online_worker.cpp)
add_executable(whatif_online_join benchmarks/online_join_worker.cpp)
add_executable(whatif_materialize_boundaries benchmarks/materialize_online_boundaries.cpp)
foreach(program IN ITEMS whatif_build whatif_query whatif_online_join whatif_materialize_boundaries)
  target_link_libraries(${program} PRIVATE whatif_index)
endforeach()
target_compile_options(whatif_build PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  # Fixed-width causal reasoning is retained only as a regression-test oracle.
  # Production programs exclusively use src/CausalAdjustment.cpp.
  add_library(whatif_test_reference STATIC tests/reference/CausalAdjustment.cpp)
  target_include_directories(whatif_test_reference PUBLIC tests)

  foreach(test IN ITEMS boundary_join_tests wide_scope_tests full_join_rollup_tests
                        streaming_full_join_tests input_loading_tests join_metrics_tests
                        core_tests shared_construction_tests)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE whatif_index)
    target_compile_options(${test} PRIVATE -UNDEBUG)
    add_test(NAME ${test} COMMAND ${test})
  endforeach()
  target_link_libraries(wide_scope_tests PRIVATE whatif_test_reference)
  target_link_libraries(core_tests PRIVATE whatif_test_reference)

  add_executable(causal_workload_test_driver tests/support/causal_workload_driver.cpp)
  target_link_libraries(causal_workload_test_driver PRIVATE whatif_core whatif_test_reference)
  target_compile_options(causal_workload_test_driver PRIVATE -UNDEBUG)
  foreach(causal_case IN ITEMS chain fork confounded)
    add_test(NAME causal_workload_${causal_case}_test
      COMMAND ${CMAKE_COMMAND}
        -DBENCHMARK=$<TARGET_FILE:causal_workload_test_driver>
        -DCASE=${causal_case}
        -DASSIGNMENTS=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/causal_workload_${causal_case}_assignments.tsv
        -DCAUSAL=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/causal_workload_${causal_case}.tsv
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_causal_workload.cmake
    )
  endforeach()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME induced_scale_test
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_induced_scale.py)
    add_test(NAME paper_methods_test
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/paper_methods_test.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
    add_test(NAME cli_smoke
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_smoke.py
              --build-dir ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endif()
