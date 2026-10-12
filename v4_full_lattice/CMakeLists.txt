cmake_minimum_required(VERSION 3.16)
project(causal_index_v4_full_lattice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(causal_index_v4
  src/BoundedScopeCache.cpp
  src/CausalAdjustment.cpp
  src/FullLatticeIndex.cpp
  src/FullJoinRollup.cpp
  src/JoinSizeGreedyWorkingPath.cpp
  src/ScopeOrderedWorkingPath.cpp
)

target_include_directories(causal_index_v4 PUBLIC include)
target_compile_options(causal_index_v4 PRIVATE -Wall -Wextra -Wpedantic)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(causal_index_v4_tests tests/v4_tests.cpp)
  target_link_libraries(causal_index_v4_tests PRIVATE causal_index_v4)

  add_executable(causal_index_v4_ldbc_assignment_benchmark tests/ldbc_assignment_benchmark.cpp)
  target_link_libraries(causal_index_v4_ldbc_assignment_benchmark PRIVATE causal_index_v4)

  enable_testing()
  add_test(NAME causal_index_v4_tests COMMAND causal_index_v4_tests)

  foreach(causal_case IN ITEMS chain fork confounded)
    add_test(
      NAME causal_workload_${causal_case}_test
      COMMAND ${CMAKE_COMMAND}
        -DBENCHMARK=$<TARGET_FILE:causal_index_v4_ldbc_assignment_benchmark>
        -DCASE=${causal_case}
        -DASSIGNMENTS=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/causal_workload_${causal_case}_assignments.tsv
        -DCAUSAL=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/causal_workload_${causal_case}.tsv
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_causal_workload.cmake
    )
  endforeach()
endif()
