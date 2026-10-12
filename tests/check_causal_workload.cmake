execute_process(
  COMMAND "${BENCHMARK}" "${ASSIGNMENTS}" 4 parent-compare 0 1 0 6 6 "${CAUSAL}" 0 cost-model
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
)

if(NOT result EQUAL 0)
  message(FATAL_ERROR "causal workload benchmark failed (${result}):\n${output}\n${error}")
endif()

set(expected_common
  "projected_causal_output_scopes=0"
  "counts_match=yes"
)
if(CASE STREQUAL "chain")
  set(expected_case
    "parent_target_scopes=15"
    "adjustment_query_candidates=28"
    "multi_treatment_queries=16"
    "adjusted_queries=4"
    "tree_lattice_scope_candidates=15"
    "tree_lattice_scopes_retained=15"
    "tree_lattice_scopes_pruned=0"
  )
elseif(CASE STREQUAL "fork")
  set(expected_case
    "parent_target_scopes=11"
    "adjustment_query_candidates=28"
    "multi_treatment_queries=16"
    "adjusted_queries=9"
    "tree_lattice_scope_candidates=15"
    "tree_lattice_scopes_retained=11"
    "tree_lattice_scopes_pruned=4"
  )
elseif(CASE STREQUAL "confounded")
  set(expected_case
    "parent_target_scopes=6"
    "adjustment_query_candidates=9"
    "multi_treatment_queries=3"
    "adjusted_queries=1"
    "tree_lattice_scope_candidates=7"
    "tree_lattice_scopes_retained=6"
    "tree_lattice_scopes_pruned=1"
  )
else()
  message(FATAL_ERROR "unknown causal workload test case: ${CASE}")
endif()

foreach(expected IN LISTS expected_common expected_case)
  string(FIND "${output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "missing '${expected}' in benchmark output:\n${output}")
  endif()
endforeach()
