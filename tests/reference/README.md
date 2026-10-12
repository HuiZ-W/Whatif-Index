# Test-only causal reference

This fixed-width causal-DAG implementation is a regression oracle for the
production arbitrary-width implementation. In particular, `wide_scope_tests`
embeds a small DAG across machine-word boundaries and compares the resulting
causal relations and adjustment sets.

These files belong exclusively to the `whatif_test_reference` CMake target, under
`BUILD_TESTING`. None of the four production executables links this target.
The production causal model is `src/CausalAdjustment.cpp`, exposed through
`include/whatif/CausalAdjustment.h`.
