#pragma once

#include "whatif/BoundaryJoinIndex.h"

namespace whatif {

// A variable ID denotes one immutable graph pattern/value expression. Shared
// roles and global injectivity are the fixed matching context of this API.
// Different pattern definitions must therefore have different variable IDs.
SharedConstructionStats constructSharedScopes(
    const std::vector<VariableInput> &variables,
    std::vector<std::vector<VariableId>> scopes,
    const BoundaryBuildConfig &config,
    const std::function<void(const std::vector<VariableId> &,
                             const std::vector<BoundaryRow> &)> &emit);

} // namespace whatif
