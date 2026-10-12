#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace whatif::testing {

// Regression oracle only; not part of the production index API.
using CausalMask = std::uint64_t;

struct AdjustmentAnalysis {
  bool has_proper_causal_path = false;
  bool observationally_connected = false;
  bool adjustment_exists = false;
  CausalMask adjustment = 0;
  CausalMask forbidden = 0;
};

// A fully observed causal DAG used by the query-workload planner. Edges are
// supplied as (parent, child) pairs.
class CausalDag {
public:
  CausalDag(std::vector<std::string> variables,
            const std::vector<std::pair<std::string, std::string>> &edges);

  std::size_t size() const { return variables_.size(); }
  const std::vector<std::string> &variables() const { return variables_; }

  CausalMask maskOf(const std::vector<std::string> &variables) const;
  std::vector<std::string> variablesOf(CausalMask mask) const;
  bool reaches(CausalMask sources, CausalMask targets) const;
  // Returns treatments, outcomes, and every variable lying on a proper
  // directed treatment-to-outcome path.
  CausalMask causalPathClosure(CausalMask treatments,
                               CausalMask outcomes) const;
  // Checks one caller-supplied adjustment set against the generalized
  // adjustment criterion. This differs from analyzeAdjustment(), which
  // searches for a minimum valid set.
  bool isValidAdjustment(CausalMask treatments, CausalMask outcomes,
                         CausalMask adjustment) const;
  // Removes observational conditions that are d-separated from the outcome
  // after intervening on treatments. The result is equivalent to the input
  // conditional intervention but may require a smaller materialized scope.
  CausalMask reduceConditioningUnderIntervention(CausalMask treatments,
                                                 CausalMask outcomes,
                                                 CausalMask conditioning) const;

  // Implements the generalized adjustment criterion for a DAG. The returned
  // adjustment set has minimum cardinality; ties follow variable order.
  AdjustmentAnalysis analyzeAdjustment(CausalMask treatments,
                                       CausalMask outcomes) const;

  // Finds a deterministic inclusion-minimal adjustment set whose variables
  // are restricted to allowed_adjustment. This is used by a fixed parent-tree
  // workload: a query is retained only when its complete materialized scope
  // can stay inside that tree.
  AdjustmentAnalysis
  analyzeAdjustmentWithin(CausalMask treatments, CausalMask outcomes,
                          CausalMask allowed_adjustment) const;

private:
  CausalMask allMask() const;
  CausalMask ancestors(CausalMask nodes,
                       const std::vector<CausalMask> &parents) const;
  CausalMask descendants(CausalMask nodes) const;
  CausalMask properCausalNodes(CausalMask treatments,
                               CausalMask outcomes) const;
  std::vector<CausalMask> properBackdoorChildren(CausalMask treatments,
                                                 CausalMask outcomes) const;
  bool dSeparated(CausalMask left, CausalMask right, CausalMask conditioned,
                  const std::vector<CausalMask> &children) const;
  bool validAdjustment(CausalMask treatments, CausalMask outcomes,
                       CausalMask adjustment, CausalMask forbidden,
                       const std::vector<CausalMask> &pbd_children) const;
  std::optional<CausalMask>
  minimumAdjustment(CausalMask treatments, CausalMask outcomes,
                    CausalMask canonical, CausalMask forbidden,
                    const std::vector<CausalMask> &pbd_children) const;

  std::optional<CausalMask>
  inclusionMinimalAdjustment(CausalMask treatments, CausalMask outcomes,
                             CausalMask canonical, CausalMask forbidden,
                             const std::vector<CausalMask> &pbd_children) const;
  std::vector<std::string> variables_;
  std::unordered_map<std::string, std::size_t> index_;
  std::vector<CausalMask> children_;
  std::vector<CausalMask> parents_;
  std::vector<CausalMask> descendants_;
};

} // namespace whatif::testing
