#include "whatif/CausalAdjustment.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>

namespace whatif {
namespace {

std::size_t bitIndex(CausalMask bit) {
  return static_cast<std::size_t>(boost::multiprecision::lsb(bit));
}

} // namespace

CausalDag::CausalDag(
    std::vector<std::string> variables,
    const std::vector<std::pair<std::string, std::string>> &edges)
    : variables_(std::move(variables)) {
  for (std::size_t i = 0; i < variables_.size(); ++i) {
    if (!index_.emplace(variables_[i], i).second) {
      throw std::runtime_error("duplicate causal variable: " + variables_[i]);
    }
  }
  children_.assign(variables_.size(), 0);
  parents_.assign(variables_.size(), 0);
  for (const auto &edge : edges) {
    const auto parent = index_.find(edge.first);
    const auto child = index_.find(edge.second);
    if (parent == index_.end() || child == index_.end()) {
      throw std::runtime_error("causal edge references an unknown variable: " +
                               edge.first + " -> " + edge.second);
    }
    if (parent->second == child->second) {
      throw std::runtime_error("causal DAG contains a self edge: " +
                               edge.first);
    }
    children_[parent->second] |= CausalMask{1} << child->second;
    parents_[child->second] |= CausalMask{1} << parent->second;
  }

  descendants_ = children_;
  for (std::size_t via = 0; via < size(); ++via) {
    const CausalMask via_bit = CausalMask{1} << via;
    for (std::size_t source = 0; source < size(); ++source) {
      if ((descendants_[source] & via_bit) != 0) {
        descendants_[source] |= descendants_[via];
      }
    }
  }
  for (std::size_t i = 0; i < size(); ++i) {
    if ((descendants_[i] & (CausalMask{1} << i)) != 0) {
      throw std::runtime_error("causal graph contains a directed cycle at: " +
                               variables_[i]);
    }
  }
}

CausalMask CausalDag::allMask() const {
  return size() == 0 ? 0 : (CausalMask{1} << size()) - 1;
}

CausalMask CausalDag::maskOf(const std::vector<std::string> &variables) const {
  CausalMask out = 0;
  for (const auto &variable : variables) {
    const auto found = index_.find(variable);
    if (found == index_.end()) {
      throw std::runtime_error("unknown causal variable: " + variable);
    }
    out |= CausalMask{1} << found->second;
  }
  return out;
}

std::vector<std::string> CausalDag::variablesOf(CausalMask mask) const {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < size(); ++i) {
    if ((mask & (CausalMask{1} << i)) != 0) {
      out.push_back(variables_[i]);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool CausalDag::reaches(CausalMask sources, CausalMask targets) const {
  while (sources != 0) {
    const CausalMask bit = sources & (~sources + 1);
    sources ^= bit;
    if ((descendants_[bitIndex(bit)] & targets) != 0) {
      return true;
    }
  }
  return false;
}

CausalMask CausalDag::causalPathClosure(CausalMask treatments,
                                        CausalMask outcomes) const {
  if (treatments == 0 || outcomes == 0 || (treatments & outcomes) != 0) {
    throw std::runtime_error(
        "treatment and outcome sets must be non-empty and disjoint");
  }
  if (((treatments | outcomes) & ~allMask()) != 0) {
    throw std::runtime_error(
        "causal path query contains an out-of-range variable");
  }
  return treatments | outcomes | properCausalNodes(treatments, outcomes);
}

CausalMask CausalDag::ancestors(CausalMask nodes,
                                const std::vector<CausalMask> &parents) const {
  CausalMask result = nodes;
  CausalMask frontier = nodes;
  while (frontier != 0) {
    const CausalMask bit = frontier & (~frontier + 1);
    frontier ^= bit;
    const CausalMask added = parents[bitIndex(bit)] & ~result;
    result |= added;
    frontier |= added;
  }
  return result;
}

CausalMask CausalDag::descendants(CausalMask nodes) const {
  CausalMask result = nodes;
  while (nodes != 0) {
    const CausalMask bit = nodes & (~nodes + 1);
    nodes ^= bit;
    result |= descendants_[bitIndex(bit)];
  }
  return result;
}

CausalMask CausalDag::properCausalNodes(CausalMask treatments,
                                        CausalMask outcomes) const {
  const CausalMask allowed = allMask() & ~treatments;
  CausalMask can_reach_outcome = outcomes;
  CausalMask frontier = outcomes;
  while (frontier != 0) {
    const CausalMask bit = frontier & (~frontier + 1);
    frontier ^= bit;
    const CausalMask added =
        parents_[bitIndex(bit)] & allowed & ~can_reach_outcome;
    can_reach_outcome |= added;
    frontier |= added;
  }

  CausalMask reachable = 0;
  CausalMask treatment_bits = treatments;
  while (treatment_bits != 0) {
    const CausalMask bit = treatment_bits & (~treatment_bits + 1);
    treatment_bits ^= bit;
    reachable |= children_[bitIndex(bit)] & allowed;
  }
  frontier = reachable;
  while (frontier != 0) {
    const CausalMask bit = frontier & (~frontier + 1);
    frontier ^= bit;
    const CausalMask added = children_[bitIndex(bit)] & allowed & ~reachable;
    reachable |= added;
    frontier |= added;
  }
  return reachable & can_reach_outcome;
}

std::vector<CausalMask>
CausalDag::properBackdoorChildren(CausalMask treatments,
                                  CausalMask outcomes) const {
  std::vector<CausalMask> out = children_;
  const CausalMask allowed = allMask() & ~treatments;
  CausalMask can_reach_outcome = outcomes;
  CausalMask frontier = outcomes;
  while (frontier != 0) {
    const CausalMask bit = frontier & (~frontier + 1);
    frontier ^= bit;
    const CausalMask added =
        parents_[bitIndex(bit)] & allowed & ~can_reach_outcome;
    can_reach_outcome |= added;
    frontier |= added;
  }
  while (treatments != 0) {
    const CausalMask bit = treatments & (~treatments + 1);
    treatments ^= bit;
    out[bitIndex(bit)] &= ~can_reach_outcome;
  }
  return out;
}

bool CausalDag::dSeparated(CausalMask left, CausalMask right,
                           CausalMask conditioned,
                           const std::vector<CausalMask> &children) const {
  std::vector<CausalMask> parents(size(), 0);
  for (std::size_t parent = 0; parent < size(); ++parent) {
    CausalMask child_bits = children[parent];
    while (child_bits != 0) {
      const CausalMask bit = child_bits & (~child_bits + 1);
      child_bits ^= bit;
      parents[bitIndex(bit)] |= CausalMask{1} << parent;
    }
  }

  const CausalMask ancestral = ancestors(left | right | conditioned, parents);
  std::vector<CausalMask> moral(size(), 0);
  for (std::size_t child = 0; child < size(); ++child) {
    if ((ancestral & (CausalMask{1} << child)) == 0) {
      continue;
    }
    const CausalMask child_bit = CausalMask{1} << child;
    const CausalMask child_parents = parents[child] & ancestral;
    CausalMask parent_bits = child_parents;
    while (parent_bits != 0) {
      const CausalMask parent_bit = parent_bits & (~parent_bits + 1);
      parent_bits ^= parent_bit;
      const auto parent = bitIndex(parent_bit);
      moral[parent] |= child_bit | (child_parents & ~parent_bit);
      moral[child] |= parent_bit;
    }
  }

  CausalMask visited = left & ancestral & ~conditioned;
  CausalMask queue = visited;
  while (queue != 0) {
    const CausalMask bit = queue & (~queue + 1);
    queue ^= bit;
    if ((bit & right) != 0) {
      return false;
    }
    const CausalMask added =
        moral[bitIndex(bit)] & ancestral & ~conditioned & ~visited;
    visited |= added;
    queue |= added;
  }
  return true;
}

bool CausalDag::validAdjustment(
    CausalMask treatments, CausalMask outcomes, CausalMask adjustment,
    CausalMask forbidden, const std::vector<CausalMask> &pbd_children) const {
  if ((treatments & outcomes) != 0 ||
      (adjustment & (treatments | outcomes | forbidden)) != 0) {
    return false;
  }
  return dSeparated(treatments, outcomes, adjustment, pbd_children);
}

std::optional<CausalMask> CausalDag::minimumAdjustment(
    CausalMask treatments, CausalMask outcomes, CausalMask canonical,
    CausalMask forbidden, const std::vector<CausalMask> &pbd_children) const {
  if (!validAdjustment(treatments, outcomes, canonical, forbidden,
                       pbd_children)) {
    return std::nullopt;
  }
  std::vector<CausalMask> candidates;
  CausalMask bits = canonical;
  while (bits != 0) {
    const CausalMask bit = bits & (~bits + 1);
    bits ^= bit;
    candidates.push_back(bit);
  }
  for (std::size_t wanted = 0; wanted <= candidates.size(); ++wanted) {
    std::optional<CausalMask> result;
    std::function<void(std::size_t, std::size_t, CausalMask)> search =
        [&](std::size_t next, std::size_t remaining, CausalMask chosen) {
          if (result.has_value()) {
            return;
          }
          if (remaining == 0) {
            if (validAdjustment(treatments, outcomes, chosen, forbidden,
                                pbd_children)) {
              result = chosen;
            }
            return;
          }
          if (candidates.size() - next < remaining) {
            return;
          }
          for (std::size_t i = next; i + remaining <= candidates.size(); ++i) {
            search(i + 1, remaining - 1, chosen | candidates[i]);
            if (result.has_value()) {
              return;
            }
          }
        };
    search(0, wanted, 0);
    if (result.has_value()) {
      return result;
    }
  }
  return std::nullopt;
}

bool CausalDag::isValidAdjustment(CausalMask treatments, CausalMask outcomes,
                                  CausalMask adjustment) const {
  if (treatments == 0 || outcomes == 0 || (treatments & outcomes) != 0) {
    throw std::runtime_error(
        "treatment and outcome sets must be non-empty and disjoint");
  }
  if (((treatments | outcomes | adjustment) & ~allMask()) != 0) {
    throw std::runtime_error(
        "adjustment query contains an out-of-range causal variable");
  }
  const CausalMask proper_nodes = properCausalNodes(treatments, outcomes);
  const CausalMask forbidden = descendants(proper_nodes);
  return validAdjustment(treatments, outcomes, adjustment, forbidden,
                         properBackdoorChildren(treatments, outcomes));
}

CausalMask CausalDag::reduceConditioningUnderIntervention(
    CausalMask treatments, CausalMask outcomes, CausalMask conditioning) const {
  if (treatments == 0 || outcomes == 0 || (treatments & outcomes) != 0) {
    throw std::runtime_error(
        "treatment and outcome sets must be non-empty and disjoint");
  }
  if (((treatments | outcomes | conditioning) & ~allMask()) != 0) {
    throw std::runtime_error(
        "conditional intervention contains an out-of-range variable");
  }
  if ((conditioning & (treatments | outcomes)) != 0) {
    throw std::runtime_error(
        "conditioning variables must be disjoint from treatment and outcome");
  }

  auto mutilated_children = children_;
  for (auto &children : mutilated_children) {
    children &= ~treatments;
  }

  CausalMask reduced = conditioning;
  bool changed = true;
  while (changed) {
    changed = false;
    CausalMask candidates = reduced;
    while (candidates != 0) {
      const CausalMask candidate = candidates & (~candidates + 1);
      candidates ^= candidate;
      const CausalMask remaining = reduced & ~candidate;
      if (dSeparated(outcomes, candidate, treatments | remaining,
                     mutilated_children)) {
        reduced = remaining;
        changed = true;
      }
    }
  }
  return reduced;
}

AdjustmentAnalysis CausalDag::analyzeAdjustment(CausalMask treatments,
                                                CausalMask outcomes) const {
  if (treatments == 0 || outcomes == 0 || (treatments & outcomes) != 0) {
    throw std::runtime_error(
        "treatment and outcome sets must be non-empty and disjoint");
  }
  if (((treatments | outcomes) & ~allMask()) != 0) {
    throw std::runtime_error(
        "adjustment query contains an out-of-range causal variable");
  }

  AdjustmentAnalysis result;
  const CausalMask proper_nodes = properCausalNodes(treatments, outcomes);
  result.has_proper_causal_path = proper_nodes != 0;
  result.observationally_connected =
      !dSeparated(treatments, outcomes, 0, children_);
  result.forbidden = descendants(proper_nodes);

  const CausalMask canonical = ancestors(treatments | outcomes, parents_) &
                               ~(treatments | outcomes | result.forbidden);
  const auto pbd_children = properBackdoorChildren(treatments, outcomes);
  const auto adjustment = minimumAdjustment(treatments, outcomes, canonical,
                                            result.forbidden, pbd_children);
  if (adjustment.has_value()) {
    result.adjustment_exists = true;
    result.adjustment = *adjustment;
  }
  return result;
}

std::optional<CausalMask> CausalDag::inclusionMinimalAdjustment(
    CausalMask treatments, CausalMask outcomes, CausalMask canonical,
    CausalMask forbidden, const std::vector<CausalMask> &pbd_children) const {
  if (!validAdjustment(treatments, outcomes, canonical, forbidden,
                       pbd_children)) {
    return std::nullopt;
  }

  CausalMask selected = canonical;
  for (std::size_t index = 0; index < size(); ++index) {
    const CausalMask bit = CausalMask{1} << index;
    if ((selected & bit) == 0) {
      continue;
    }
    const CausalMask candidate = selected & ~bit;
    if (validAdjustment(treatments, outcomes, candidate, forbidden,
                        pbd_children)) {
      selected = candidate;
    }
  }
  return selected;
}

AdjustmentAnalysis
CausalDag::analyzeAdjustmentWithin(CausalMask treatments, CausalMask outcomes,
                                   CausalMask allowed_adjustment) const {
  if (treatments == 0 || outcomes == 0 || (treatments & outcomes) != 0) {
    throw std::runtime_error(
        "treatment and outcome sets must be non-empty and disjoint");
  }
  if (((treatments | outcomes | allowed_adjustment) & ~allMask()) != 0) {
    throw std::runtime_error(
        "adjustment query contains an out-of-range causal variable");
  }

  AdjustmentAnalysis result;
  const CausalMask proper_nodes = properCausalNodes(treatments, outcomes);
  result.has_proper_causal_path = proper_nodes != 0;
  result.observationally_connected =
      !dSeparated(treatments, outcomes, 0, children_);
  result.forbidden = descendants(proper_nodes);

  const CausalMask canonical = ancestors(treatments | outcomes, parents_) &
                               allowed_adjustment &
                               ~(treatments | outcomes | result.forbidden);
  const auto pbd_children = properBackdoorChildren(treatments, outcomes);
  const auto adjustment = inclusionMinimalAdjustment(
      treatments, outcomes, canonical, result.forbidden, pbd_children);
  if (adjustment.has_value()) {
    result.adjustment_exists = true;
    result.adjustment = *adjustment;
  }
  return result;
}

} // namespace whatif
