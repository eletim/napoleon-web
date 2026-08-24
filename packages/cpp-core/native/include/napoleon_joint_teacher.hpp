#pragma once

#include "napoleon_core.hpp"

#include <cstdint>
#include <string>

namespace napoleon::joint_teacher {

constexpr int kAdjutantCompactStateFeatureCount = 237;
constexpr int kAdjutantCompactValueInputFeatureCount = 290;
constexpr int kExchangeCompactStateFeatureCount = 343;
constexpr int kExchangeCompactValueInputFeatureCount = 396;
constexpr int kAdjutantCandidateCount = 53;
constexpr int kExchangeDiscardCombinationCount = 286;

struct JointTeacherOptions {
  std::uint32_t start_seed = 444000000;
  int requested_source_states = 5;
  int max_deal_attempts = 500;
  int exhaustive_state_count = 5;
  int heuristic_top_k = 8;
  std::uint32_t agent_seed = 444;
};

struct JointTeacherReport {
  std::string json;
  int source_state_count = 0;
  int exhaustive_state_count = 0;
  int terminal_rollout_count = 0;
};

JointTeacherReport run_joint_teacher_diagnostic(const JointTeacherOptions& options);

std::string compact290_audit_json();
std::string joint_teacher_definition_json();

}  // namespace napoleon::joint_teacher
