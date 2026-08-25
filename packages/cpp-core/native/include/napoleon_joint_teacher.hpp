#pragma once

#include "napoleon_core.hpp"

#include <cstdint>
#include <iosfwd>
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

struct AdjutantValueStreamOptions {
  std::string mode = "proposal";
  std::string output_directory;
  std::string bidding_policy_id = "frozen-raise-v1";
  std::string bidding_margin_onnx_path =
      "benchmarks/bidding-margin-policies/frozen-raise-v1/margin.onnx";
  std::string playing_policy_id = "ppo-separated-v1000";
  std::string playing_policy_onnx_path =
      "benchmarks/playing-policies/ppo-separated-v1000/policy.onnx";
  std::string playing_critic_onnx_path =
      "benchmarks/playing-policies/ppo-separated-v1000/critic.onnx";
  std::string policy_device = "cpu";
  std::uint32_t start_seed = 446000000;
  int requested_source_states = 10000;
  int max_deal_attempts = 250000;
  int proposal_top_k = 16;
  int diversity_count = 8;
  int scorer_top_k = 64;
  std::uint32_t agent_seed = 446;
};

struct AdjutantValueStreamReport {
  int source_state_count = 0;
  int sample_count = 0;
  int terminal_rollout_count = 0;
  std::string manifest_json;
};

JointTeacherReport run_joint_teacher_diagnostic(const JointTeacherOptions& options);
AdjutantValueStreamReport run_adjutant_value_stream_teacher(
    const AdjutantValueStreamOptions& options,
    std::istream& scorer_response,
    std::ostream& scorer_request);

std::string compact290_audit_json();
std::string joint_teacher_definition_json();

}  // namespace napoleon::joint_teacher
