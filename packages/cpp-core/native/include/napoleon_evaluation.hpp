#pragma once

#include "napoleon_simulation_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace napoleon::evaluation {

enum class EvaluationScenario : std::uint8_t {
  CandidateVsRuleBased,
  CandidateVsFrozen,
  CandidateVsOpponentPool,
  Tournament
};

struct EvaluationOptions {
  EvaluationScenario scenario = EvaluationScenario::CandidateVsRuleBased;
  std::uint32_t start_seed = 0;
  std::uint32_t seed_count = 1;
  std::uint32_t roster_seed = 0;
  std::size_t max_concurrent_games = 256;
  std::size_t inference_max_batch_size = 32;
  std::vector<std::uint32_t> rotation_offsets = {0, 1, 2, 3, 4};
  std::string candidate_id = "candidate";
  std::string frozen_id = "rl-v740";
};

struct EvaluationArtifact {
  std::string json;
  std::uint32_t scheduled_games = 0;
  std::uint32_t completed_games = 0;
  std::uint32_t failed_games = 0;
};

EvaluationArtifact run_evaluation(const EvaluationOptions& options);
std::string evaluation_scenario_id(EvaluationScenario scenario);

}  // namespace napoleon::evaluation
