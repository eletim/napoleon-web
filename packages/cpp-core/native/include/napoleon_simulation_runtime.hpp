#pragma once

#include "napoleon_core.hpp"
#include "napoleon_roster.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace napoleon {

enum class RuntimeGameStatus : std::uint8_t {
  Runnable,
  WaitingAgent,
  Finished
};

struct AgentRequest;
using AgentRequestPayloadBuilder =
    std::function<void(const GameState& state, int player_index, AgentRequest& request)>;

struct SimulationRuntimeConfig {
  RosterSpec roster;
  std::uint32_t base_seed = 0;
  std::uint32_t roster_seed = 0;
  std::size_t max_concurrent_games = 1024;
  AgentRequestPayloadBuilder build_agent_request_payload;
};

struct RuntimeGameSnapshot {
  std::uint32_t game_id = 0;
  std::uint32_t game_index = 0;
  std::uint32_t seed = 0;
  RuntimeGameStatus status = RuntimeGameStatus::Runnable;
  RosterAssignment roster;
  std::optional<std::uint64_t> pending_request_id;
  std::uint64_t agent_decision_count = 0;
  std::uint64_t internal_transition_count = 0;
};

struct ScheduledGame {
  std::uint32_t seed = 0;
  RosterAssignment roster;
};

struct AgentRequest {
  std::uint64_t request_id = 0;
  std::uint32_t game_id = 0;
  std::uint32_t game_index = 0;
  std::uint64_t sequence = 0;
  int player_index = 0;
  AgentIdentity agent;
  Phase phase = Phase::Bidding;
  std::vector<Action> legal_actions;
  std::string snapshot_json;
  std::vector<float> playing_model_input;
  std::vector<int> legal_play_mask;
};

struct AgentResult {
  std::uint64_t request_id = 0;
  Action action;
  int selected_card_index = -1;
  double behavior_log_probability = 0.0;
  std::string policy_key;
};

struct FinishedGame {
  std::uint32_t game_id = 0;
  std::uint32_t game_index = 0;
  std::uint32_t seed = 0;
  RosterAssignment roster;
  GameResult result;
  std::uint64_t agent_decision_count = 0;
  std::uint64_t internal_transition_count = 0;
  std::string snapshot_json;
};

struct RuntimeMetrics {
  std::uint64_t added_games = 0;
  std::uint64_t finished_games = 0;
  std::uint64_t agent_request_count = 0;
  std::uint64_t submitted_agent_result_count = 0;
  std::uint64_t internal_transition_count = 0;
  std::uint64_t runnable_pass_count = 0;
  std::uint64_t cpu_elapsed_ns = 0;
  double games_per_second = 0.0;
  double decisions_per_second = 0.0;
};

class SimulationRuntime {
 public:
  explicit SimulationRuntime(SimulationRuntimeConfig config);
  ~SimulationRuntime();

  std::vector<std::uint32_t> add_games(std::size_t count);
  std::vector<std::uint32_t> add_scheduled_games(const std::vector<ScheduledGame>& schedule);
  std::size_t advance_runnable_games(std::size_t max_transitions = 0);
  std::vector<AgentRequest> collect_agent_requests();
  void submit_agent_results(const std::vector<AgentResult>& results);
  std::vector<FinishedGame> collect_finished_games();

  RuntimeMetrics metrics() const;
  std::vector<RuntimeGameSnapshot> game_snapshots() const;

 private:
  struct RuntimeGame;

  SimulationRuntimeConfig config_;
  std::vector<RuntimeGame> games_;
  std::vector<AgentRequest> pending_requests_;
  RuntimeMetrics metrics_;
  std::chrono::steady_clock::time_point started_at_;
  std::uint32_t next_game_id_ = 1;
  std::uint32_t next_game_index_ = 0;
  std::uint64_t next_request_id_ = 1;
  std::uint64_t next_request_sequence_ = 1;
};

std::string runtime_game_status_id(RuntimeGameStatus status);

}  // namespace napoleon
