#include "napoleon_simulation_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace napoleon {
namespace {

bool is_cpu_agent(const AgentIdentity& agent) {
  return agent.type == AgentType::RuleBased;
}

std::vector<Action> runtime_legal_actions(const GameState& state, int player_index) {
  std::vector<Action> actions = get_legal_actions(state, player_index);
  if (!actions.empty() || state.phase != Phase::Exchanging ||
      state.current_player_index != player_index || state.is_game_over) {
    return actions;
  }

  const auto& hand = state.hands[static_cast<std::size_t>(player_index)];
  if (hand.size() < 3) {
    return actions;
  }

  for (std::size_t first = 0; first + 2 < hand.size(); ++first) {
    for (std::size_t second = first + 1; second + 1 < hand.size(); ++second) {
      for (std::size_t third = second + 1; third < hand.size(); ++third) {
        Action action;
        action.type = Action::Type::DiscardCards;
        action.player_index = player_index;
        action.cards = {hand[first], hand[second], hand[third]};
        actions.push_back(action);
      }
    }
  }
  return actions;
}

Action select_cpu_action(const GameState& state, int player_index) {
  const std::vector<Action> actions = runtime_legal_actions(state, player_index);
  if (actions.empty()) {
    throw std::runtime_error("runnable game has no legal action");
  }

  return actions.front();
}

std::uint32_t game_seed(std::uint32_t base_seed, std::uint32_t game_index) {
  return base_seed + game_index;
}

}  // namespace

struct SimulationRuntime::RuntimeGame {
  std::uint32_t game_id = 0;
  std::uint32_t game_index = 0;
  std::uint32_t seed = 0;
  GameState state;
  RosterAssignment roster;
  RuntimeGameStatus status = RuntimeGameStatus::Runnable;
  std::optional<std::uint64_t> pending_request_id;
  std::uint64_t agent_decision_count = 0;
  std::uint64_t internal_transition_count = 0;
  bool finished_collected = false;
};

SimulationRuntime::SimulationRuntime(SimulationRuntimeConfig config)
    : config_(std::move(config)), started_at_(std::chrono::steady_clock::now()) {
  if (config_.max_concurrent_games == 0) {
    throw std::runtime_error("max_concurrent_games must be positive");
  }
}

SimulationRuntime::~SimulationRuntime() = default;

std::vector<std::uint32_t> SimulationRuntime::add_games(std::size_t count) {
  const std::size_t active_count = static_cast<std::size_t>(std::count_if(
      games_.begin(), games_.end(), [](const RuntimeGame& game) {
        return game.status != RuntimeGameStatus::Finished;
      }));
  if (active_count + count > config_.max_concurrent_games) {
    throw std::runtime_error("add_games exceeds max_concurrent_games");
  }

  std::vector<std::uint32_t> game_ids;
  game_ids.reserve(count);
  for (std::size_t offset = 0; offset < count; ++offset) {
    const std::uint32_t game_index = next_game_index_++;
    const std::uint32_t seed = game_seed(config_.base_seed, game_index);

    RuntimeGame game;
    game.game_id = next_game_id_++;
    game.game_index = game_index;
    game.seed = seed;
    game.state = create_initial_game(seed);
    game.roster = sample_roster(config_.roster, config_.roster_seed, game_index);
    games_.push_back(std::move(game));

    game_ids.push_back(games_.back().game_id);
  }

  metrics_.added_games += count;
  return game_ids;
}

std::size_t SimulationRuntime::advance_runnable_games(std::size_t max_transitions) {
  const auto started = std::chrono::steady_clock::now();
  std::size_t transitions = 0;
  bool progressed = true;

  while (progressed && (max_transitions == 0 || transitions < max_transitions)) {
    progressed = false;
    metrics_.runnable_pass_count += 1;

    for (RuntimeGame& game : games_) {
      if (max_transitions != 0 && transitions >= max_transitions) {
        break;
      }
      if (game.status != RuntimeGameStatus::Runnable) {
        continue;
      }

      if (game.state.phase == Phase::Finished || game.state.is_game_over) {
        game.status = RuntimeGameStatus::Finished;
        metrics_.finished_games += 1;
        progressed = true;
        continue;
      }

      if (game.state.is_trick_complete) {
        Action action;
        action.type = Action::Type::AdvanceToNextTrick;
        apply_action(game.state, action);
        game.internal_transition_count += 1;
        metrics_.internal_transition_count += 1;
        transitions += 1;
        progressed = true;
        continue;
      }

      const int player_index = game.state.current_player_index;
      const AgentIdentity& agent = game.roster.agents[static_cast<std::size_t>(player_index)];
      if (!is_cpu_agent(agent)) {
        if (game.pending_request_id.has_value()) {
          throw std::runtime_error("runnable game already has a pending request");
        }

        AgentRequest request;
        request.request_id = next_request_id_++;
        request.game_id = game.game_id;
        request.game_index = game.game_index;
        request.sequence = next_request_sequence_++;
        request.player_index = player_index;
        request.agent = agent;
        request.phase = game.state.phase;
        request.legal_actions = runtime_legal_actions(game.state, player_index);
        request.snapshot_json = canonical_snapshot_json(game.state);
        if (request.legal_actions.empty()) {
          throw std::runtime_error("agent request has no legal actions");
        }

        game.status = RuntimeGameStatus::WaitingAgent;
        game.pending_request_id = request.request_id;
        pending_requests_.push_back(std::move(request));
        metrics_.agent_request_count += 1;
        progressed = true;
        continue;
      }

      const Action action = select_cpu_action(game.state, player_index);
      apply_action(game.state, action);
      if (game.state.is_game_over) {
        game.status = RuntimeGameStatus::Finished;
        metrics_.finished_games += 1;
      }
      game.internal_transition_count += 1;
      metrics_.internal_transition_count += 1;
      transitions += 1;
      progressed = true;
    }
  }

  const auto ended = std::chrono::steady_clock::now();
  metrics_.cpu_elapsed_ns += static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
  return transitions;
}

std::vector<AgentRequest> SimulationRuntime::collect_agent_requests() {
  std::vector<AgentRequest> requests;
  requests.swap(pending_requests_);
  std::sort(requests.begin(), requests.end(), [](const AgentRequest& left, const AgentRequest& right) {
    return left.sequence < right.sequence;
  });
  return requests;
}

void SimulationRuntime::submit_agent_results(const std::vector<AgentResult>& results) {
  std::vector<AgentResult> ordered = results;
  std::sort(ordered.begin(), ordered.end(), [](const AgentResult& left, const AgentResult& right) {
    return left.request_id < right.request_id;
  });

  for (const AgentResult& result : ordered) {
    auto game_it = std::find_if(games_.begin(), games_.end(), [&](const RuntimeGame& game) {
      return game.pending_request_id.has_value() && *game.pending_request_id == result.request_id;
    });
    if (game_it == games_.end()) {
      throw std::runtime_error("unknown or already submitted request id");
    }

    RuntimeGame& game = *game_it;
    if (game.status != RuntimeGameStatus::WaitingAgent) {
      throw std::runtime_error("request game is not waiting for an agent");
    }

    try {
      GameState validation_state = game.state;
      apply_action(validation_state, result.action);
    } catch (const std::exception&) {
      throw std::runtime_error("submitted action is not legal for request");
    }

    apply_action(game.state, result.action);
    pending_requests_.erase(
        std::remove_if(
            pending_requests_.begin(),
            pending_requests_.end(),
            [&](const AgentRequest& request) {
              return request.request_id == result.request_id;
            }),
        pending_requests_.end());
    if (game.state.is_game_over) {
      game.status = RuntimeGameStatus::Finished;
      metrics_.finished_games += 1;
    } else {
      game.status = RuntimeGameStatus::Runnable;
    }
    game.pending_request_id = std::nullopt;
    game.agent_decision_count += 1;
    metrics_.submitted_agent_result_count += 1;
  }
}

std::vector<FinishedGame> SimulationRuntime::collect_finished_games() {
  std::vector<FinishedGame> finished;
  for (RuntimeGame& game : games_) {
    if (game.status != RuntimeGameStatus::Finished || game.finished_collected) {
      continue;
    }
    if (!game.state.result.has_value()) {
      throw std::runtime_error("finished game has no result");
    }

    finished.push_back(FinishedGame{
        game.game_id,
        game.game_index,
        game.seed,
        game.roster,
        *game.state.result,
        game.agent_decision_count,
        game.internal_transition_count,
        canonical_snapshot_json(game.state)});
    game.finished_collected = true;
  }
  return finished;
}

RuntimeMetrics SimulationRuntime::metrics() const {
  RuntimeMetrics snapshot = metrics_;
  const auto now = std::chrono::steady_clock::now();
  const double elapsed_seconds =
      std::max(1e-9, std::chrono::duration<double>(now - started_at_).count());
  snapshot.games_per_second = static_cast<double>(snapshot.finished_games) / elapsed_seconds;
  snapshot.decisions_per_second =
      static_cast<double>(snapshot.internal_transition_count +
                          snapshot.submitted_agent_result_count) /
      elapsed_seconds;
  return snapshot;
}

std::vector<RuntimeGameSnapshot> SimulationRuntime::game_snapshots() const {
  std::vector<RuntimeGameSnapshot> snapshots;
  snapshots.reserve(games_.size());
  for (const RuntimeGame& game : games_) {
    snapshots.push_back(RuntimeGameSnapshot{
        game.game_id,
        game.game_index,
        game.seed,
        game.status,
        game.roster,
        game.pending_request_id,
        game.agent_decision_count,
        game.internal_transition_count});
  }
  return snapshots;
}

std::string runtime_game_status_id(RuntimeGameStatus status) {
  switch (status) {
    case RuntimeGameStatus::Runnable:
      return "RUNNABLE";
    case RuntimeGameStatus::WaitingAgent:
      return "WAITING_AGENT";
    case RuntimeGameStatus::Finished:
      return "FINISHED";
  }

  throw std::runtime_error("invalid runtime game status");
}

}  // namespace napoleon
