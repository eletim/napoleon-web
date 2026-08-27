#include "napoleon_simulation_runtime.hpp"

#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace napoleon {
namespace {

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point started) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
}

bool is_cpu_agent(const AgentIdentity& agent) {
  return agent.type == AgentType::RuleBased;
}

AgentIdentity resolve_agent(
    const SimulationRuntimeConfig& config,
    const GameState& state,
    int player_index,
    const RosterAssignment& roster) {
  if (config.resolve_agent_identity) {
    return config.resolve_agent_identity(state, player_index, roster);
  }
  return roster.agents[static_cast<std::size_t>(player_index)];
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

Action select_cpu_action(
    const AgentIdentity& agent,
    const GameState& state,
    int player_index,
    std::uint32_t decision_seed) {
  if (agent.type == AgentType::RuleBased &&
      ((state.phase == Phase::Bidding) ||
       (state.phase == Phase::ChoosingAdjutant) ||
       (state.phase == Phase::Exchanging) ||
       (state.phase == Phase::Playing && !state.is_trick_complete))) {
    SeededRandom rng(decision_seed);
    return select_agent_action(agent, state, player_index, rng);
  }

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

void SimulationRuntime::mark_finished(std::size_t game_index) {
  RuntimeGame& game = games_[game_index];
  if (game.status == RuntimeGameStatus::Finished) {
    return;
  }
  game.status = RuntimeGameStatus::Finished;
  if (active_game_count_ == 0) {
    throw std::runtime_error("active game count underflow");
  }
  active_game_count_ -= 1;
  metrics_.finished_games += 1;
  finished_game_indices_.push_back(game_index);
}

SimulationRuntime::SimulationRuntime(SimulationRuntimeConfig config)
    : config_(std::move(config)), started_at_(std::chrono::steady_clock::now()) {
  if (config_.max_concurrent_games == 0) {
    throw std::runtime_error("max_concurrent_games must be positive");
  }
}

SimulationRuntime::~SimulationRuntime() = default;

std::vector<std::uint32_t> SimulationRuntime::add_games(std::size_t count) {
  if (active_game_count_ + count > config_.max_concurrent_games) {
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
    active_game_indices_.push_back(games_.size() - 1);
    active_game_count_ += 1;

    game_ids.push_back(games_.back().game_id);
  }

  metrics_.added_games += count;
  return game_ids;
}

std::vector<std::uint32_t> SimulationRuntime::add_scheduled_games(
    const std::vector<ScheduledGame>& schedule) {
  const auto started = std::chrono::steady_clock::now();
  if (active_game_count_ + schedule.size() > config_.max_concurrent_games) {
    throw std::runtime_error("add_scheduled_games exceeds max_concurrent_games");
  }

  std::vector<std::uint32_t> game_ids;
  game_ids.reserve(schedule.size());
  for (const ScheduledGame& scheduled : schedule) {
    RuntimeGame game;
    game.game_id = next_game_id_++;
    game.game_index = next_game_index_++;
    game.seed = scheduled.seed;
    game.state = create_initial_game(scheduled.seed);
    game.roster = scheduled.roster;
    games_.push_back(std::move(game));
    active_game_indices_.push_back(games_.size() - 1);
    active_game_count_ += 1;

    game_ids.push_back(games_.back().game_id);
  }

  metrics_.added_games += schedule.size();
  metrics_.add_scheduled_games_elapsed_ns += elapsed_ns(started);
  return game_ids;
}

std::size_t SimulationRuntime::advance_runnable_games(std::size_t max_transitions) {
  const auto started = std::chrono::steady_clock::now();
  std::size_t transitions = 0;
  bool progressed = true;

  while (progressed && (max_transitions == 0 || transitions < max_transitions)) {
    progressed = false;
    metrics_.runnable_pass_count += 1;
    bool finished_in_pass = false;

    for (const std::size_t game_index : active_game_indices_) {
      if (max_transitions != 0 && transitions >= max_transitions) {
        break;
      }
      RuntimeGame& game = games_[game_index];
      if (game.status != RuntimeGameStatus::Runnable) {
        continue;
      }

      if (game.state.phase == Phase::Finished || game.state.is_game_over) {
        mark_finished(game_index);
        finished_in_pass = true;
        progressed = true;
        continue;
      }

      if (game.state.is_trick_complete) {
        Action action;
        action.type = Action::Type::AdvanceToNextTrick;
        const auto transition_started = std::chrono::steady_clock::now();
        apply_action(game.state, action);
        metrics_.state_transition_elapsed_ns += elapsed_ns(transition_started);
        game.internal_transition_count += 1;
        metrics_.internal_transition_count += 1;
        transitions += 1;
        progressed = true;
        continue;
      }

      const int player_index = game.state.current_player_index;
      const AgentIdentity agent = resolve_agent(config_, game.state, player_index, game.roster);
      if (!is_cpu_agent(agent)) {
        if (game.pending_request_id.has_value()) {
          throw std::runtime_error("runnable game already has a pending request");
        }

        AgentRequest request;
        request.request_id = next_request_id_++;
        request.game_id = game.game_id;
        request.game_index = game.game_index;
        request.seed = game.seed;
        request.sequence = next_request_sequence_++;
        request.game_decision_count =
            game.internal_transition_count + game.agent_decision_count + 1;
        request.player_index = player_index;
        request.agent = agent;
        request.phase = game.state.phase;
        const auto legal_started = std::chrono::steady_clock::now();
        request.legal_actions = runtime_legal_actions(game.state, player_index);
        metrics_.legal_action_elapsed_ns += elapsed_ns(legal_started);
        if (config_.materialize_agent_request_snapshot) {
          const auto request_build_started = std::chrono::steady_clock::now();
          request.snapshot_json = canonical_snapshot_json(game.state);
          metrics_.request_build_elapsed_ns += elapsed_ns(request_build_started);
        }
        if (request.legal_actions.empty()) {
          throw std::runtime_error("agent request has no legal actions");
        }
        if (config_.build_agent_request_payload) {
          const auto observation_started = std::chrono::steady_clock::now();
          config_.build_agent_request_payload(game.state, player_index, request);
          metrics_.observation_generation_elapsed_ns += elapsed_ns(observation_started);
        }

        game.status = RuntimeGameStatus::WaitingAgent;
        game.pending_request_id = request.request_id;
        pending_request_game_indices_[request.request_id] = game_index;
        pending_requests_.push_back(std::move(request));
        metrics_.agent_request_count += 1;
        progressed = true;
        continue;
      }

      const std::uint32_t decision_seed =
          game.seed ^ static_cast<std::uint32_t>(game.internal_transition_count + 0x9e3779b9u);
      const auto action_started = std::chrono::steady_clock::now();
      const Action action = select_cpu_action(agent, game.state, player_index, decision_seed);
      metrics_.rule_based_action_elapsed_ns += elapsed_ns(action_started);
      const auto transition_started = std::chrono::steady_clock::now();
      apply_action(game.state, action);
      metrics_.state_transition_elapsed_ns += elapsed_ns(transition_started);
      if (game.state.is_game_over) {
        mark_finished(game_index);
        finished_in_pass = true;
      }
      game.internal_transition_count += 1;
      metrics_.internal_transition_count += 1;
      transitions += 1;
      progressed = true;
    }

    if (finished_in_pass) {
      active_game_indices_.erase(
          std::remove_if(
              active_game_indices_.begin(),
              active_game_indices_.end(),
              [&](std::size_t index) {
                return games_[index].status == RuntimeGameStatus::Finished;
              }),
          active_game_indices_.end());
    }
  }

  const auto ended = std::chrono::steady_clock::now();
  const std::uint64_t advance_elapsed = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
  metrics_.cpu_elapsed_ns += advance_elapsed;
  metrics_.advance_runnable_games_elapsed_ns += advance_elapsed;
  return transitions;
}

std::vector<AgentRequest> SimulationRuntime::collect_agent_requests() {
  const auto started = std::chrono::steady_clock::now();
  std::vector<AgentRequest> requests;
  requests.swap(pending_requests_);
  const auto sort_started = std::chrono::steady_clock::now();
  std::sort(requests.begin(), requests.end(), [](const AgentRequest& left, const AgentRequest& right) {
    return left.sequence < right.sequence;
  });
  metrics_.request_sort_elapsed_ns += elapsed_ns(sort_started);
  metrics_.collect_agent_requests_elapsed_ns += elapsed_ns(started);
  return requests;
}

void SimulationRuntime::submit_agent_results(const std::vector<AgentResult>& results) {
  const auto started = std::chrono::steady_clock::now();
  struct ValidatedResult {
    std::size_t game_index = 0;
    AgentResult result;
    GameState next_state;
  };

  std::vector<AgentResult> ordered = results;
  const auto sort_started = std::chrono::steady_clock::now();
  std::sort(ordered.begin(), ordered.end(), [](const AgentResult& left, const AgentResult& right) {
    return left.request_id < right.request_id;
  });
  metrics_.result_sort_elapsed_ns += elapsed_ns(sort_started);

  for (std::size_t index = 1; index < ordered.size(); ++index) {
    if (ordered[index - 1].request_id == ordered[index].request_id) {
      throw std::runtime_error("duplicate request id in agent results");
    }
  }

  std::vector<ValidatedResult> validated;
  validated.reserve(ordered.size());
  for (const AgentResult& result : ordered) {
    const auto lookup_started = std::chrono::steady_clock::now();
    const auto pending_it = pending_request_game_indices_.find(result.request_id);
    metrics_.result_lookup_elapsed_ns += elapsed_ns(lookup_started);
    if (pending_it == pending_request_game_indices_.end()) {
      throw std::runtime_error("unknown or already submitted request id");
    }

    RuntimeGame& game = games_[pending_it->second];
    if (game.status != RuntimeGameStatus::WaitingAgent) {
      throw std::runtime_error("request game is not waiting for an agent");
    }

    try {
      const auto validation_started = std::chrono::steady_clock::now();
      GameState validation_state = game.state;
      apply_action(validation_state, result.action);
      metrics_.result_validation_elapsed_ns += elapsed_ns(validation_started);
      validated.push_back(ValidatedResult{
          pending_it->second,
          result,
          std::move(validation_state)});
    } catch (const std::exception&) {
      throw std::runtime_error("submitted action is not legal for request");
    }
  }

  bool finished_in_results = false;
  for (ValidatedResult& result : validated) {
    const auto commit_started = std::chrono::steady_clock::now();
    RuntimeGame& game = games_[result.game_index];
    game.state = std::move(result.next_state);
    pending_request_game_indices_.erase(result.result.request_id);
    if (game.state.is_game_over) {
      mark_finished(result.game_index);
      finished_in_results = true;
    } else {
      game.status = RuntimeGameStatus::Runnable;
    }
    game.pending_request_id = std::nullopt;
    game.agent_decision_count += 1;
    metrics_.submitted_agent_result_count += 1;
    metrics_.result_commit_elapsed_ns += elapsed_ns(commit_started);
  }

  if (finished_in_results) {
    active_game_indices_.erase(
        std::remove_if(
            active_game_indices_.begin(),
            active_game_indices_.end(),
            [&](std::size_t index) {
              return games_[index].status == RuntimeGameStatus::Finished;
            }),
        active_game_indices_.end());
  }
  metrics_.submit_agent_results_elapsed_ns += elapsed_ns(started);
}

std::vector<FinishedGame> SimulationRuntime::collect_finished_games() {
  const auto started = std::chrono::steady_clock::now();
  std::vector<FinishedGame> finished;
  finished.reserve(finished_game_indices_.size());
  std::sort(finished_game_indices_.begin(), finished_game_indices_.end());
  for (const std::size_t game_index : finished_game_indices_) {
    RuntimeGame& game = games_[game_index];
    if (game.status != RuntimeGameStatus::Finished || game.finished_collected) {
      continue;
    }
    if (!game.state.result.has_value()) {
      throw std::runtime_error("finished game has no result");
    }

    const auto materialize_started = std::chrono::steady_clock::now();
    finished.push_back(FinishedGame{
        game.game_id,
        game.game_index,
        game.seed,
        game.roster,
        *game.state.result,
        game.agent_decision_count,
        game.internal_transition_count,
        canonical_snapshot_json(game.state)});
    metrics_.finished_materialization_elapsed_ns += elapsed_ns(materialize_started);
    game.finished_collected = true;
  }
  finished_game_indices_.clear();
  metrics_.collect_finished_games_elapsed_ns += elapsed_ns(started);
  return finished;
}

std::size_t SimulationRuntime::active_game_count() const {
  return active_game_count_;
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
