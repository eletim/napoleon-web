#include "napoleon_evaluation.hpp"
#include "napoleon_onnx_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <map>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace napoleon::evaluation {
namespace {

constexpr int kScheduledBatchSize = 64;

struct MutableStats {
  std::uint64_t games = 0;
  std::uint64_t wins = 0;
  std::uint64_t losses = 0;
  std::uint64_t contract_successes = 0;
  std::uint64_t point_cards_total = 0;
};

struct PolicyInferenceStats {
  std::string policy_id;
  std::string policy_type;
  std::uint64_t request_count = 0;
  std::uint64_t session_run_count = 0;
  std::uint64_t batch_item_total = 0;
  std::uint64_t max_batch = 0;
  std::uint64_t elapsed_ns = 0;
};

struct CompletedRecord {
  FinishedGame game;
  std::uint32_t evaluation_seed = 0;
  std::uint32_t rotation_offset = 0;
};

struct DriverMetrics {
  RuntimeMetrics runtime;
  std::uint64_t total_elapsed_ns = 0;
  std::uint64_t inference_elapsed_ns = 0;
  std::uint64_t serialization_elapsed_ns = 0;
  std::uint64_t inference_request_count = 0;
  std::uint64_t session_run_count = 0;
  std::uint64_t mean_batch_numerator = 0;
  std::uint64_t max_batch = 0;
  std::map<std::string, PolicyInferenceStats> per_policy;
};

struct EvaluationRun {
  std::vector<CompletedRecord> completed;
  DriverMetrics metrics;
};

void json_escape(std::ostream& out, const std::string& value) {
  out << '"';
  for (char ch : value) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      default:
        out << ch;
        break;
    }
  }
  out << '"';
}

std::string agent_key(const AgentIdentity& agent) {
  return agent_type_id(agent.type) + ":" + agent.id;
}

bool same_agent(const AgentIdentity& left, const AgentIdentity& right) {
  return left.type == right.type && left.id == right.id;
}

std::string winning_team_for_seat(const GameResult& result, int seat_index) {
  if (result.napoleon_player_index == seat_index ||
      (result.adjutant_player_index.has_value() && *result.adjutant_player_index == seat_index)) {
    return "napoleon-team";
  }
  return "alliance";
}

int point_cards_for_seat(const GameResult& result, int seat_index) {
  return winning_team_for_seat(result, seat_index) == "napoleon-team"
             ? result.napoleon_team_point_cards
             : result.alliance_point_cards;
}

void count_seat(MutableStats& stats, const GameResult& result, int seat_index) {
  stats.games += 1;
  const bool won = winning_team_for_seat(result, seat_index) == result.winner;
  stats.wins += won ? 1 : 0;
  stats.losses += won ? 0 : 1;
  stats.contract_successes += result.winner == "napoleon-team" ? 1 : 0;
  stats.point_cards_total += static_cast<std::uint64_t>(point_cards_for_seat(result, seat_index));
}

Action deterministic_policy_action(const AgentRequest& request) {
  if (request.legal_actions.empty()) {
    throw std::runtime_error("policy request has no legal actions");
  }

  if (request.phase != Phase::Playing) {
    return request.legal_actions.front();
  }

  std::uint32_t hash = 2166136261u;
  for (char ch : request.agent.id) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 16777619u;
  }
  hash ^= static_cast<std::uint32_t>(request.sequence);
  hash *= 16777619u;
  hash ^= request.game_index;

  return request.legal_actions[hash % request.legal_actions.size()];
}

std::vector<RosterAssignment> fixed_rotations(
    const std::array<AgentIdentity, kPlayerCount>& source_agents,
    const std::vector<std::uint32_t>& rotation_offsets) {
  std::vector<RosterAssignment> rotations;
  rotations.reserve(rotation_offsets.size());
  for (std::uint32_t offset : rotation_offsets) {
    RosterAssignment assignment;
    assignment.current_seat_index = static_cast<int>(offset % kPlayerCount);
    for (int seat = 0; seat < kPlayerCount; ++seat) {
      const int source_index =
          (seat + kPlayerCount - static_cast<int>(offset % kPlayerCount)) % kPlayerCount;
      assignment.agents[static_cast<std::size_t>(seat)] =
          source_agents[static_cast<std::size_t>(source_index)];
    }
    rotations.push_back(std::move(assignment));
  }
  return rotations;
}

std::array<AgentIdentity, kPlayerCount> agents_for_scenario(const EvaluationOptions& options) {
  const AgentIdentity candidate = current_policy_agent(options.candidate_id);
  const AgentIdentity rule = rule_based_agent("RuleBasedAgent");
  const AgentIdentity frozen = frozen_policy_agent(options.frozen_id);

  if (options.scenario == EvaluationScenario::CandidateVsRuleBased ||
      options.scenario == EvaluationScenario::CandidateVsOpponentPool) {
    return {candidate, rule, rule, rule, rule};
  }
  if (options.scenario == EvaluationScenario::CandidateVsFrozen) {
    return {candidate, frozen, frozen, frozen, frozen};
  }

  return {
      candidate,
      frozen,
      rule_based_agent("RuleBasedAgent-A"),
      frozen_policy_agent(options.frozen_id + "-alt"),
      rule_based_agent("RuleBasedAgent-B")};
}

RosterAssignment opponent_pool_assignment(
    const EvaluationOptions& options,
    std::uint32_t index,
    std::uint32_t rotation_offset) {
  const RosterSpec spec = current_plus_opponent_pool_roster(
      current_policy_agent(options.candidate_id),
      {
          WeightedAgent{rule_based_agent("RuleBasedAgent"), 1},
          WeightedAgent{frozen_policy_agent(options.frozen_id), 1},
      },
      false,
      static_cast<int>(rotation_offset % kPlayerCount));
  return sample_roster(spec, options.roster_seed, index);
}

std::vector<ScheduledGame> create_schedule(const EvaluationOptions& options) {
  std::vector<ScheduledGame> schedule;
  schedule.reserve(static_cast<std::size_t>(options.seed_count) * options.rotation_offsets.size());

  const std::array<AgentIdentity, kPlayerCount> base_agents = agents_for_scenario(options);
  const std::vector<RosterAssignment> rotations = fixed_rotations(base_agents, options.rotation_offsets);
  std::uint32_t evaluation_index = 0;
  for (std::uint32_t seed_offset = 0; seed_offset < options.seed_count; ++seed_offset) {
    const std::uint32_t seed = options.start_seed + seed_offset;
    for (std::size_t rotation_index = 0; rotation_index < rotations.size(); ++rotation_index) {
      ScheduledGame scheduled;
      scheduled.seed = seed;
      scheduled.roster = options.scenario == EvaluationScenario::CandidateVsOpponentPool
                             ? opponent_pool_assignment(
                                   options,
                                   evaluation_index,
                                   options.rotation_offsets[rotation_index])
                             : rotations[rotation_index];
      schedule.push_back(std::move(scheduled));
      evaluation_index += 1;
    }
  }
  return schedule;
}

void submit_policy_requests(
    SimulationRuntime& runtime,
    const std::vector<AgentRequest>& requests,
    std::size_t max_batch,
    DriverMetrics& metrics,
    onnx_policy::BatchedPolicyExecutor& executor) {
  std::map<std::string, std::vector<AgentRequest>> grouped;
  for (const AgentRequest& request : requests) {
    grouped[agent_key(request.agent)].push_back(request);
  }

  for (const auto& [key, policy_requests] : grouped) {
    std::size_t offset = 0;
    while (offset < policy_requests.size()) {
      const std::size_t batch_size = std::min(max_batch, policy_requests.size() - offset);
      const auto started = std::chrono::steady_clock::now();

      std::vector<AgentResult> results;
      results.reserve(batch_size);
      std::vector<AgentRequest> playing_requests;
      playing_requests.reserve(batch_size);
      for (std::size_t index = 0; index < batch_size; ++index) {
        const AgentRequest& request = policy_requests[offset + index];
        if (request.phase == Phase::Playing) {
          playing_requests.push_back(request);
        } else {
          AgentResult result;
          result.request_id = request.request_id;
          result.action = deterministic_policy_action(request);
          results.push_back(result);
        }
      }
      if (!playing_requests.empty()) {
        std::vector<onnx_policy::PolicyActionResult> policy_results =
            executor.run(playing_requests);
        if (policy_results.size() != playing_requests.size()) {
          throw std::runtime_error("policy executor returned a mismatched result count");
        }
        for (const onnx_policy::PolicyActionResult& result : policy_results) {
          results.push_back(result.result);
        }
      }
      runtime.submit_agent_results(results);

      const auto ended = std::chrono::steady_clock::now();
      const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
      const AgentRequest& first = policy_requests[offset];
      PolicyInferenceStats& stats = metrics.per_policy[key];
      stats.policy_id = first.agent.id;
      stats.policy_type = agent_type_id(first.agent.type);
      stats.request_count += batch_size;
      stats.session_run_count += 1;
      stats.batch_item_total += batch_size;
      stats.max_batch = std::max<std::uint64_t>(stats.max_batch, batch_size);
      stats.elapsed_ns += elapsed_ns;

      metrics.inference_request_count += batch_size;
      metrics.session_run_count += 1;
      metrics.mean_batch_numerator += batch_size;
      metrics.max_batch = std::max<std::uint64_t>(metrics.max_batch, batch_size);
      metrics.inference_elapsed_ns += elapsed_ns;
      offset += batch_size;
    }
  }
}

onnx_policy::InferenceDevice policy_inference_device(const EvaluationOptions& options) {
  return options.inference_device == "cuda" ? onnx_policy::InferenceDevice::Cuda
                                            : onnx_policy::InferenceDevice::Cpu;
}

std::unique_ptr<onnx_policy::PolicySession> create_policy_session(
    const EvaluationOptions& options,
    onnx_policy::PolicyKey key,
    const std::string& onnx_path) {
  if (options.policy_backend == "onnx") {
    return onnx_policy::create_onnxruntime_policy_session(onnx_policy::PolicySessionConfig{
        key,
        onnx_path,
        "model_input",
        "logits",
        policy_inference_device(options)});
  }
  return std::make_unique<onnx_policy::DeterministicPolicySession>(
      onnx_policy::DeterministicPolicySession::default_logits(),
      options.inference_device == "cuda" ? onnx_policy::ExecutionProvider::Cuda
                                         : onnx_policy::ExecutionProvider::Cpu);
}

std::unique_ptr<onnx_policy::BatchedPolicyExecutor> create_policy_executor(
    const EvaluationOptions& options) {
  auto executor = std::make_unique<onnx_policy::BatchedPolicyExecutor>(
      onnx_policy::BatchedPolicyConfig{
          std::max<std::size_t>(1, options.inference_max_batch_size),
          1.0,
          options.roster_seed});
  const onnx_policy::PolicyKey candidate_key{AgentType::CurrentPolicy, options.candidate_id};
  const onnx_policy::PolicyKey frozen_key{AgentType::FrozenPolicy, options.frozen_id};
  const onnx_policy::PolicyKey frozen_alt_key{AgentType::FrozenPolicy, options.frozen_id + "-alt"};
  executor->add_policy(
      candidate_key,
      create_policy_session(options, candidate_key, options.candidate_onnx_path));
  executor->add_policy(
      frozen_key,
      create_policy_session(options, frozen_key, options.frozen_onnx_path));
  executor->add_policy(
      frozen_alt_key,
      create_policy_session(options, frozen_alt_key, options.frozen_onnx_path));
  return executor;
}

void copy_executor_stats(
    const onnx_policy::BatchedPolicyStats& stats,
    DriverMetrics& metrics) {
  metrics.inference_request_count = stats.request_count;
  metrics.session_run_count = stats.session_run_count;
  metrics.mean_batch_numerator = stats.request_count;
  metrics.max_batch = stats.max_observed_batch_size;
  metrics.inference_elapsed_ns = stats.inference_elapsed_ns;
  metrics.per_policy.clear();
  for (const auto& [key, policy_stats] : stats.policy_stats) {
    PolicyInferenceStats out;
    out.policy_id = key;
    out.policy_type = key.substr(0, key.find(':'));
    out.request_count = policy_stats.request_count;
    out.session_run_count = policy_stats.session_run_count;
    out.batch_item_total = policy_stats.request_count;
    out.max_batch = policy_stats.max_observed_batch_size;
    out.elapsed_ns = policy_stats.inference_elapsed_ns;
    metrics.per_policy[key] = out;
  }
}

EvaluationRun drive_schedule(const EvaluationOptions& options, const std::vector<ScheduledGame>& schedule) {
  const auto total_started = std::chrono::steady_clock::now();
  SimulationRuntime runtime(SimulationRuntimeConfig{
      fixed_roster({rule_based_agent(), rule_based_agent(), rule_based_agent(), rule_based_agent(), rule_based_agent()}),
      options.start_seed,
      options.roster_seed,
      options.max_concurrent_games,
      onnx_policy::attach_playing_model_input});
  auto executor = create_policy_executor(options);

  EvaluationRun run;
  std::size_t next_schedule_index = 0;
  while (run.completed.size() < schedule.size()) {
    const std::size_t in_flight = runtime.game_snapshots().size() - run.completed.size();
    if (next_schedule_index < schedule.size() && in_flight < options.max_concurrent_games) {
      const std::size_t open_slots = options.max_concurrent_games - in_flight;
      const std::size_t batch_count =
          std::min({open_slots, static_cast<std::size_t>(kScheduledBatchSize), schedule.size() - next_schedule_index});
      std::vector<ScheduledGame> batch(
          schedule.begin() + static_cast<std::ptrdiff_t>(next_schedule_index),
          schedule.begin() + static_cast<std::ptrdiff_t>(next_schedule_index + batch_count));
      runtime.add_scheduled_games(batch);
      next_schedule_index += batch_count;
    }

    runtime.advance_runnable_games();
    std::vector<AgentRequest> requests = runtime.collect_agent_requests();
    if (!requests.empty()) {
      submit_policy_requests(
          runtime,
          requests,
          std::max<std::size_t>(1, options.inference_max_batch_size),
          run.metrics,
          *executor);
    }

    for (FinishedGame& finished : runtime.collect_finished_games()) {
      const std::uint32_t evaluation_seed =
          options.start_seed + finished.game_index / static_cast<std::uint32_t>(options.rotation_offsets.size());
      const std::uint32_t rotation_offset =
          options.rotation_offsets[finished.game_index % options.rotation_offsets.size()];
      run.completed.push_back(CompletedRecord{std::move(finished), evaluation_seed, rotation_offset});
    }
  }

  copy_executor_stats(executor->stats(), run.metrics);
  run.metrics.runtime = runtime.metrics();
  const auto total_ended = std::chrono::steady_clock::now();
  run.metrics.total_elapsed_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(total_ended - total_started).count());
  return run;
}

double seconds(std::uint64_t ns) {
  return static_cast<double>(ns) / 1000000000.0;
}

void write_rate(std::ostream& out, std::uint64_t numerator, std::uint64_t denominator) {
  out << "{\"numerator\":" << numerator
      << ",\"denominator\":" << denominator
      << ",\"rate\":";
  if (denominator == 0) {
    out << "null";
  } else {
    out << static_cast<double>(numerator) / static_cast<double>(denominator);
  }
  out << '}';
}

void write_stats(std::ostream& out, const MutableStats& stats) {
  out << "{\"games\":" << stats.games
      << ",\"wins\":" << stats.wins
      << ",\"losses\":" << stats.losses
      << ",\"winRate\":";
  write_rate(out, stats.wins, stats.games);
  out << ",\"contractSuccesses\":" << stats.contract_successes
      << ",\"contractSuccessRate\":";
  write_rate(out, stats.contract_successes, stats.games);
  out << ",\"averagePointCards\":";
  if (stats.games == 0) {
    out << "null";
  } else {
    out << static_cast<double>(stats.point_cards_total) / static_cast<double>(stats.games);
  }
  out << '}';
}

std::map<std::string, MutableStats> collect_agent_stats(const std::vector<CompletedRecord>& records) {
  std::map<std::string, MutableStats> stats;
  for (const CompletedRecord& record : records) {
    for (int seat = 0; seat < kPlayerCount; ++seat) {
      count_seat(stats[agent_key(record.game.roster.agents[static_cast<std::size_t>(seat)])],
                 record.game.result,
                 seat);
    }
  }
  return stats;
}

MutableStats collect_candidate_stats(
    const std::vector<CompletedRecord>& records,
    const AgentIdentity& candidate) {
  MutableStats stats;
  for (const CompletedRecord& record : records) {
    for (int seat = 0; seat < kPlayerCount; ++seat) {
      if (same_agent(record.game.roster.agents[static_cast<std::size_t>(seat)], candidate)) {
        count_seat(stats, record.game.result, seat);
      }
    }
  }
  return stats;
}

void write_roster_assignment(std::ostream& out, const RosterAssignment& roster) {
  out << roster_assignment_manifest_json(roster);
}

void write_completed_games(std::ostream& out, const std::vector<CompletedRecord>& records) {
  out << '[';
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const CompletedRecord& record = records[index];
    const GameResult& result = record.game.result;
    out << "{\"schemaVersion\":1,\"status\":\"completed\""
        << ",\"gameIndex\":" << record.game.game_index
        << ",\"seed\":" << record.evaluation_seed
        << ",\"runtimeSeed\":" << record.game.seed
        << ",\"rotationOffset\":" << record.rotation_offset
        << ",\"roster\":";
    write_roster_assignment(out, record.game.roster);
    out << ",\"contract\":{\"napoleonPlayerId\":\"player-" << result.napoleon_player_index
        << "\",\"targetPointCards\":" << result.target_point_cards
        << ",\"adjutantPlayerId\":";
    if (result.adjutant_player_index.has_value()) {
      out << "\"player-" << *result.adjutant_player_index << "\"";
    } else {
      out << "null";
    }
    out << "},\"pointCards\":{\"napoleonTeam\":" << result.napoleon_team_point_cards
        << ",\"alliance\":" << result.alliance_point_cards
        << "},\"winner\":";
    json_escape(out, result.winner);
    out << ",\"contractSucceeded\":" << (result.winner == "napoleon-team" ? "true" : "false")
        << '}';
  }
  out << ']';
}

void write_metrics(std::ostream& out, const DriverMetrics& metrics) {
  const double elapsed = std::max(1e-9, seconds(metrics.total_elapsed_ns));
  out << "{\"totalElapsedSeconds\":" << seconds(metrics.total_elapsed_ns)
      << ",\"simulationCpuElapsedSeconds\":" << seconds(metrics.runtime.cpu_elapsed_ns)
      << ",\"inferenceElapsedSeconds\":" << seconds(metrics.inference_elapsed_ns)
      << ",\"serializationWriteElapsedSeconds\":" << seconds(metrics.serialization_elapsed_ns)
      << ",\"gamesPerSecond\":" << static_cast<double>(metrics.runtime.finished_games) / elapsed
      << ",\"decisionsPerSecond\":"
      << static_cast<double>(metrics.runtime.internal_transition_count +
                             metrics.runtime.submitted_agent_result_count) /
             elapsed
      << ",\"requestCount\":" << metrics.inference_request_count
      << ",\"sessionRunCount\":" << metrics.session_run_count
      << ",\"meanBatchSize\":";
  if (metrics.session_run_count == 0) {
    out << "null";
  } else {
    out << static_cast<double>(metrics.mean_batch_numerator) /
               static_cast<double>(metrics.session_run_count);
  }
  out << ",\"maxBatchSize\":" << metrics.max_batch
      << ",\"runtime\":{\"addedGames\":" << metrics.runtime.added_games
      << ",\"finishedGames\":" << metrics.runtime.finished_games
      << ",\"agentRequestCount\":" << metrics.runtime.agent_request_count
      << ",\"submittedAgentResultCount\":" << metrics.runtime.submitted_agent_result_count
      << ",\"internalTransitionCount\":" << metrics.runtime.internal_transition_count
      << ",\"runnablePassCount\":" << metrics.runtime.runnable_pass_count
      << "},\"policyStats\":[";
  bool first = true;
  for (const auto& [key, stats] : metrics.per_policy) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << "{\"key\":";
    json_escape(out, key);
    out << ",\"policyId\":";
    json_escape(out, stats.policy_id);
    out << ",\"policyType\":";
    json_escape(out, stats.policy_type);
    out << ",\"requestCount\":" << stats.request_count
        << ",\"sessionRunCount\":" << stats.session_run_count
        << ",\"meanBatchSize\":";
    if (stats.session_run_count == 0) {
      out << "null";
    } else {
      out << static_cast<double>(stats.batch_item_total) /
                 static_cast<double>(stats.session_run_count);
    }
    out << ",\"maxBatchSize\":" << stats.max_batch
        << ",\"inferenceElapsedSeconds\":" << seconds(stats.elapsed_ns)
        << '}';
  }
  out << "]}";
}

std::string create_artifact_json(const EvaluationOptions& options, const EvaluationRun& run) {
  const auto serialization_started = std::chrono::steady_clock::now();
  std::vector<CompletedRecord> completed = run.completed;
  std::sort(completed.begin(), completed.end(), [](const CompletedRecord& left, const CompletedRecord& right) {
    return left.game.game_index < right.game.game_index;
  });

  const std::map<std::string, MutableStats> agent_stats = collect_agent_stats(completed);
  const MutableStats candidate_stats =
      collect_candidate_stats(completed, current_policy_agent(options.candidate_id));

  std::ostringstream out;
  out.precision(12);
  out << "{\"schemaVersion\":1,\"application\":\"cpp-evaluation-benchmark-tournament\""
      << ",\"scenario\":";
  json_escape(out, evaluation_scenario_id(options.scenario));
  out << ",\"configuration\":{\"startSeed\":" << options.start_seed
      << ",\"endSeed\":" << (options.start_seed + options.seed_count - 1)
      << ",\"seedCount\":" << options.seed_count
      << ",\"rotationOffsets\":[";
  for (std::size_t index = 0; index < options.rotation_offsets.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << options.rotation_offsets[index];
  }
  out << "],\"rosterSeed\":" << options.roster_seed
      << ",\"maxConcurrentGames\":" << options.max_concurrent_games
      << ",\"inferenceMaxBatchSize\":" << options.inference_max_batch_size
      << ",\"candidateId\":";
  json_escape(out, options.candidate_id);
  out << ",\"frozenPolicyId\":";
  json_escape(out, options.frozen_id);
  out << ",\"candidateRuntime\":{\"requestedInferenceDevice\":";
  json_escape(out, options.inference_device);
  out << ",\"resolvedInferenceDevice\":";
  json_escape(out, options.inference_device == "cuda" ? "cuda" : "cpu");
  out << ",\"executionProvider\":";
  json_escape(out, options.inference_device == "cuda" ? "cuda" : "cpu");
  out << ",\"policyBackend\":";
  json_escape(out, options.policy_backend);
  out << "}";
  out << ",\"frozenRuntime\":{\"requestedInferenceDevice\":";
  json_escape(out, options.inference_device);
  out << ",\"resolvedInferenceDevice\":";
  json_escape(out, options.inference_device == "cuda" ? "cuda" : "cpu");
  out << ",\"executionProvider\":";
  json_escape(out, options.inference_device == "cuda" ? "cuda" : "cpu");
  out << ",\"policyBackend\":";
  json_escape(out, options.policy_backend);
  out << "}";
  out << ",\"baseline\":{\"tsCudaBatch1Workers4SecondsPer2000Games\":11.2"
      << ",\"tsBatchedPathSecondsPer2000Games\":{\"min\":18,\"max\":22}}"
      << ",\"usesRlDatasetGeneration\":false"
      << ",\"browserOrWebRuntimeIntegration\":false}"
      << ",\"summary\":{\"scheduledGames\":" << run.completed.size()
      << ",\"completedGames\":" << run.completed.size()
      << ",\"failedGames\":0,\"candidate\":";
  write_stats(out, candidate_stats);
  out << ",\"agents\":[";
  bool first = true;
  for (const auto& [key, stats] : agent_stats) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << "{\"key\":";
    json_escape(out, key);
    out << ",\"stats\":";
    write_stats(out, stats);
    out << '}';
  }
  out << "]},\"metrics\":";
  DriverMetrics metrics = run.metrics;
  const auto serialization_ended = std::chrono::steady_clock::now();
  metrics.serialization_elapsed_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          serialization_ended - serialization_started)
          .count());
  write_metrics(out, metrics);
  out << ",\"games\":";
  write_completed_games(out, completed);
  out << '}';
  return out.str();
}

}  // namespace

EvaluationArtifact run_evaluation(const EvaluationOptions& options) {
  if (options.seed_count == 0) {
    throw std::runtime_error("seed_count must be positive");
  }
  if (options.rotation_offsets.empty()) {
    throw std::runtime_error("rotation_offsets must not be empty");
  }
  if (options.max_concurrent_games == 0) {
    throw std::runtime_error("max_concurrent_games must be positive");
  }
  if (options.inference_max_batch_size == 0) {
    throw std::runtime_error("inference_max_batch_size must be positive");
  }

  const std::vector<ScheduledGame> schedule = create_schedule(options);
  const EvaluationRun run = drive_schedule(options, schedule);
  return EvaluationArtifact{
      create_artifact_json(options, run),
      static_cast<std::uint32_t>(schedule.size()),
      static_cast<std::uint32_t>(run.completed.size()),
      0};
}

std::string evaluation_scenario_id(EvaluationScenario scenario) {
  switch (scenario) {
    case EvaluationScenario::CandidateVsRuleBased:
      return "candidate-vs-rule-based";
    case EvaluationScenario::CandidateVsFrozen:
      return "candidate-vs-frozen-policy";
    case EvaluationScenario::CandidateVsOpponentPool:
      return "candidate-vs-opponent-pool";
    case EvaluationScenario::Tournament:
      return "tournament";
  }

  throw std::runtime_error("invalid evaluation scenario");
}

}  // namespace napoleon::evaluation
