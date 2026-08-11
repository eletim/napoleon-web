#include "napoleon_core.hpp"
#include "napoleon_onnx_policy.hpp"
#include "napoleon_rule_based.hpp"
#include "napoleon_roster.hpp"
#include "napoleon_simulation_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

bool same_agent(const napoleon::AgentIdentity& left, const napoleon::AgentIdentity& right) {
  return left.type == right.type && left.id == right.id;
}

napoleon::AgentResult first_result_for_request(const napoleon::AgentRequest& request) {
  napoleon::AgentResult result;
  result.request_id = request.request_id;
  result.action = request.legal_actions.front();
  return result;
}

std::vector<napoleon::FinishedGame> drive_external_first_legal(
    napoleon::SimulationRuntime& runtime,
    std::size_t expected_finished_count) {
  std::vector<napoleon::FinishedGame> all_finished;
  for (int iteration = 0; iteration < 10000; ++iteration) {
    runtime.advance_runnable_games();
    std::vector<napoleon::AgentRequest> requests = runtime.collect_agent_requests();
    if (!requests.empty()) {
      std::vector<napoleon::AgentResult> results;
      results.reserve(requests.size());
      for (const napoleon::AgentRequest& request : requests) {
        assert(!request.legal_actions.empty());
        results.push_back(first_result_for_request(request));
      }
      runtime.submit_agent_results(results);
    }

    std::vector<napoleon::FinishedGame> finished = runtime.collect_finished_games();
    all_finished.insert(
        all_finished.end(),
        std::make_move_iterator(finished.begin()),
        std::make_move_iterator(finished.end()));
    if (all_finished.size() == expected_finished_count) {
      return all_finished;
    }
  }

  assert(false && "runtime did not finish within iteration budget");
  return {};
}

std::vector<napoleon::AgentRequest> collect_playing_requests(
    napoleon::SimulationRuntime& runtime,
    std::size_t expected_request_count) {
  for (int iteration = 0; iteration < 10000; ++iteration) {
    runtime.advance_runnable_games();
    std::vector<napoleon::AgentRequest> requests = runtime.collect_agent_requests();
    if (requests.empty()) {
      continue;
    }

    const bool all_playing = std::all_of(
        requests.begin(),
        requests.end(),
        [](const napoleon::AgentRequest& request) {
          return request.phase == napoleon::Phase::Playing;
        });
    if (all_playing) {
      assert(requests.size() == expected_request_count);
      return requests;
    }

    std::vector<napoleon::AgentResult> setup_results;
    setup_results.reserve(requests.size());
    for (const napoleon::AgentRequest& request : requests) {
      assert(request.phase != napoleon::Phase::Playing);
      assert(!request.legal_actions.empty());
      setup_results.push_back(first_result_for_request(request));
    }
    runtime.submit_agent_results(setup_results);
  }

  assert(false && "runtime did not reach playing requests within iteration budget");
  return {};
}

bool same_policy_action_result(
    const napoleon::onnx_policy::PolicyActionResult& left,
    const napoleon::onnx_policy::PolicyActionResult& right) {
  return left.result.request_id == right.result.request_id &&
         left.selected_card_index == right.selected_card_index &&
         left.result.action.type == right.result.action.type &&
         left.result.action.card.id == right.result.action.card.id &&
         std::fabs(left.behavior_log_probability - right.behavior_log_probability) < 1e-12 &&
         left.policy_key == right.policy_key;
}

}  // namespace

int main() {
  napoleon::SeededRandom rng(0);
  assert(std::fabs(rng.next() - 0.26642920868471265) < 1e-15);
  assert(std::fabs(rng.next() - 0.0003297457005828619) < 1e-15);

  napoleon::GameState first = napoleon::create_initial_game(1234);
  napoleon::GameState second = napoleon::create_initial_game(1234);
  napoleon::GameState different = napoleon::create_initial_game(1235);

  assert(napoleon::canonical_snapshot_json(first) == napoleon::canonical_snapshot_json(second));
  assert(napoleon::canonical_snapshot_json(first) != napoleon::canonical_snapshot_json(different));

  for (const auto& hand : first.hands) {
    assert(hand.size() == 10);
  }
  assert(first.unused_cards.size() == 3);

  for (int player_index = 0; player_index < 5; ++player_index) {
    napoleon::Action pass;
    pass.type = napoleon::Action::Type::Pass;
    pass.player_index = player_index;
    napoleon::apply_action(first, pass);
  }

  assert(first.phase == napoleon::Phase::ChoosingAdjutant);
  assert(first.contract.has_value());
  assert(first.contract->napoleon_player_index == 0);
  assert(first.contract->trump_suit == napoleon::Suit::Spades);
  assert(first.contract->target_point_cards == 12);

  const napoleon::AgentIdentity current = napoleon::current_policy_agent("current");
  const napoleon::AgentIdentity rule = napoleon::rule_based_agent("rule");
  const napoleon::AgentIdentity frozen = napoleon::frozen_policy_agent("rl-v740");

  const napoleon::RosterSpec fixed =
      napoleon::fixed_roster({current, rule, frozen, rule, frozen});
  const napoleon::RosterAssignment fixed_assignment = napoleon::sample_roster(fixed, 7, 99);
  assert(fixed_assignment.current_seat_index == -1);
  assert(same_agent(fixed_assignment.agents[0], current));
  assert(same_agent(fixed_assignment.agents[2], frozen));

  const napoleon::RosterAssignment self_play =
      napoleon::sample_roster(napoleon::self_play_roster(current), 7, 99);
  for (const napoleon::AgentIdentity& agent : self_play.agents) {
    assert(same_agent(agent, current));
  }

  const napoleon::RosterSpec pool = napoleon::current_plus_opponent_pool_roster(
      current,
      {napoleon::WeightedAgent{rule, 1}, napoleon::WeightedAgent{frozen, 1}});
  const napoleon::RosterAssignment sampled_a = napoleon::sample_roster(pool, 12345, 17);
  const napoleon::RosterAssignment sampled_b = napoleon::sample_roster(pool, 12345, 17);
  assert(napoleon::roster_assignment_manifest_json(sampled_a) ==
         napoleon::roster_assignment_manifest_json(sampled_b));

  std::array<int, napoleon::kPlayerCount> current_seat_counts{};
  for (std::uint32_t game_index = 0; game_index < 100; ++game_index) {
    const napoleon::RosterAssignment sampled = napoleon::sample_roster(pool, 999, game_index);
    ++current_seat_counts[static_cast<std::size_t>(sampled.current_seat_index)];
    for (int seat_index = 0; seat_index < napoleon::kPlayerCount; ++seat_index) {
      if (seat_index == sampled.current_seat_index) {
        assert(same_agent(sampled.agents[static_cast<std::size_t>(seat_index)], current));
      } else {
        assert(!same_agent(sampled.agents[static_cast<std::size_t>(seat_index)], current));
      }
    }
  }
  for (int count : current_seat_counts) {
    assert(count == 20);
  }

  bool found_independent_seat_sample = false;
  for (std::uint32_t game_index = 0; game_index < 1000; ++game_index) {
    const napoleon::RosterAssignment sampled = napoleon::sample_roster(pool, 321, game_index);
    bool saw_rule = false;
    bool saw_frozen = false;
    for (int seat_index = 0; seat_index < napoleon::kPlayerCount; ++seat_index) {
      if (seat_index == sampled.current_seat_index) {
        continue;
      }
      saw_rule = saw_rule || same_agent(sampled.agents[static_cast<std::size_t>(seat_index)], rule);
      saw_frozen =
          saw_frozen || same_agent(sampled.agents[static_cast<std::size_t>(seat_index)], frozen);
    }
    found_independent_seat_sample = found_independent_seat_sample || (saw_rule && saw_frozen);
  }
  assert(found_independent_seat_sample);

  const std::string spec_manifest = napoleon::roster_spec_manifest_json(pool);
  assert(spec_manifest.find("current-plus-opponent-pool") != std::string::npos);
  assert(spec_manifest.find("rl-v740") != std::string::npos);
  assert(napoleon::roster_assignment_manifest_json(sampled_a).find("\"seats\"") !=
         std::string::npos);

  int completed_rule_games = 0;
  int selected_rule_actions = 0;
  for (std::uint32_t game_seed : {424242u, 424243u, 424244u, 424245u}) {
    napoleon::GameState rule_game = napoleon::create_initial_game(game_seed);
    for (int player_index = 0; player_index < 5; ++player_index) {
      napoleon::Action pass;
      pass.type = napoleon::Action::Type::Pass;
      pass.player_index = player_index;
      napoleon::apply_action(rule_game, pass);
    }
    napoleon::Action choose;
    choose.type = napoleon::Action::Type::ChooseAdjutant;
    choose.player_index = rule_game.current_player_index;
    choose.card = napoleon::parse_card_id("joker");
    napoleon::apply_action(rule_game, choose);
    napoleon::Action discard;
    discard.type = napoleon::Action::Type::DiscardCards;
    discard.player_index = rule_game.current_player_index;
    discard.cards = {
        rule_game.hands[static_cast<std::size_t>(discard.player_index)][0],
        rule_game.hands[static_cast<std::size_t>(discard.player_index)][1],
        rule_game.hands[static_cast<std::size_t>(discard.player_index)][2]};
    napoleon::apply_action(rule_game, discard);

    napoleon::SeededRandom rule_rng(123 + game_seed);
    while (!rule_game.is_game_over) {
      if (rule_game.is_trick_complete) {
        napoleon::Action next;
        next.type = napoleon::Action::Type::AdvanceToNextTrick;
        napoleon::apply_action(rule_game, next);
        continue;
      }

      const napoleon::Action action = napoleon::select_agent_action(
          rule,
          rule_game,
          rule_game.current_player_index,
          rule_rng);
      assert(action.type == napoleon::Action::Type::PlayCard);
      napoleon::apply_action(rule_game, action);
      ++selected_rule_actions;
    }
    assert(rule_game.phase == napoleon::Phase::Finished);
    assert(rule_game.result.has_value());
    ++completed_rule_games;
  }
  assert(completed_rule_games == 4);
  assert(selected_rule_actions == 200);

  napoleon::SimulationRuntime cpu_runtime(napoleon::SimulationRuntimeConfig{
      napoleon::self_play_roster(rule),
      7000,
      8000,
      1000,
      nullptr});
  const std::vector<std::uint32_t> cpu_game_ids = cpu_runtime.add_games(1000);
  assert(cpu_game_ids.size() == 1000);
  const std::size_t cpu_transitions = cpu_runtime.advance_runnable_games();
  assert(cpu_transitions > 1000);
  assert(cpu_runtime.collect_agent_requests().empty());
  const std::vector<napoleon::FinishedGame> cpu_finished = cpu_runtime.collect_finished_games();
  assert(cpu_finished.size() == 1000);
  const napoleon::RuntimeMetrics cpu_metrics = cpu_runtime.metrics();
  assert(cpu_metrics.added_games == 1000);
  assert(cpu_metrics.finished_games == 1000);
  assert(cpu_metrics.agent_request_count == 0);
  assert(cpu_metrics.internal_transition_count == cpu_transitions);
  assert(cpu_metrics.games_per_second >= 0.0);
  assert(cpu_metrics.decisions_per_second > 0.0);

  napoleon::SimulationRuntime waiting_runtime(napoleon::SimulationRuntimeConfig{
      napoleon::fixed_roster({current, rule, rule, rule, rule}),
      42,
      24,
      16,
      nullptr});
  waiting_runtime.add_games(12);
  waiting_runtime.advance_runnable_games();
  std::vector<napoleon::AgentRequest> initial_requests =
      waiting_runtime.collect_agent_requests();
  assert(initial_requests.size() == 12);
  for (std::size_t index = 0; index < initial_requests.size(); ++index) {
    assert(initial_requests[index].game_id == index + 1);
    assert(initial_requests[index].sequence == index + 1);
    assert(initial_requests[index].player_index == 0);
    assert(initial_requests[index].phase == napoleon::Phase::Bidding);
    assert(same_agent(initial_requests[index].agent, current));
    assert(!initial_requests[index].snapshot_json.empty());
  }

  std::vector<napoleon::AgentResult> reversed_passes;
  for (auto it = initial_requests.rbegin(); it != initial_requests.rend(); ++it) {
    const auto pass_it = std::find_if(
        it->legal_actions.begin(),
        it->legal_actions.end(),
        [](const napoleon::Action& action) {
          return action.type == napoleon::Action::Type::Pass;
        });
    assert(pass_it != it->legal_actions.end());
    napoleon::AgentResult result;
    result.request_id = it->request_id;
    result.action = *pass_it;
    reversed_passes.push_back(result);
  }
  waiting_runtime.submit_agent_results(reversed_passes);
  waiting_runtime.advance_runnable_games();
  const std::vector<napoleon::AgentRequest> adjutant_requests =
      waiting_runtime.collect_agent_requests();
  assert(adjutant_requests.size() == 12);
  for (std::size_t index = 0; index < adjutant_requests.size(); ++index) {
    assert(adjutant_requests[index].game_id == index + 1);
    assert(adjutant_requests[index].sequence == index + 13);
    assert(adjutant_requests[index].phase == napoleon::Phase::ChoosingAdjutant);
  }
  const napoleon::RuntimeMetrics waiting_metrics = waiting_runtime.metrics();
  assert(waiting_metrics.agent_request_count == 24);
  assert(waiting_metrics.submitted_agent_result_count == 12);

  napoleon::SimulationRuntime onnx_request_runtime(napoleon::SimulationRuntimeConfig{
      napoleon::self_play_roster(current),
      2024,
      3030,
      8,
      napoleon::onnx_policy::attach_playing_model_input});
  onnx_request_runtime.add_games(5);
  std::vector<napoleon::AgentRequest> playing_requests =
      collect_playing_requests(onnx_request_runtime, 5);
  for (const napoleon::AgentRequest& request : playing_requests) {
    assert(request.phase == napoleon::Phase::Playing);
    assert(request.playing_model_input.size() ==
           napoleon::observation::kPlayingModelInputFeatureCount);
    assert(request.legal_play_mask.size() == napoleon::observation::kCardCount);
    assert(std::accumulate(request.legal_play_mask.begin(), request.legal_play_mask.end(), 0) > 0);
  }

  std::array<float, napoleon::onnx_policy::kPolicyLogitCount> logits{};
  logits.fill(0.0F);
  logits[52] = 100000.0F;
  logits[0] = 0.25F;
  logits[7] = 0.5F;

  napoleon::onnx_policy::BatchedPolicyExecutor batch_one(
      napoleon::onnx_policy::BatchedPolicyConfig{1, 1.0, 5150});
  batch_one.add_policy(
      napoleon::onnx_policy::policy_key_from_agent(current),
      std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(logits));
  const std::vector<napoleon::onnx_policy::PolicyActionResult> batch_one_results =
      batch_one.run(playing_requests);

  napoleon::onnx_policy::BatchedPolicyExecutor batch_many(
      napoleon::onnx_policy::BatchedPolicyConfig{4, 1.0, 5150});
  batch_many.add_policy(
      napoleon::onnx_policy::policy_key_from_agent(current),
      std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(logits));
  const std::vector<napoleon::onnx_policy::PolicyActionResult> batch_many_results =
      batch_many.run(playing_requests);

  assert(batch_one_results.size() == batch_many_results.size());
  for (std::size_t index = 0; index < batch_one_results.size(); ++index) {
    assert(same_policy_action_result(batch_one_results[index], batch_many_results[index]));
    const int selected_card_index = batch_many_results[index].selected_card_index;
    assert(playing_requests[index].legal_play_mask[static_cast<std::size_t>(selected_card_index)] == 1);
    assert(std::isfinite(batch_many_results[index].behavior_log_probability));
    assert(batch_many_results[index].behavior_log_probability <= 1e-12);
  }
  const napoleon::onnx_policy::BatchedPolicyStats batch_one_stats = batch_one.stats();
  const napoleon::onnx_policy::BatchedPolicyStats batch_many_stats = batch_many.stats();
  assert(batch_one_stats.request_count == playing_requests.size());
  assert(batch_one_stats.session_run_count == playing_requests.size());
  assert(batch_one_stats.max_observed_batch_size == 1);
  assert(batch_many_stats.request_count == playing_requests.size());
  assert(batch_many_stats.session_run_count == 2);
  assert(batch_many_stats.max_observed_batch_size == 4);
  assert(batch_many_stats.batch_size_histogram.at(4) == 1);
  assert(batch_many_stats.batch_size_histogram.at(1) == 1);
  assert(batch_many_stats.policy_stats.at("current-policy:current").request_count ==
         playing_requests.size());

  std::vector<napoleon::AgentRequest> mixed_policy_requests;
  for (std::size_t index = 0; index < 5; ++index) {
    napoleon::AgentRequest request = playing_requests[index];
    request.request_id = static_cast<std::uint64_t>(index + 1);
    request.sequence = static_cast<std::uint64_t>(index + 1);
    request.agent = index % 2 == 0 ? current : frozen;
    mixed_policy_requests.push_back(std::move(request));
  }
  napoleon::onnx_policy::BatchedPolicyExecutor mixed_executor(
      napoleon::onnx_policy::BatchedPolicyConfig{2, 1.0, 777});
  mixed_executor.add_policy(
      napoleon::onnx_policy::policy_key_from_agent(current),
      std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(logits));
  mixed_executor.add_policy(
      napoleon::onnx_policy::policy_key_from_agent(frozen),
      std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(logits));
  const std::vector<napoleon::onnx_policy::PolicyActionResult> mixed_results =
      mixed_executor.run(mixed_policy_requests);
  assert(mixed_results.size() == mixed_policy_requests.size());
  for (std::size_t index = 0; index < mixed_results.size(); ++index) {
    assert(mixed_results[index].result.request_id == index + 1);
  }
  const napoleon::onnx_policy::BatchedPolicyStats mixed_stats = mixed_executor.stats();
  assert(mixed_stats.request_count == 5);
  assert(mixed_stats.session_run_count == 3);
  assert(mixed_stats.mean_batch_size > 1.6 && mixed_stats.mean_batch_size < 1.7);
  assert(mixed_stats.max_observed_batch_size == 2);
  assert(mixed_stats.policy_stats.at("current-policy:current").request_count == 3);
  assert(mixed_stats.policy_stats.at("current-policy:current").session_run_count == 2);
  assert(mixed_stats.policy_stats.at("frozen-policy:rl-v740").request_count == 2);
  assert(mixed_stats.policy_stats.at("frozen-policy:rl-v740").session_run_count == 1);

  napoleon::AgentRequest forced_request = playing_requests.front();
  forced_request.legal_actions = {playing_requests.front().legal_actions.front()};
  forced_request.legal_play_mask.assign(napoleon::observation::kCardCount, 0);
  const int forced_card_index =
      napoleon::observation::playing_card_model_index(forced_request.legal_actions.front().card);
  forced_request.legal_play_mask[static_cast<std::size_t>(forced_card_index)] = 1;
  napoleon::onnx_policy::BatchedPolicyExecutor forced_executor(
      napoleon::onnx_policy::BatchedPolicyConfig{8, 1.0, 999});
  forced_executor.add_policy(
      napoleon::onnx_policy::policy_key_from_agent(current),
      std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(logits));
  const std::vector<napoleon::onnx_policy::PolicyActionResult> forced_results =
      forced_executor.run({forced_request});
  assert(forced_results.size() == 1);
  assert(forced_results.front().selected_card_index == forced_card_index);
  assert(forced_results.front().behavior_log_probability == 0.0);

  bool rejected_non_playing = false;
  try {
    napoleon::AgentRequest invalid_request = playing_requests.front();
    invalid_request.phase = napoleon::Phase::Bidding;
    forced_executor.run({invalid_request});
  } catch (const std::runtime_error&) {
    rejected_non_playing = true;
  }
  assert(rejected_non_playing);

  bool rejected_missing_session = false;
  try {
    napoleon::onnx_policy::BatchedPolicyExecutor missing_session_executor(
        napoleon::onnx_policy::BatchedPolicyConfig{2, 1.0, 1});
    missing_session_executor.run({playing_requests.front()});
  } catch (const std::runtime_error&) {
    rejected_missing_session = true;
  }
  assert(rejected_missing_session);

  bool rejected_unenabled_onnxruntime = false;
  try {
    napoleon::onnx_policy::create_onnxruntime_policy_session(
        napoleon::onnx_policy::PolicySessionConfig{
            napoleon::onnx_policy::policy_key_from_agent(current),
            "/tmp/missing-policy.onnx",
            "model_input",
            "logits",
            napoleon::onnx_policy::InferenceDevice::Cuda});
  } catch (const std::runtime_error&) {
    rejected_unenabled_onnxruntime = true;
  }
  assert(rejected_unenabled_onnxruntime);

  napoleon::SimulationRuntime deterministic_a(napoleon::SimulationRuntimeConfig{
      napoleon::fixed_roster({current, rule, frozen, rule, current}),
      123,
      456,
      4,
      nullptr});
  napoleon::SimulationRuntime deterministic_b(napoleon::SimulationRuntimeConfig{
      napoleon::fixed_roster({current, rule, frozen, rule, current}),
      123,
      456,
      4,
      nullptr});
  deterministic_a.add_games(4);
  deterministic_b.add_games(4);
  const std::vector<napoleon::FinishedGame> finished_a =
      drive_external_first_legal(deterministic_a, 4);
  const std::vector<napoleon::FinishedGame> finished_b =
      drive_external_first_legal(deterministic_b, 4);
  assert(finished_a.size() == finished_b.size());
  for (std::size_t index = 0; index < finished_a.size(); ++index) {
    assert(finished_a[index].game_id == finished_b[index].game_id);
    assert(finished_a[index].seed == finished_b[index].seed);
    assert(finished_a[index].snapshot_json == finished_b[index].snapshot_json);
    assert(napoleon::roster_assignment_manifest_json(finished_a[index].roster) ==
           napoleon::roster_assignment_manifest_json(finished_b[index].roster));
  }

  std::cout << "napoleon_core_self_test ok\n";
  return 0;
}
