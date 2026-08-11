#include "napoleon_core.hpp"
#include "napoleon_evaluation.hpp"
#include "napoleon_rule_based.hpp"
#include "napoleon_roster.hpp"
#include "napoleon_simulation_runtime.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool same_agent(const napoleon::AgentIdentity& left, const napoleon::AgentIdentity& right) {
  return left.type == right.type && left.id == right.id;
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
        results.push_back(napoleon::AgentResult{request.request_id, request.legal_actions.front()});
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
      1000});
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
      16});
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
    reversed_passes.push_back(napoleon::AgentResult{it->request_id, *pass_it});
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

  napoleon::SimulationRuntime deterministic_a(napoleon::SimulationRuntimeConfig{
      napoleon::fixed_roster({current, rule, frozen, rule, current}),
      123,
      456,
      4});
  napoleon::SimulationRuntime deterministic_b(napoleon::SimulationRuntimeConfig{
      napoleon::fixed_roster({current, rule, frozen, rule, current}),
      123,
      456,
      4});
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

  napoleon::SimulationRuntime scheduled_runtime(napoleon::SimulationRuntimeConfig{
      napoleon::self_play_roster(rule),
      1,
      2,
      2});
  scheduled_runtime.add_scheduled_games({
      napoleon::ScheduledGame{
          900,
          napoleon::sample_roster(napoleon::fixed_roster({current, rule, rule, rule, rule}), 0, 0)},
      napoleon::ScheduledGame{
          900,
          napoleon::sample_roster(napoleon::fixed_roster({rule, current, rule, rule, rule}), 0, 1)}});
  scheduled_runtime.advance_runnable_games();
  const std::vector<napoleon::AgentRequest> scheduled_requests =
      scheduled_runtime.collect_agent_requests();
  assert(scheduled_requests.size() == 2);
  assert(scheduled_runtime.game_snapshots()[0].seed == 900);
  assert(scheduled_runtime.game_snapshots()[1].seed == 900);

  const napoleon::evaluation::EvaluationArtifact eval_artifact =
      napoleon::evaluation::run_evaluation(napoleon::evaluation::EvaluationOptions{
          napoleon::evaluation::EvaluationScenario::CandidateVsRuleBased,
          12,
          2,
          34,
          8,
          3,
          {0, 1, 2, 3, 4},
          "candidate",
          "rl-v740"});
  assert(eval_artifact.scheduled_games == 10);
  assert(eval_artifact.completed_games == 10);
  assert(eval_artifact.failed_games == 0);
  assert(eval_artifact.json.find("\"candidate-vs-rule-based\"") != std::string::npos);
  assert(eval_artifact.json.find("\"requestCount\"") != std::string::npos);
  assert(eval_artifact.json.find("\"policyStats\"") != std::string::npos);
  assert(eval_artifact.json.find("\"tsCudaBatch1Workers4SecondsPer2000Games\":11.2") !=
         std::string::npos);
  assert(eval_artifact.json.find("\"usesRlDatasetGeneration\":false") != std::string::npos);

  const napoleon::evaluation::EvaluationArtifact tournament_artifact =
      napoleon::evaluation::run_evaluation(napoleon::evaluation::EvaluationOptions{
          napoleon::evaluation::EvaluationScenario::Tournament,
          22,
          1,
          44,
          8,
          4,
          {0, 1, 2, 3, 4},
          "candidate",
          "rl-v740"});
  assert(tournament_artifact.scheduled_games == 5);
  assert(tournament_artifact.completed_games == 5);
  assert(tournament_artifact.json.find("\"tournament\"") != std::string::npos);
  assert(tournament_artifact.json.find("RuleBasedAgent-A") != std::string::npos);

  std::cout << "napoleon_core_self_test ok\n";
  return 0;
}
