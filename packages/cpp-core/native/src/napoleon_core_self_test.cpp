#include "napoleon_core.hpp"
#include "napoleon_roster.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

bool same_agent(const napoleon::AgentIdentity& left, const napoleon::AgentIdentity& right) {
  return left.type == right.type && left.id == right.id;
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

  std::cout << "napoleon_core_self_test ok\n";
  return 0;
}
