#pragma once

#include "napoleon_core.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace napoleon {

enum class AgentType : std::uint8_t {
  RuleBased,
  CurrentPolicy,
  FrozenPolicy
};

struct AgentIdentity {
  AgentType type = AgentType::RuleBased;
  std::string id;
};

struct WeightedAgent {
  AgentIdentity agent;
  std::uint32_t weight = 1;
};

struct RosterAssignment {
  std::array<AgentIdentity, kPlayerCount> agents;
  int current_seat_index = -1;
};

struct RosterSpec {
  enum class Kind : std::uint8_t {
    Fixed,
    CurrentPlusOpponentPool
  };

  Kind kind = Kind::Fixed;
  std::array<AgentIdentity, kPlayerCount> fixed_agents;
  AgentIdentity current_agent;
  std::vector<WeightedAgent> opponent_pool;
  bool rotate_current_seat = true;
  int current_seat_index = 0;
};

AgentIdentity rule_based_agent(std::string id = "rule-based");
AgentIdentity current_policy_agent(std::string id = "current");
AgentIdentity frozen_policy_agent(std::string id);
RosterSpec fixed_roster(const std::array<AgentIdentity, kPlayerCount>& agents);
RosterSpec self_play_roster(const AgentIdentity& agent);
RosterSpec current_plus_opponent_pool_roster(
    const AgentIdentity& current_agent,
    std::vector<WeightedAgent> opponent_pool,
    bool rotate_current_seat = true,
    int current_seat_index = 0);
RosterAssignment sample_roster(
    const RosterSpec& spec,
    std::uint32_t roster_seed,
    std::uint32_t game_index);
std::string roster_spec_manifest_json(const RosterSpec& spec);
std::string roster_assignment_manifest_json(const RosterAssignment& assignment);

std::string agent_type_id(AgentType type);

}  // namespace napoleon
