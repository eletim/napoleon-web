#pragma once

#include "napoleon_core.hpp"
#include "napoleon_roster.hpp"

namespace napoleon {

Action select_rule_based_action(
    const GameState& state,
    int player_index,
    SeededRandom& rng);
Action select_agent_action(
    const AgentIdentity& agent,
    const GameState& state,
    int player_index,
    SeededRandom& rng);

}  // namespace napoleon
