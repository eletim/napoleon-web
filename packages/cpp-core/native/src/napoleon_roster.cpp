#include "napoleon_roster.hpp"

#include <cmath>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace napoleon {
namespace {

void validate_seat_index(int seat_index) {
  if (seat_index < 0 || seat_index >= kPlayerCount) {
    throw std::runtime_error("seat index out of range");
  }
}

void validate_agent(const AgentIdentity& agent) {
  if (agent.id.empty()) {
    throw std::runtime_error("agent id must not be empty");
  }
}

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

template <typename T, typename Writer>
void write_array(std::ostream& out, const std::vector<T>& values, Writer writer) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    writer(out, values[index]);
  }
  out << ']';
}

std::uint32_t mix_roster_seed(
    std::uint32_t roster_seed,
    std::uint32_t game_index,
    int seat_index) {
  std::uint32_t value = roster_seed;
  value ^= game_index + 0x9e3779b9u + (value << 6) + (value >> 2);
  value ^= static_cast<std::uint32_t>(seat_index) + 0x85ebca6bu + (value << 6) + (value >> 2);
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

void validate_opponent_pool(const std::vector<WeightedAgent>& pool) {
  if (pool.empty()) {
    throw std::runtime_error("opponent pool must not be empty");
  }

  std::uint64_t total_weight = 0;
  for (const WeightedAgent& entry : pool) {
    validate_agent(entry.agent);
    if (entry.weight == 0) {
      throw std::runtime_error("opponent pool weight must be positive");
    }
    total_weight += entry.weight;
  }

  if (total_weight == 0) {
    throw std::runtime_error("opponent pool total weight must be positive");
  }
}

AgentIdentity sample_weighted_agent(
    const std::vector<WeightedAgent>& pool,
    std::uint32_t roster_seed,
    std::uint32_t game_index,
    int seat_index) {
  validate_opponent_pool(pool);

  std::uint64_t total_weight = 0;
  for (const WeightedAgent& entry : pool) {
    total_weight += entry.weight;
  }

  SeededRandom rng(mix_roster_seed(roster_seed, game_index, seat_index));
  const auto threshold = static_cast<std::uint64_t>(std::floor(rng.next() * total_weight));
  std::uint64_t cumulative = 0;
  for (const WeightedAgent& entry : pool) {
    cumulative += entry.weight;
    if (threshold < cumulative) {
      return entry.agent;
    }
  }

  return pool.back().agent;
}

void write_agent_identity(std::ostream& out, const AgentIdentity& agent) {
  out << "{\"type\":";
  json_escape(out, agent_type_id(agent.type));
  out << ",\"id\":";
  json_escape(out, agent.id);
  out << '}';
}

void write_weighted_agent(std::ostream& out, const WeightedAgent& entry) {
  out << "{\"agent\":";
  write_agent_identity(out, entry.agent);
  out << ",\"weight\":" << entry.weight << '}';
}

}  // namespace

AgentIdentity rule_based_agent(std::string id) {
  return AgentIdentity{AgentType::RuleBased, std::move(id)};
}

AgentIdentity current_policy_agent(std::string id) {
  return AgentIdentity{AgentType::CurrentPolicy, std::move(id)};
}

AgentIdentity frozen_policy_agent(std::string id) {
  return AgentIdentity{AgentType::FrozenPolicy, std::move(id)};
}

RosterSpec fixed_roster(const std::array<AgentIdentity, kPlayerCount>& agents) {
  RosterSpec spec;
  spec.kind = RosterSpec::Kind::Fixed;
  spec.fixed_agents = agents;
  for (const AgentIdentity& agent : spec.fixed_agents) {
    validate_agent(agent);
  }
  return spec;
}

RosterSpec self_play_roster(const AgentIdentity& agent) {
  validate_agent(agent);

  std::array<AgentIdentity, kPlayerCount> agents;
  agents.fill(agent);
  return fixed_roster(agents);
}

RosterSpec current_plus_opponent_pool_roster(
    const AgentIdentity& current_agent,
    std::vector<WeightedAgent> opponent_pool,
    bool rotate_current_seat,
    int current_seat_index) {
  validate_agent(current_agent);
  validate_opponent_pool(opponent_pool);
  validate_seat_index(current_seat_index);

  RosterSpec spec;
  spec.kind = RosterSpec::Kind::CurrentPlusOpponentPool;
  spec.current_agent = current_agent;
  spec.opponent_pool = std::move(opponent_pool);
  spec.rotate_current_seat = rotate_current_seat;
  spec.current_seat_index = current_seat_index;
  return spec;
}

RosterAssignment sample_roster(
    const RosterSpec& spec,
    std::uint32_t roster_seed,
    std::uint32_t game_index) {
  if (spec.kind == RosterSpec::Kind::Fixed) {
    for (const AgentIdentity& agent : spec.fixed_agents) {
      validate_agent(agent);
    }
    return RosterAssignment{spec.fixed_agents, -1};
  }

  validate_agent(spec.current_agent);
  validate_opponent_pool(spec.opponent_pool);
  validate_seat_index(spec.current_seat_index);

  const int current_seat =
      spec.rotate_current_seat
          ? static_cast<int>(game_index % static_cast<std::uint32_t>(kPlayerCount))
          : spec.current_seat_index;

  RosterAssignment assignment;
  assignment.current_seat_index = current_seat;
  for (int seat_index = 0; seat_index < kPlayerCount; ++seat_index) {
    assignment.agents[static_cast<std::size_t>(seat_index)] =
        seat_index == current_seat
            ? spec.current_agent
            : sample_weighted_agent(spec.opponent_pool, roster_seed, game_index, seat_index);
  }
  return assignment;
}

std::string roster_spec_manifest_json(const RosterSpec& spec) {
  std::ostringstream out;
  out << "{\"kind\":";
  if (spec.kind == RosterSpec::Kind::Fixed) {
    for (const AgentIdentity& agent : spec.fixed_agents) {
      validate_agent(agent);
    }
    json_escape(out, "fixed");
    out << ",\"seats\":[";
    for (int seat_index = 0; seat_index < kPlayerCount; ++seat_index) {
      if (seat_index != 0) {
        out << ',';
      }
      out << "{\"seatIndex\":" << seat_index << ",\"agent\":";
      write_agent_identity(out, spec.fixed_agents[static_cast<std::size_t>(seat_index)]);
      out << '}';
    }
    out << "]}";
    return out.str();
  }

  validate_agent(spec.current_agent);
  validate_opponent_pool(spec.opponent_pool);
  validate_seat_index(spec.current_seat_index);

  json_escape(out, "current-plus-opponent-pool");
  out << ",\"currentAgent\":";
  write_agent_identity(out, spec.current_agent);
  out << ",\"rotateCurrentSeat\":" << (spec.rotate_current_seat ? "true" : "false");
  out << ",\"currentSeatIndex\":";
  if (spec.rotate_current_seat) {
    out << "null";
  } else {
    out << spec.current_seat_index;
  }
  out << ",\"opponentPool\":";
  write_array(out, spec.opponent_pool, write_weighted_agent);
  out << '}';
  return out.str();
}

std::string roster_assignment_manifest_json(const RosterAssignment& assignment) {
  for (const AgentIdentity& agent : assignment.agents) {
    validate_agent(agent);
  }

  std::ostringstream out;
  out << "{\"currentSeatIndex\":";
  if (assignment.current_seat_index < 0) {
    out << "null";
  } else {
    validate_seat_index(assignment.current_seat_index);
    out << assignment.current_seat_index;
  }

  out << ",\"seats\":[";
  for (int seat_index = 0; seat_index < kPlayerCount; ++seat_index) {
    if (seat_index != 0) {
      out << ',';
    }
    out << "{\"seatIndex\":" << seat_index << ",\"agent\":";
    write_agent_identity(out, assignment.agents[static_cast<std::size_t>(seat_index)]);
    out << '}';
  }
  out << "]}";
  return out.str();
}

std::string agent_type_id(AgentType type) {
  switch (type) {
    case AgentType::RuleBased:
      return "rule-based";
    case AgentType::CurrentPolicy:
      return "current-policy";
    case AgentType::FrozenPolicy:
      return "frozen-policy";
  }

  throw std::runtime_error("invalid agent type");
}

}  // namespace napoleon
