#include "napoleon_core.hpp"
#include "napoleon_observation.hpp"
#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::uint32_t parse_seed(const std::string& value) {
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error("seed must be an integer between 0 and 4294967295");
  }

  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("seed must be an integer between 0 and 4294967295");
  }

  return static_cast<std::uint32_t>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::uint32_t seed = 0;
    std::uint32_t agent_seed = 0;
    bool select_rule_based_action = false;
    bool adjutant_location_oracle = false;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--seed" && index + 1 < argc) {
        seed = parse_seed(argv[++index]);
      } else if (arg == "--agent-seed" && index + 1 < argc) {
        agent_seed = parse_seed(argv[++index]);
      } else if (arg == "--select-rule-based-action") {
        select_rule_based_action = true;
      } else if (arg == "--adjutant-location-oracle") {
        adjutant_location_oracle = true;
      } else if (arg == "--snapshot") {
        continue;
      } else {
        throw std::runtime_error(
            "usage: napoleon_core_cli (--snapshot | --select-rule-based-action | "
            "--adjutant-location-oracle) "
            "--seed <uint32> [--agent-seed <uint32>]");
      }
    }

    napoleon::GameState state = napoleon::create_initial_game(seed);
    if (!adjutant_location_oracle) {
      for (std::string line; std::getline(std::cin, line);) {
        if (line.empty() || line[0] == '#') {
          continue;
        }
        napoleon::apply_action(state, napoleon::parse_action_line(line));
      }
    }

    if (adjutant_location_oracle) {
      // Diagnostic-only batch protocol: each stdin line is "seed napoleonSeatIndex".
      // Location is measured on the pre-exchange deal; bidding cannot move cards.
      for (std::string line; std::getline(std::cin, line);) {
        if (line.empty() || line[0] == '#') {
          continue;
        }
        std::istringstream input(line);
        unsigned long long raw_seed = 0;
        int napoleon_index = -1;
        if (!(input >> raw_seed >> napoleon_index) ||
            raw_seed > std::numeric_limits<std::uint32_t>::max() ||
            napoleon_index < 0 || napoleon_index >= napoleon::kPlayerCount) {
          throw std::runtime_error("oracle input must be: <uint32 seed> <napoleon seat 0..4>");
        }
        const napoleon::GameState deal =
            napoleon::create_initial_game(static_cast<std::uint32_t>(raw_seed));
        std::cout << "{\"seed\":" << raw_seed << ",\"napoleonSeatIndex\":"
                  << napoleon_index << ",\"classIndices\":[";
        for (std::size_t card_index = 0; card_index < napoleon::create_deck().size();
             ++card_index) {
          if (card_index != 0) std::cout << ',';
          const napoleon::Card card{static_cast<std::uint8_t>(card_index)};
          int owner = -1;
          for (int player_index = 0; player_index < napoleon::kPlayerCount; ++player_index) {
            const auto& hand = deal.hands[static_cast<std::size_t>(player_index)];
            if (std::find_if(hand.begin(), hand.end(), [&](napoleon::Card held) {
                  return held.id == card.id;
                }) != hand.end()) {
              owner = player_index;
              break;
            }
          }
          const int relative = owner < 0 ? 0 : (owner - napoleon_index + napoleon::kPlayerCount) % napoleon::kPlayerCount;
          std::cout << (relative == 0 ? 4 : relative - 1);
        }
        std::cout << "],\"cardIds\":[";
        const auto deck = napoleon::create_deck();
        for (std::size_t card_index = 0; card_index < deck.size(); ++card_index) {
          if (card_index != 0) std::cout << ',';
          std::cout << '\"' << napoleon::card_id(deck[card_index]) << '\"';
        }
        std::cout << "]}\n";
      }
    } else if (select_rule_based_action) {
      napoleon::SeededRandom rng(agent_seed);
      const napoleon::Action action = napoleon::select_agent_action(
          napoleon::rule_based_agent(), state, state.current_player_index, rng);
      std::cout << napoleon::action_json(action) << '\n';
    } else {
      std::cout
          << napoleon::observation::
                 canonical_snapshot_with_current_player_playing_model_input_json(state)
          << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
