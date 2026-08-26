#include "napoleon_parameterized_policy.hpp"
#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

bool is_joker(napoleon::Card card) { return card.id == 52; }
int rank_index(napoleon::Card card) { return card.id < 52 ? card.id % 13 : -1; }
int suit_index(napoleon::Card card) { return card.id < 52 ? card.id / 13 : -1; }
bool is_oruma(napoleon::Card card) { return card.id == 0; }
bool is_yoromeki(napoleon::Card card) { return card.id == 24; }
napoleon::Card sei_jack(napoleon::Suit trump) {
  return napoleon::Card{static_cast<std::uint8_t>(static_cast<int>(trump) * 13 + 10)};
}
napoleon::Card ura_jack(napoleon::Suit trump) {
  static constexpr int suits[] = {3, 2, 1, 0};
  return napoleon::Card{static_cast<std::uint8_t>(suits[static_cast<int>(trump)] * 13 + 10)};
}
double ai_rank(napoleon::Card card) {
  static constexpr double values[] = {20, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  return card.id < 52 ? values[rank_index(card)] : 0;
}
double rule_card_value(napoleon::Card card, napoleon::Suit trump) {
  if (is_joker(card)) return 20;
  if (is_oruma(card)) return 60;
  if (card.id == sei_jack(trump).id) return 55;
  if (card.id == ura_jack(trump).id) return 50;
  if (is_yoromeki(card)) return suit_index(card) == static_cast<int>(trump) ? 45 : 30;
  return ai_rank(card) + (suit_index(card) == static_cast<int>(trump) ? 30 : 0);
}
double kept_value(
    const napoleon::GameState& state,
    const std::vector<napoleon::Card>& discarded) {
  std::set<std::uint8_t> ids;
  for (auto card : discarded) ids.insert(card.id);
  double value = 0;
  for (auto card : state.hands[state.current_player_index]) {
    if (ids.count(card.id) == 0) value += rule_card_value(card, state.contract->trump_suit);
  }
  return value;
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    using namespace napoleon;
    using namespace napoleon::parameterized_policy;
    require(feature_schema().size() == static_cast<std::size_t>(kParameterCount), "schema size");
    require(kParameterCount == 95, "parameter count fixture");
    const Parameters initial = initial_rule_based_parameters();
    validate_parameters(initial);

    int checked = 0;
    for (std::uint32_t seed = 452000; seed < 470000 && checked < 100; ++seed) {
      GameState source = create_initial_game(seed);
      SeededRandom bidding_rng(seed ^ 17u);
      int guard = 0;
      while (source.phase == Phase::Bidding && ++guard < 200) {
        apply_action(
            source,
            select_rule_based_action(source, source.current_player_index, bidding_rng));
      }
      if (source.phase != Phase::ChoosingAdjutant || !source.contract.has_value()) continue;
      const int napoleon = source.contract->napoleon_player_index;
      SeededRandom rb_rng(seed ^ 31u);
      const Action rb_adjutant = select_rule_based_action(source, napoleon, rb_rng);
      const SelectionResult learned_adjutant = select_adjutant(source, napoleon, initial);
      require(rb_adjutant.card.id == learned_adjutant.action.card.id, "initial adjutant mismatch");
      const auto first_features = extract_adjutant_features(source, napoleon, rb_adjutant.card);
      const auto second_features = extract_adjutant_features(source, napoleon, rb_adjutant.card);
      require(first_features == second_features, "adjutant features are not deterministic");
      GameState hidden_mutation = source;
      for (int player = 0; player < kPlayerCount; ++player) {
        if (player != napoleon) std::reverse(hidden_mutation.hands[player].begin(), hidden_mutation.hands[player].end());
      }
      require(
          first_features == extract_adjutant_features(hidden_mutation, napoleon, rb_adjutant.card),
          "adjutant features depend on hidden hands");

      std::vector<std::uint8_t> kitty;
      for (Card card : source.unused_cards) kitty.push_back(card.id);
      apply_action(source, learned_adjutant.action);
      const Action rb_exchange = select_rule_based_action(source, napoleon, rb_rng);
      const SelectionResult learned_exchange = select_exchange(source, napoleon, kitty, initial);
      require(
          std::fabs(kept_value(source, rb_exchange.cards) -
                    kept_value(source, learned_exchange.action.cards)) < 1e-9,
          "initial exchange does not reproduce RuleBased value");
      require(
          extract_exchange_features(source, napoleon, learned_exchange.action.cards, kitty) ==
              extract_exchange_features(source, napoleon, learned_exchange.action.cards, kitty),
          "exchange features are not deterministic");
      apply_action(source, learned_exchange.action);
      require(source.phase == Phase::Playing, "joint policy did not reach playing");
      ++checked;
    }
    require(checked == 100, "insufficient policy fixtures");
    std::cout << "parameterized policy tests passed: " << checked << " fixtures\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
