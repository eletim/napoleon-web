#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace napoleon {
namespace {

constexpr double kEpsilon = 1e-9;

bool is_joker(Card card) {
  return card.id == 52;
}

Suit card_suit(Card card) {
  if (is_joker(card)) {
    throw std::runtime_error("joker has no suit");
  }
  return static_cast<Suit>(card.id / 13);
}

Rank card_rank(Card card) {
  if (is_joker(card)) {
    throw std::runtime_error("joker has no rank");
  }
  return static_cast<Rank>(card.id % 13);
}

int rank_value(Rank rank) {
  switch (rank) {
    case Rank::Two:
      return 2;
    case Rank::Three:
      return 3;
    case Rank::Four:
      return 4;
    case Rank::Five:
      return 5;
    case Rank::Six:
      return 6;
    case Rank::Seven:
      return 7;
    case Rank::Eight:
      return 8;
    case Rank::Nine:
      return 9;
    case Rank::Ten:
      return 10;
    case Rank::Jack:
      return 11;
    case Rank::Queen:
      return 12;
    case Rank::King:
      return 13;
    case Rank::Ace:
      return 14;
  }
  throw std::runtime_error("invalid rank");
}

int ai_rank_value(Rank rank) {
  switch (rank) {
    case Rank::Ace:
      return 20;
    case Rank::King:
      return 13;
    case Rank::Queen:
      return 12;
    case Rank::Jack:
      return 11;
    case Rank::Ten:
      return 10;
    case Rank::Nine:
      return 9;
    case Rank::Eight:
      return 8;
    case Rank::Seven:
      return 7;
    case Rank::Six:
      return 6;
    case Rank::Five:
      return 5;
    case Rank::Four:
      return 4;
    case Rank::Three:
      return 3;
    case Rank::Two:
      return 12;
  }
  throw std::runtime_error("invalid rank");
}

bool is_point_card(Card card) {
  if (is_joker(card)) {
    return false;
  }

  const Rank rank = card_rank(card);
  return rank == Rank::Ten || rank == Rank::Jack || rank == Rank::Queen ||
         rank == Rank::King || rank == Rank::Ace;
}

bool is_oruma(Card card) {
  return card.id == parse_card_id("spades-A").id;
}

bool is_yoromeki(Card card) {
  return card.id == parse_card_id("hearts-Q").id;
}

Card sei_jack(Suit trump_suit) {
  return Card{static_cast<std::uint8_t>(static_cast<int>(trump_suit) * 13 + 10)};
}

Card ura_jack(Suit trump_suit) {
  switch (trump_suit) {
    case Suit::Spades:
      return parse_card_id("clubs-J");
    case Suit::Clubs:
      return parse_card_id("spades-J");
    case Suit::Hearts:
      return parse_card_id("diamonds-J");
    case Suit::Diamonds:
      return parse_card_id("hearts-J");
  }
  throw std::runtime_error("invalid trump suit");
}

std::optional<Suit> lead_suit(const std::vector<PlayedCard>& trick, Suit trump_suit) {
  if (trick.empty()) {
    return std::nullopt;
  }

  const Card lead_card = trick.front().card;
  return is_joker(lead_card) ? trump_suit : card_suit(lead_card);
}

std::optional<int> find_same_two_winner(
    const std::vector<PlayedCard>& trick,
    Suit trump_suit,
    int trick_number) {
  if (trick_number <= 1 || trick.size() != kPlayerCount) {
    return std::nullopt;
  }

  if (std::any_of(trick.begin(), trick.end(), [](const PlayedCard& played) {
        return is_joker(played.card);
      })) {
    return std::nullopt;
  }

  const Suit same_two_suit = card_suit(trick.front().card);
  for (const PlayedCard& played : trick) {
    if (card_suit(played.card) != same_two_suit || is_oruma(played.card) ||
        played.card.id == sei_jack(trump_suit).id ||
        played.card.id == ura_jack(trump_suit).id) {
      return std::nullopt;
    }
  }

  for (const PlayedCard& played : trick) {
    if (card_suit(played.card) == same_two_suit && card_rank(played.card) == Rank::Two) {
      return played.player_index;
    }
  }
  return std::nullopt;
}

std::optional<int> determine_special_winner(
    const std::vector<PlayedCard>& trick,
    Suit trump_suit) {
  auto oruma = std::find_if(trick.begin(), trick.end(), [](const PlayedCard& played) {
    return is_oruma(played.card);
  });
  auto yoromeki = std::find_if(trick.begin(), trick.end(), [](const PlayedCard& played) {
    return is_yoromeki(played.card);
  });

  if (oruma != trick.end() && yoromeki != trick.end()) {
    return yoromeki->player_index;
  }
  if (oruma != trick.end()) {
    return oruma->player_index;
  }

  const bool has_jack = std::any_of(trick.begin(), trick.end(), [](const PlayedCard& played) {
    return !is_joker(played.card) && card_rank(played.card) == Rank::Jack;
  });
  if (!has_jack) {
    return std::nullopt;
  }

  const Card sei = sei_jack(trump_suit);
  auto sei_play = std::find_if(trick.begin(), trick.end(), [&](const PlayedCard& played) {
    return played.card.id == sei.id;
  });
  if (sei_play != trick.end()) {
    return sei_play->player_index;
  }

  const Card ura = ura_jack(trump_suit);
  auto ura_play = std::find_if(trick.begin(), trick.end(), [&](const PlayedCard& played) {
    return played.card.id == ura.id;
  });
  if (ura_play != trick.end()) {
    return ura_play->player_index;
  }

  return std::nullopt;
}

struct Strength {
  int category_priority = 0;
  int rank_priority = 0;
};

Strength card_strength(Card card, Suit lead, Suit trump_suit, bool is_lead_card) {
  int category_priority = 0;
  if (is_joker(card)) {
    category_priority = is_lead_card ? 2 : 1;
  } else if (card_suit(card) == trump_suit) {
    category_priority = 2;
  } else if (card_suit(card) == lead) {
    category_priority = 1;
  }

  return Strength{category_priority, is_joker(card) ? 1 : rank_value(card_rank(card))};
}

int determine_current_winning_player(
    const std::vector<PlayedCard>& trick,
    Suit trump_suit,
    int trick_number) {
  const std::optional<int> same_two = find_same_two_winner(trick, trump_suit, trick_number);
  if (same_two.has_value()) {
    return *same_two;
  }

  const std::optional<int> special = determine_special_winner(trick, trump_suit);
  if (special.has_value()) {
    return *special;
  }

  const std::optional<Suit> lead = lead_suit(trick, trump_suit);
  if (!lead.has_value()) {
    throw std::runtime_error("cannot determine winner for empty trick");
  }

  int winner_index = 0;
  Strength winner_strength = card_strength(trick.front().card, *lead, trump_suit, true);
  for (std::size_t index = 1; index < trick.size(); ++index) {
    const Strength candidate_strength = card_strength(trick[index].card, *lead, trump_suit, false);
    if (candidate_strength.category_priority > winner_strength.category_priority ||
        (candidate_strength.category_priority == winner_strength.category_priority &&
         candidate_strength.rank_priority > winner_strength.rank_priority)) {
      winner_index = static_cast<int>(index);
      winner_strength = candidate_strength;
    }
  }

  return trick[static_cast<std::size_t>(winner_index)].player_index;
}

Suit get_trump_suit(const GameState& state) {
  if (state.contract.has_value()) {
    return state.contract->trump_suit;
  }
  if (state.trump_suit.has_value()) {
    return *state.trump_suit;
  }
  throw std::runtime_error("trump suit required");
}

double evaluate_card_for_trump(Card card, Suit trump_suit) {
  if (is_joker(card)) {
    return 20;
  }
  if (is_oruma(card)) {
    return 60;
  }
  if (card.id == sei_jack(trump_suit).id) {
    return 55;
  }
  if (card.id == ura_jack(trump_suit).id) {
    return 50;
  }
  if (is_yoromeki(card)) {
    return card_suit(card) == trump_suit ? 45 : 30;
  }
  return ai_rank_value(card_rank(card)) + (card_suit(card) == trump_suit ? 30 : 0);
}

double estimate_lead_win_probability(double card_score) {
  if (card_score < 10) {
    return 0.05;
  }
  if (card_score < 20) {
    return 0.1;
  }
  if (card_score < 30) {
    return 0.2;
  }
  if (card_score < 40) {
    return 0.35;
  }
  if (card_score < 50) {
    return 0.55;
  }
  if (card_score < 55) {
    return 0.7;
  }
  if (card_score < 60) {
    return 0.8;
  }
  return 0.9;
}

double adjust_team_win_probability(
    double base_probability,
    int remaining_players,
    bool is_team_currently_winning) {
  const double adjusted =
      is_team_currently_winning
          ? 1 - ((1 - base_probability) * remaining_players) / 4
          : (base_probability * remaining_players) / 4;
  return std::min(1.0, std::max(0.0, adjusted));
}

double calculate_used_card_value(double card_score) {
  return (card_score - 20) / 30;
}

bool contains_card(const std::vector<Card>& cards, Card card) {
  return std::any_of(cards.begin(), cards.end(), [&](Card candidate) {
    return candidate.id == card.id;
  });
}

bool self_has_called_card(const GameState& state, int self_index) {
  return state.adjutant.has_value() &&
         contains_card(state.hands[static_cast<std::size_t>(self_index)], state.adjutant->called_card);
}

bool is_self_team_player(const GameState& state, int candidate_player_index, int self_index) {
  if (candidate_player_index == self_index) {
    return true;
  }

  if (!state.contract.has_value()) {
    return false;
  }

  const int napoleon_player_index = state.contract->napoleon_player_index;
  const std::optional<int> revealed_adjutant_player_index =
      state.adjutant.has_value() && state.adjutant->revealed
          ? state.adjutant->player_index
          : std::nullopt;

  if (self_index == napoleon_player_index) {
    return revealed_adjutant_player_index.has_value() &&
           candidate_player_index == *revealed_adjutant_player_index;
  }

  if (self_has_called_card(state, self_index)) {
    return candidate_player_index == napoleon_player_index;
  }

  if (revealed_adjutant_player_index.has_value()) {
    return candidate_player_index != napoleon_player_index &&
           candidate_player_index != *revealed_adjutant_player_index;
  }

  return false;
}

double calculate_expected_point_cards_in_trick(
    const GameState& state,
    int player_index,
    Card candidate_card) {
  std::unordered_set<std::uint8_t> known_card_ids;
  for (Card card : state.hands[static_cast<std::size_t>(player_index)]) {
    known_card_ids.insert(card.id);
  }
  for (const CompletedTrick& trick : state.completed_tricks) {
    for (const PlayedCard& played : trick.cards) {
      known_card_ids.insert(played.card.id);
    }
  }
  for (const PlayedCard& played : state.current_trick) {
    known_card_ids.insert(played.card.id);
  }
  for (const AwardedPointCards& award : state.awarded_point_cards) {
    for (Card card : award.cards) {
      known_card_ids.insert(card.id);
    }
  }
  known_card_ids.insert(candidate_card.id);

  int unknown_card_count = 0;
  int unknown_point_card_count = 0;
  for (Card card : create_deck()) {
    if (known_card_ids.count(card.id) == 0) {
      ++unknown_card_count;
      if (is_point_card(card)) {
        ++unknown_point_card_count;
      }
    }
  }

  const int remaining_players =
      std::max(0, kPlayerCount - static_cast<int>(state.current_trick.size()) - 1);
  int current_point_card_count = is_point_card(candidate_card) ? 1 : 0;
  for (const PlayedCard& played : state.current_trick) {
    if (is_point_card(played.card)) {
      ++current_point_card_count;
    }
  }

  if (unknown_card_count == 0) {
    return current_point_card_count;
  }

  return current_point_card_count +
         (static_cast<double>(remaining_players) * unknown_point_card_count) / unknown_card_count;
}

double evaluate_play_action(
    const GameState& state,
    int player_index,
    Card card,
    Suit trump_suit) {
  const double card_score = evaluate_card_for_trump(card, trump_suit);
  const double base_probability = estimate_lead_win_probability(card_score);
  std::vector<PlayedCard> provisional_trick = state.current_trick;
  provisional_trick.push_back(PlayedCard{player_index, card});
  const int provisional_winner =
      determine_current_winning_player(provisional_trick, trump_suit, state.trick_number);
  const int remaining_players =
      std::max(0, kPlayerCount - static_cast<int>(provisional_trick.size()));
  const double team_win_probability = adjust_team_win_probability(
      base_probability,
      remaining_players,
      is_self_team_player(state, provisional_winner, player_index));
  const double expected_point_cards =
      calculate_expected_point_cards_in_trick(state, player_index, card);
  const double used_card_value = calculate_used_card_value(card_score);

  return (2 * team_win_probability - 1) * expected_point_cards - used_card_value;
}

Action select_random(const std::vector<Action>& actions, SeededRandom& rng) {
  if (actions.empty()) {
    throw std::runtime_error("cannot select from empty actions");
  }
  const int index = static_cast<int>(std::floor(rng.next() * actions.size()));
  return actions[static_cast<std::size_t>(std::min(index, static_cast<int>(actions.size()) - 1))];
}

}  // namespace

Action select_rule_based_action(
    const GameState& state,
    int player_index,
    SeededRandom& rng) {
  if (state.phase != Phase::Playing || state.is_trick_complete || state.is_game_over) {
    throw std::runtime_error("RuleBased C++ agent currently supports playing decisions only");
  }

  std::vector<Action> play_actions;
  for (const Action& action : get_legal_actions(state, player_index)) {
    if (action.type == Action::Type::PlayCard) {
      play_actions.push_back(action);
    }
  }

  if (play_actions.empty()) {
    throw std::runtime_error("no legal play actions");
  }
  if (play_actions.size() == 1) {
    return play_actions.front();
  }

  const Suit trump_suit = get_trump_suit(state);
  double max_value = -std::numeric_limits<double>::infinity();
  std::vector<Action> best_actions;
  for (const Action& action : play_actions) {
    const double value = evaluate_play_action(state, player_index, action.card, trump_suit);
    if (value > max_value + kEpsilon) {
      max_value = value;
      best_actions = {action};
    } else if (std::fabs(value - max_value) < kEpsilon) {
      best_actions.push_back(action);
    }
  }

  return select_random(best_actions, rng);
}

Action select_agent_action(
    const AgentIdentity& agent,
    const GameState& state,
    int player_index,
    SeededRandom& rng) {
  if (agent.type != AgentType::RuleBased) {
    throw std::runtime_error("only RuleBased CPU agent is available in this implementation");
  }
  return select_rule_based_action(state, player_index, rng);
}

}  // namespace napoleon
