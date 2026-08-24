#include "napoleon_joint_teacher.hpp"

#include "napoleon_observation.hpp"
#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace napoleon::joint_teacher {
namespace {

constexpr int kBiddingBidPositionCount = 28;
constexpr int kBiddingBidOwnerClassCount = kPlayerCount + 1;
constexpr int kMinBiddingTargetPointCards = 13;
constexpr double kEpsilon = 1e-9;

struct RolloutValue {
  double margin = 0.0;
  double relative_reward = 0.0;
  bool success = false;
  int napoleon_points = 0;
};

struct ExchangeEvaluation {
  std::vector<Card> discard;
  RolloutValue value;
  int candidate_index = 0;
};

struct AdjutantEvaluation {
  Card candidate;
  ExchangeEvaluation best_exchange;
  ExchangeEvaluation rb_exchange;
  ExchangeEvaluation approx_exchange;
  double exchange_spread = 0.0;
  bool approx_contains_gold = false;
  bool rb_exchange_is_gold = false;
};

struct SourceDiagnostic {
  std::uint32_t seed = 0;
  GameState choosing_state;
  Card rb_adjutant;
  AdjutantEvaluation best_adjutant;
  AdjutantEvaluation rb_adjutant_eval;
  AdjutantEvaluation approx_best_adjutant;
  int adjutant_candidate_count = 0;
  int terminal_rollout_count = 0;
  double adjutant_spread = 0.0;
  double top1_top3_gap = 0.0;
  bool approx_joint_matches_gold = false;
};

struct RunningStats {
  int count = 0;
  double sum = 0.0;
  double min = std::numeric_limits<double>::infinity();
  double max = -std::numeric_limits<double>::infinity();

  void add(double value) {
    ++count;
    sum += value;
    min = std::min(min, value);
    max = std::max(max, value);
  }

  double mean() const {
    return count == 0 ? 0.0 : sum / count;
  }
};

bool is_joker(Card card) {
  return card.id == 52;
}

Rank card_rank(Card card) {
  if (is_joker(card)) {
    throw std::runtime_error("joker has no rank");
  }
  return static_cast<Rank>(card.id % 13);
}

Suit card_suit(Card card) {
  if (is_joker(card)) {
    throw std::runtime_error("joker has no suit");
  }
  return static_cast<Suit>(card.id / 13);
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

int card_model_index(Card card) {
  return observation::playing_card_model_index(card);
}

int relative_player_index(int self_player_index, int player_index) {
  return (player_index - self_player_index + kPlayerCount) % kPlayerCount;
}

int bidding_bid_position_suit_index(Suit suit) {
  switch (suit) {
    case Suit::Clubs:
      return 0;
    case Suit::Diamonds:
      return 1;
    case Suit::Hearts:
      return 2;
    case Suit::Spades:
      return 3;
  }
  throw std::runtime_error("invalid bidding suit");
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

template <typename T>
void write_number_array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << values[index];
  }
  out << ']';
}

std::string cards_key(const std::vector<Card>& cards) {
  std::vector<std::string> ids;
  ids.reserve(cards.size());
  for (Card card : cards) {
    ids.push_back(card_id(card));
  }
  std::sort(ids.begin(), ids.end());
  std::ostringstream out;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      out << '|';
    }
    out << ids[index];
  }
  return out.str();
}

std::vector<std::string> card_ids(const std::vector<Card>& cards) {
  std::vector<std::string> ids;
  ids.reserve(cards.size());
  for (Card card : cards) {
    ids.push_back(card_id(card));
  }
  return ids;
}

void append_card_mask(std::vector<float>& values, const std::vector<Card>& cards) {
  std::array<float, observation::kCardCount> mask{};
  for (Card card : cards) {
    mask[static_cast<std::size_t>(card_model_index(card))] = 1.0F;
  }
  values.insert(values.end(), mask.begin(), mask.end());
}

void append_card_one_hot(std::vector<float>& values, Card card) {
  std::array<float, observation::kCardCount> mask{};
  mask[static_cast<std::size_t>(card_model_index(card))] = 1.0F;
  values.insert(values.end(), mask.begin(), mask.end());
}

void append_one_hot(std::vector<float>& values, int index, int count) {
  if (index < 0 || index >= count) {
    throw std::runtime_error("one-hot index out of range");
  }
  for (int candidate = 0; candidate < count; ++candidate) {
    values.push_back(candidate == index ? 1.0F : 0.0F);
  }
}

void append_bid_owner_table(std::vector<float>& values, const GameState& state, int self_player_index) {
  std::array<float, kBiddingBidPositionCount * kBiddingBidOwnerClassCount> table{};
  for (int position_index = 0; position_index < kBiddingBidPositionCount; ++position_index) {
    table[static_cast<std::size_t>(position_index * kBiddingBidOwnerClassCount)] = 1.0F;
  }

  int highest_seen_position = -1;
  for (const BiddingHistoryEntry& entry : state.public_bidding_history) {
    if (!entry.is_bid) {
      continue;
    }
    if (!entry.suit.has_value() || !entry.target_point_cards.has_value()) {
      throw std::runtime_error("bid history entry is incomplete");
    }
    const int target_offset = *entry.target_point_cards - kMinBiddingTargetPointCards;
    const int position_index =
        target_offset * 4 + bidding_bid_position_suit_index(*entry.suit);
    if (position_index < 0 || position_index >= kBiddingBidPositionCount) {
      throw std::runtime_error("bid history target out of compact range");
    }
    if (position_index <= highest_seen_position) {
      throw std::runtime_error("bid history positions must be strictly increasing");
    }
    highest_seen_position = position_index;

    const std::size_t offset =
        static_cast<std::size_t>(position_index * kBiddingBidOwnerClassCount);
    table[offset] = 0.0F;
    table[offset + static_cast<std::size_t>(relative_player_index(self_player_index, entry.player_index) + 1)] =
        1.0F;
  }
  values.insert(values.end(), table.begin(), table.end());
}

std::vector<float> create_adjutant_compact290_input(const GameState& state, Card candidate) {
  if (state.phase != Phase::ChoosingAdjutant || !state.contract.has_value()) {
    throw std::runtime_error("compact290 requires choosing-adjutant source state");
  }
  const int napoleon = state.contract->napoleon_player_index;
  std::vector<float> values;
  values.reserve(kAdjutantCompactValueInputFeatureCount);
  append_card_mask(values, state.hands[static_cast<std::size_t>(napoleon)]);
  append_one_hot(values, static_cast<int>(state.contract->trump_suit), 4);
  append_one_hot(values, state.contract->target_point_cards - kMinBiddingTargetPointCards, 7);
  append_one_hot(values, relative_player_index(napoleon, state.public_bidding_history.empty()
      ? 0
      : state.public_bidding_history.front().player_index), kPlayerCount);
  append_bid_owner_table(values, state, napoleon);
  append_card_one_hot(values, candidate);
  if (static_cast<int>(values.size()) != kAdjutantCompactValueInputFeatureCount) {
    throw std::runtime_error("compact290 feature count drift");
  }
  return values;
}

std::vector<float> create_exchange_compact396_input(
    const GameState& source,
    const GameState& exchange_state,
    const std::vector<Card>& discard) {
  if (source.phase != Phase::ChoosingAdjutant || exchange_state.phase != Phase::Exchanging ||
      !exchange_state.contract.has_value() || !exchange_state.adjutant.has_value()) {
    throw std::runtime_error("compact396 requires choosing source and exchange state");
  }
  const int napoleon = exchange_state.contract->napoleon_player_index;
  std::vector<Card> kitty;
  for (Card card : exchange_state.hands[static_cast<std::size_t>(napoleon)]) {
    const auto& original = source.hands[static_cast<std::size_t>(napoleon)];
    const bool in_original = std::any_of(original.begin(), original.end(), [&](Card original_card) {
      return original_card.id == card.id;
    });
    if (!in_original) {
      kitty.push_back(card);
    }
  }
  if (kitty.size() != 3) {
    throw std::runtime_error("compact396 kitty pickup reconstruction failed");
  }

  std::vector<float> values;
  values.reserve(kExchangeCompactValueInputFeatureCount);
  append_card_mask(values, source.hands[static_cast<std::size_t>(napoleon)]);
  append_card_mask(values, kitty);
  append_card_one_hot(values, exchange_state.adjutant->called_card);
  append_one_hot(values, static_cast<int>(exchange_state.contract->trump_suit), 4);
  append_one_hot(values, exchange_state.contract->target_point_cards - kMinBiddingTargetPointCards, 7);
  append_one_hot(values, relative_player_index(napoleon, source.public_bidding_history.empty()
      ? 0
      : source.public_bidding_history.front().player_index), kPlayerCount);
  append_bid_owner_table(values, source, napoleon);
  append_card_mask(values, discard);
  if (static_cast<int>(values.size()) != kExchangeCompactValueInputFeatureCount) {
    throw std::runtime_error("compact396 feature count drift");
  }
  return values;
}

double card_keep_score(Card card, Suit trump_suit, Card called_adjutant) {
  if (card.id == called_adjutant.id) {
    return 70.0;
  }
  if (is_joker(card)) {
    return 50.0;
  }
  if (is_oruma(card)) {
    return 80.0;
  }
  if (card.id == sei_jack(trump_suit).id) {
    return 75.0;
  }
  if (card.id == ura_jack(trump_suit).id) {
    return 72.0;
  }
  if (is_yoromeki(card)) {
    return card_suit(card) == trump_suit ? 65.0 : 25.0;
  }
  int rank_score = 0;
  switch (card_rank(card)) {
    case Rank::Ace:
      rank_score = 14;
      break;
    case Rank::King:
      rank_score = 13;
      break;
    case Rank::Queen:
      rank_score = 12;
      break;
    case Rank::Jack:
      rank_score = 11;
      break;
    case Rank::Ten:
      rank_score = 10;
      break;
    case Rank::Nine:
      rank_score = 9;
      break;
    case Rank::Eight:
      rank_score = 8;
      break;
    case Rank::Seven:
      rank_score = 7;
      break;
    case Rank::Six:
      rank_score = 6;
      break;
    case Rank::Five:
      rank_score = 5;
      break;
    case Rank::Four:
      rank_score = 4;
      break;
    case Rank::Three:
      rank_score = 3;
      break;
    case Rank::Two:
      rank_score = 8;
      break;
  }
  return static_cast<double>(rank_score) + (card_suit(card) == trump_suit ? 35.0 : 0.0);
}

double discard_proxy_score(
    const GameState& exchange_state,
    const std::vector<Card>& discard) {
  if (!exchange_state.contract.has_value() || !exchange_state.adjutant.has_value()) {
    throw std::runtime_error("discard proxy requires exchange state");
  }
  const Suit trump = exchange_state.contract->trump_suit;
  const Card called = exchange_state.adjutant->called_card;
  double score = 0.0;
  for (Card card : discard) {
    score -= card_keep_score(card, trump, called);
    if (is_point_card(card)) {
      score -= 8.0;
    }
  }
  return score;
}

void enumerate_discard_combinations_rec(
    const std::vector<Card>& hand,
    std::size_t start,
    std::vector<Card>& current,
    std::vector<std::vector<Card>>& out) {
  if (current.size() == 3) {
    out.push_back(current);
    return;
  }
  const std::size_t remaining = 3 - current.size();
  for (std::size_t index = start; index + remaining <= hand.size(); ++index) {
    current.push_back(hand[index]);
    enumerate_discard_combinations_rec(hand, index + 1, current, out);
    current.pop_back();
  }
}

std::vector<std::vector<Card>> enumerate_discard_combinations(const std::vector<Card>& hand) {
  std::vector<std::vector<Card>> combinations;
  std::vector<Card> current;
  enumerate_discard_combinations_rec(hand, 0, current, combinations);
  return combinations;
}

std::vector<std::vector<Card>> heuristic_top_k_discards(
    const GameState& exchange_state,
    int top_k) {
  const auto& hand = exchange_state.hands[static_cast<std::size_t>(
      exchange_state.contract->napoleon_player_index)];
  std::vector<std::vector<Card>> combinations = enumerate_discard_combinations(hand);
  std::sort(combinations.begin(), combinations.end(), [&](const auto& left, const auto& right) {
    const double left_score = discard_proxy_score(exchange_state, left);
    const double right_score = discard_proxy_score(exchange_state, right);
    if (std::fabs(left_score - right_score) > kEpsilon) {
      return left_score > right_score;
    }
    return cards_key(left) < cards_key(right);
  });
  if (top_k < static_cast<int>(combinations.size())) {
    combinations.resize(static_cast<std::size_t>(top_k));
  }
  return combinations;
}

Action discard_action(int player_index, const std::vector<Card>& cards) {
  Action action;
  action.type = Action::Type::DiscardCards;
  action.player_index = player_index;
  action.cards = cards;
  return action;
}

Action choose_adjutant_action(int player_index, Card card) {
  Action action;
  action.type = Action::Type::ChooseAdjutant;
  action.player_index = player_index;
  action.card = card;
  return action;
}

RolloutValue finish_with_rule_based_playing(GameState state, std::uint32_t seed) {
  SeededRandom rng(seed);
  int guard = 0;
  while (!state.is_game_over && state.phase != Phase::Finished) {
    if (++guard > 1000) {
      throw std::runtime_error("playing rollout did not terminate");
    }
    if (state.phase == Phase::Playing && state.is_trick_complete) {
      Action action;
      action.type = Action::Type::AdvanceToNextTrick;
      apply_action(state, action);
      continue;
    }
    if (state.phase != Phase::Playing) {
      throw std::runtime_error("rollout expected playing state");
    }
    const Action action = select_rule_based_action(state, state.current_player_index, rng);
    apply_action(state, action);
  }
  if (!state.result.has_value() || state.result->result_type != "standard") {
    throw std::runtime_error("standard result required");
  }
  const bool success = state.result->winner == "napoleon-team";
  return RolloutValue{
      static_cast<double>(state.result->napoleon_team_point_cards - state.result->target_point_cards),
      success ? (7.0 * state.result->target_point_cards) / 4.0
              : (-3.0 * state.result->target_point_cards) / 4.0,
      success,
      state.result->napoleon_team_point_cards};
}

bool better_value(const RolloutValue& left, const RolloutValue& right) {
  if (std::fabs(left.margin - right.margin) > kEpsilon) {
    return left.margin > right.margin;
  }
  if (std::fabs(left.relative_reward - right.relative_reward) > kEpsilon) {
    return left.relative_reward > right.relative_reward;
  }
  return left.napoleon_points > right.napoleon_points;
}

bool same_cards(const std::vector<Card>& left, const std::vector<Card>& right) {
  return cards_key(left) == cards_key(right);
}

ExchangeEvaluation evaluate_discard(
    const GameState& choosing_source,
    const GameState& exchange_state,
    const std::vector<Card>& discard,
    int candidate_index,
    std::uint32_t rollout_seed) {
  (void)create_exchange_compact396_input(choosing_source, exchange_state, discard);
  GameState playing = exchange_state;
  apply_action(playing, discard_action(playing.contract->napoleon_player_index, discard));
  return ExchangeEvaluation{discard, finish_with_rule_based_playing(playing, rollout_seed), candidate_index};
}

ExchangeEvaluation best_exchange_exhaustive(
    const GameState& choosing_source,
    const GameState& exchange_state,
    std::uint32_t rollout_seed,
    int& terminal_rollout_count,
    double& spread) {
  const auto& hand = exchange_state.hands[static_cast<std::size_t>(
      exchange_state.contract->napoleon_player_index)];
  const std::vector<std::vector<Card>> combinations = enumerate_discard_combinations(hand);
  if (static_cast<int>(combinations.size()) != kExchangeDiscardCombinationCount) {
    throw std::runtime_error("expected 286 discard candidates");
  }

  std::optional<ExchangeEvaluation> best;
  double min_margin = std::numeric_limits<double>::infinity();
  double max_margin = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < combinations.size(); ++index) {
    ExchangeEvaluation evaluated = evaluate_discard(
        choosing_source,
        exchange_state,
        combinations[index],
        static_cast<int>(index),
        rollout_seed + static_cast<std::uint32_t>(index));
    ++terminal_rollout_count;
    min_margin = std::min(min_margin, evaluated.value.margin);
    max_margin = std::max(max_margin, evaluated.value.margin);
    if (!best.has_value() || better_value(evaluated.value, best->value) ||
        (!better_value(best->value, evaluated.value) && cards_key(evaluated.discard) < cards_key(best->discard))) {
      best = evaluated;
    }
  }
  spread = max_margin - min_margin;
  return *best;
}

ExchangeEvaluation best_exchange_from_candidates(
    const GameState& choosing_source,
    const GameState& exchange_state,
    const std::vector<std::vector<Card>>& candidates,
    std::uint32_t rollout_seed,
    int& terminal_rollout_count) {
  std::optional<ExchangeEvaluation> best;
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    ExchangeEvaluation evaluated = evaluate_discard(
        choosing_source,
        exchange_state,
        candidates[index],
        static_cast<int>(index),
        rollout_seed + static_cast<std::uint32_t>(index));
    ++terminal_rollout_count;
    if (!best.has_value() || better_value(evaluated.value, best->value) ||
        (!better_value(best->value, evaluated.value) && cards_key(evaluated.discard) < cards_key(best->discard))) {
      best = evaluated;
    }
  }
  if (!best.has_value()) {
    throw std::runtime_error("no exchange candidates evaluated");
  }
  return *best;
}

std::vector<std::vector<Card>> unique_candidate_discards(
    std::vector<std::vector<Card>> candidates) {
  std::vector<std::vector<Card>> unique;
  std::unordered_set<std::string> seen;
  for (const auto& candidate : candidates) {
    if (seen.insert(cards_key(candidate)).second) {
      unique.push_back(candidate);
    }
  }
  return unique;
}

AdjutantEvaluation evaluate_adjutant_candidate(
    const GameState& choosing_source,
    Card adjutant,
    Card rb_adjutant,
    int heuristic_top_k,
    std::uint32_t rollout_seed,
    int& terminal_rollout_count) {
  (void)rb_adjutant;
  (void)create_adjutant_compact290_input(choosing_source, adjutant);
  const int napoleon = choosing_source.contract->napoleon_player_index;

  GameState exchange_state = choosing_source;
  apply_action(exchange_state, choose_adjutant_action(napoleon, adjutant));
  if (exchange_state.phase != Phase::Exchanging ||
      exchange_state.hands[static_cast<std::size_t>(napoleon)].size() != 13 ||
      !exchange_state.unused_cards.empty()) {
    throw std::runtime_error("adjutant -> kitty pickup -> exchanging invariant failed");
  }

  double exchange_spread = 0.0;
  ExchangeEvaluation gold = best_exchange_exhaustive(
      choosing_source,
      exchange_state,
      rollout_seed,
      terminal_rollout_count,
      exchange_spread);

  SeededRandom rb_rng(rollout_seed ^ 0x9e3779b9U);
  const Action rb_discard_action = select_rule_based_action(exchange_state, napoleon, rb_rng);
  if (rb_discard_action.type != Action::Type::DiscardCards ||
      rb_discard_action.cards.size() != 3) {
    throw std::runtime_error("RuleBased exchange did not return a 3-card discard");
  }
  ExchangeEvaluation rb = evaluate_discard(
      choosing_source,
      exchange_state,
      rb_discard_action.cards,
      -1,
      rollout_seed ^ 0xa511e9b3U);
  ++terminal_rollout_count;

  std::vector<std::vector<Card>> approx_candidates = heuristic_top_k_discards(exchange_state, heuristic_top_k);
  approx_candidates.push_back(rb_discard_action.cards);
  approx_candidates = unique_candidate_discards(std::move(approx_candidates));
  const bool approx_contains_gold = std::any_of(
      approx_candidates.begin(),
      approx_candidates.end(),
      [&](const auto& candidate) {
        return same_cards(candidate, gold.discard);
      });
  ExchangeEvaluation approx = best_exchange_from_candidates(
      choosing_source,
      exchange_state,
      approx_candidates,
      rollout_seed ^ 0x632be59bU,
      terminal_rollout_count);

  return AdjutantEvaluation{
      adjutant,
      gold,
      rb,
      approx,
      exchange_spread,
      approx_contains_gold,
      same_cards(rb.discard, gold.discard)};
}

GameState create_choosing_source_state(std::uint32_t seed, std::uint32_t agent_seed) {
  GameState state = create_initial_game(seed);
  SeededRandom rng(agent_seed);
  int guard = 0;
  while (state.phase == Phase::Bidding) {
    if (++guard > 200) {
      throw std::runtime_error("bidding did not terminate");
    }
    const Action action = select_rule_based_action(state, state.current_player_index, rng);
    apply_action(state, action);
  }
  return state;
}

std::vector<SourceDiagnostic> collect_sources_and_evaluate(const JointTeacherOptions& options) {
  if (options.requested_source_states <= 0 || options.max_deal_attempts <= 0 ||
      options.exhaustive_state_count <= 0 || options.heuristic_top_k <= 0) {
    throw std::runtime_error("joint teacher diagnostic counts must be positive");
  }
  if (options.exhaustive_state_count != options.requested_source_states) {
    throw std::runtime_error(
        "this diagnostic CLI currently requires --exhaustive-states to equal --states");
  }

  std::vector<SourceDiagnostic> diagnostics;
  for (int attempt = 0; attempt < options.max_deal_attempts &&
       static_cast<int>(diagnostics.size()) < options.requested_source_states; ++attempt) {
    const std::uint32_t seed = options.start_seed + static_cast<std::uint32_t>(attempt);
    GameState source = create_choosing_source_state(seed, options.agent_seed + static_cast<std::uint32_t>(attempt));
    if (source.phase != Phase::ChoosingAdjutant || !source.contract.has_value()) {
      continue;
    }

    SeededRandom rb_rng(options.agent_seed ^ seed);
    const Action rb_adjutant_action =
        select_rule_based_action(source, source.current_player_index, rb_rng);
    if (rb_adjutant_action.type != Action::Type::ChooseAdjutant) {
      throw std::runtime_error("RuleBased adjutant did not choose an adjutant card");
    }

    std::vector<AdjutantEvaluation> adjutants;
    int terminal_rollout_count = 0;
    int candidate_index = 0;
    for (Card candidate : create_deck()) {
      AdjutantEvaluation evaluated = evaluate_adjutant_candidate(
          source,
          candidate,
          rb_adjutant_action.card,
          options.heuristic_top_k,
          seed + static_cast<std::uint32_t>(candidate_index * 1009),
          terminal_rollout_count);
      adjutants.push_back(evaluated);
      ++candidate_index;
    }
    if (static_cast<int>(adjutants.size()) != kAdjutantCandidateCount) {
      throw std::runtime_error("expected 53 adjutant candidates");
    }

    std::sort(adjutants.begin(), adjutants.end(), [](const auto& left, const auto& right) {
      if (better_value(left.best_exchange.value, right.best_exchange.value)) {
        return true;
      }
      if (better_value(right.best_exchange.value, left.best_exchange.value)) {
        return false;
      }
      return card_id(left.candidate) < card_id(right.candidate);
    });

    const auto rb_it = std::find_if(adjutants.begin(), adjutants.end(), [&](const auto& item) {
      return item.candidate.id == rb_adjutant_action.card.id;
    });
    if (rb_it == adjutants.end()) {
      throw std::runtime_error("RuleBased adjutant missing from candidate evaluations");
    }
    const auto approx_it = std::max_element(adjutants.begin(), adjutants.end(), [](const auto& left, const auto& right) {
      if (better_value(left.approx_exchange.value, right.approx_exchange.value)) {
        return true;
      }
      if (better_value(right.approx_exchange.value, left.approx_exchange.value)) {
        return false;
      }
      return card_id(left.candidate) > card_id(right.candidate);
    });

    std::vector<double> adj_values;
    adj_values.reserve(adjutants.size());
    for (const auto& item : adjutants) {
      adj_values.push_back(item.best_exchange.value.margin);
    }
    const double adj_spread =
        *std::max_element(adj_values.begin(), adj_values.end()) -
        *std::min_element(adj_values.begin(), adj_values.end());
    const double top3_margin = adjutants.size() >= 3
        ? adjutants[2].best_exchange.value.margin
        : adjutants.back().best_exchange.value.margin;

    diagnostics.push_back(SourceDiagnostic{
        seed,
        source,
        rb_adjutant_action.card,
        adjutants.front(),
        *rb_it,
        *approx_it,
        static_cast<int>(adjutants.size()),
        terminal_rollout_count,
        adj_spread,
        adjutants.front().best_exchange.value.margin - top3_margin,
        approx_it->candidate.id == adjutants.front().candidate.id &&
            same_cards(approx_it->approx_exchange.discard, adjutants.front().best_exchange.discard)});
  }
  return diagnostics;
}

void write_value(std::ostream& out, const RolloutValue& value) {
  out << "{\"contractMargin\":" << value.margin
      << ",\"napoleonRelativeReward\":" << value.relative_reward
      << ",\"contractSuccess\":" << (value.success ? "true" : "false")
      << ",\"napoleonPointCards\":" << value.napoleon_points << '}';
}

void write_exchange(std::ostream& out, const ExchangeEvaluation& exchange) {
  out << "{\"discardCardIds\":";
  const std::vector<std::string> ids = card_ids(exchange.discard);
  out << '[';
  for (std::size_t index = 0; index < ids.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    json_escape(out, ids[index]);
  }
  out << "],\"candidateIndex\":" << exchange.candidate_index << ",\"value\":";
  write_value(out, exchange.value);
  out << '}';
}

void write_numeric_summary(std::ostream& out, const RunningStats& stats) {
  out << "{\"count\":" << stats.count
      << ",\"min\":" << (stats.count == 0 ? 0.0 : stats.min)
      << ",\"max\":" << (stats.count == 0 ? 0.0 : stats.max)
      << ",\"mean\":" << stats.mean() << '}';
}

std::string render_report(
    const JointTeacherOptions& options,
    const std::vector<SourceDiagnostic>& diagnostics,
    long long elapsed_ms) {
  RunningStats rb_regret;
  RunningStats exchange_only_gain;
  RunningStats adjutant_gain;
  RunningStats adjutant_spread;
  RunningStats top1_top3_gap;
  RunningStats exchange_spread;
  int terminal_rollouts = 0;
  int approx_joint_matches = 0;
  int approx_exchange_contains = 0;
  int rb_exchange_matches = 0;
  int rb_adjutant_matches = 0;
  std::unordered_set<std::string> best_adjutants;

  for (const SourceDiagnostic& diagnostic : diagnostics) {
    terminal_rollouts += diagnostic.terminal_rollout_count;
    if (diagnostic.approx_joint_matches_gold) {
      ++approx_joint_matches;
    }
    if (diagnostic.best_adjutant.approx_contains_gold) {
      ++approx_exchange_contains;
    }
    if (diagnostic.best_adjutant.rb_exchange_is_gold) {
      ++rb_exchange_matches;
    }
    if (diagnostic.best_adjutant.candidate.id == diagnostic.rb_adjutant.id) {
      ++rb_adjutant_matches;
    }
    best_adjutants.insert(card_id(diagnostic.best_adjutant.candidate));
    rb_regret.add(
        diagnostic.best_adjutant.best_exchange.value.margin -
        diagnostic.rb_adjutant_eval.best_exchange.value.margin);
    exchange_only_gain.add(
        diagnostic.rb_adjutant_eval.best_exchange.value.margin -
        diagnostic.rb_adjutant_eval.rb_exchange.value.margin);
    adjutant_gain.add(
        diagnostic.best_adjutant.best_exchange.value.margin -
        diagnostic.rb_adjutant_eval.best_exchange.value.margin);
    adjutant_spread.add(diagnostic.adjutant_spread);
    top1_top3_gap.add(diagnostic.top1_top3_gap);
    exchange_spread.add(diagnostic.best_adjutant.exchange_spread);
  }

  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "{\"issue\":444";
  out << ",\"teacherDefinition\":";
  out << joint_teacher_definition_json();
  out << ",\"runtimeOrder\":[\"choose-adjutant\",\"kitty-pickup\",\"discard-3\",\"playing\"]";
  out << ",\"forbiddenRuntimeAction\":\"no 53x286 joint action\"";
  out << ",\"compact290Audit\":";
  out << compact290_audit_json();
  out << ",\"compact396Reuse\":{\"stateFeatureCount\":" << kExchangeCompactStateFeatureCount
      << ",\"candidateDiscardMaskFeatureCount\":53,\"valueInputFeatureCount\":"
      << kExchangeCompactValueInputFeatureCount
      << ",\"source\":\"Issue #438 compact396 layout\"}";
  out << ",\"options\":{\"startSeed\":" << options.start_seed
      << ",\"requestedSourceStates\":" << options.requested_source_states
      << ",\"maxDealAttempts\":" << options.max_deal_attempts
      << ",\"exhaustiveStateCount\":" << options.exhaustive_state_count
      << ",\"heuristicTopK\":" << options.heuristic_top_k
      << ",\"agentSeed\":" << options.agent_seed << '}';
  const int gold_exhaustive_rollouts =
      static_cast<int>(diagnostics.size()) * kAdjutantCandidateCount *
      kExchangeDiscardCombinationCount;
  out << ",\"compute\":{\"sourceStateCount\":" << diagnostics.size()
      << ",\"exhaustiveGoldStates\":" << diagnostics.size()
      << ",\"adjutantCandidatesPerState\":53"
      << ",\"discardCandidatesPerAdjutant\":286"
      << ",\"goldExhaustiveRollouts\":" << gold_exhaustive_rollouts
      << ",\"totalTerminalRollouts\":" << terminal_rollouts
      << ",\"elapsedMs\":" << elapsed_ms
      << ",\"rolloutsPerSecond\":"
      << (elapsed_ms == 0 ? 0.0 : (1000.0 * terminal_rollouts) / elapsed_ms)
      << '}';
  out << ",\"goldApproximation\":{\"method\":\"RuleBased exchange plus compact396-proxy heuristic top-k rollouts\""
      << ",\"note\":\"The proxy uses the same compact396 candidate-value slot but does not load a PyTorch checkpoint; replace the scorer with compact396 checkpoint inference in the production teacher generator.\""
      << ",\"bestAdjutantExchangeGoldContainedRate\":"
      << (diagnostics.empty() ? 0.0 : static_cast<double>(approx_exchange_contains) / diagnostics.size())
      << ",\"jointTop1ExactMatchRate\":"
      << (diagnostics.empty() ? 0.0 : static_cast<double>(approx_joint_matches) / diagnostics.size())
      << ",\"ruleBasedExchangeGoldMatchRate\":"
      << (diagnostics.empty() ? 0.0 : static_cast<double>(rb_exchange_matches) / diagnostics.size())
      << '}';
  out << ",\"decomposition\":{\"rbAdjutantRbExchangeMarginMean\":";
  RunningStats rb_rb;
  RunningStats rb_opt;
  RunningStats opt_opt;
  for (const SourceDiagnostic& diagnostic : diagnostics) {
    rb_rb.add(diagnostic.rb_adjutant_eval.rb_exchange.value.margin);
    rb_opt.add(diagnostic.rb_adjutant_eval.best_exchange.value.margin);
    opt_opt.add(diagnostic.best_adjutant.best_exchange.value.margin);
  }
  out << rb_rb.mean()
      << ",\"rbAdjutantOptimizedExchangeMarginMean\":" << rb_opt.mean()
      << ",\"optimizedAdjutantOptimizedExchangeMarginMean\":" << opt_opt.mean()
      << ",\"exchangeOnlyGain\":";
  write_numeric_summary(out, exchange_only_gain);
  out << ",\"adjutantAfterExchangeGain\":";
  write_numeric_summary(out, adjutant_gain);
  out << '}';
  out << ",\"adjutant\":{\"ruleBasedMatchRate\":"
      << (diagnostics.empty() ? 0.0 : static_cast<double>(rb_adjutant_matches) / diagnostics.size())
      << ",\"ruleBasedRegret\":";
  write_numeric_summary(out, rb_regret);
  out << ",\"candidateValueSpread\":";
  write_numeric_summary(out, adjutant_spread);
  out << ",\"top1Top3Gap\":";
  write_numeric_summary(out, top1_top3_gap);
  out << ",\"bestCardVariation\":{\"sourceStateCount\":" << diagnostics.size()
      << ",\"uniqueBestAdjutantCount\":" << best_adjutants.size()
      << ",\"historyConsistentHiddenDealRepeatGroups\":0"
      << ",\"note\":\"This diagnostic keeps each accepted deal history-consistent and does not reshuffle after bidding; repeated fixed-hand groups are deferred to the production teacher dataset.\"}}";
  out << ",\"exchange\":{\"bestAdjutantExchangeValueSpread\":";
  write_numeric_summary(out, exchange_spread);
  out << '}';
  out << ",\"sourceDiagnostics\":[";
  for (std::size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const SourceDiagnostic& diagnostic = diagnostics[index];
    out << "{\"seed\":" << diagnostic.seed
        << ",\"napoleonSeatIndex\":" << diagnostic.choosing_state.contract->napoleon_player_index
        << ",\"contractSuit\":";
    json_escape(out, suit_id(diagnostic.choosing_state.contract->trump_suit));
    out << ",\"contractTarget\":" << diagnostic.choosing_state.contract->target_point_cards
        << ",\"candidateCount\":" << diagnostic.adjutant_candidate_count
        << ",\"ruleBasedAdjutant\":";
    json_escape(out, card_id(diagnostic.rb_adjutant));
    out << ",\"bestAdjutant\":";
    json_escape(out, card_id(diagnostic.best_adjutant.candidate));
    out << ",\"approxBestAdjutant\":";
    json_escape(out, card_id(diagnostic.approx_best_adjutant.candidate));
    out << ",\"rbAdjRbExchange\":";
    write_exchange(out, diagnostic.rb_adjutant_eval.rb_exchange);
    out << ",\"rbAdjBestExchange\":";
    write_exchange(out, diagnostic.rb_adjutant_eval.best_exchange);
    out << ",\"bestAdjBestExchange\":";
    write_exchange(out, diagnostic.best_adjutant.best_exchange);
    out << ",\"approxJointMatchesGold\":" << (diagnostic.approx_joint_matches_gold ? "true" : "false")
        << ",\"terminalRollouts\":" << diagnostic.terminal_rollout_count << '}';
  }
  out << "]}";
  return out.str();
}

constexpr std::uint32_t kStreamRequestMagic = 0x3151544aU;
constexpr std::uint32_t kStreamResponseMagic = 0x3153544aU;
constexpr std::uint32_t kStreamDoneMagic = 0x3044544aU;

struct StreamSourceDiagnostic {
  std::uint32_t seed = 0;
  int source_index = 0;
  int napoleon_index = 0;
  Suit contract_suit = Suit::Spades;
  int contract_target = 13;
  Card rule_based_adjutant;
  int rule_based_adjutant_index = 0;
  double rb_adj_rb_exchange_margin = 0.0;
  double rb_adj_optimized_exchange_margin = 0.0;
  double best_adjutant_margin = 0.0;
  int best_adjutant_index = 0;
  int terminal_rollouts = 0;
  int proposal_gold_containment_top4 = 0;
  int proposal_gold_containment_top8 = 0;
  int proposal_gold_containment_top16 = 0;
  int proposal_gold_containment_top32 = 0;
  int proposal_gold_containment_top64 = 0;
  double proposal_best_regret_sum = 0.0;
};

template <typename T>
void write_binary(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) {
    throw std::runtime_error("failed to write binary stream");
  }
}

template <typename T>
T read_binary(std::istream& in, const char* name) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error(std::string("failed to read ") + name);
  }
  return value;
}

void write_float_vector(std::ofstream& out, const std::vector<float>& values) {
  out.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write feature vector");
  }
}

void write_float_value(std::ofstream& out, float value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(float));
  if (!out) {
    throw std::runtime_error("failed to write label value");
  }
}

void write_uint32_value(std::ofstream& out, std::uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(std::uint32_t));
  if (!out) {
    throw std::runtime_error("failed to write uint32 value");
  }
}

void write_uint8_value(std::ofstream& out, std::uint8_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(std::uint8_t));
  if (!out) {
    throw std::runtime_error("failed to write uint8 value");
  }
}

std::vector<std::uint32_t> read_top_indices_response(
    std::istream& in,
    int source_index,
    int expected_top_k) {
  const std::uint32_t magic = read_binary<std::uint32_t>(in, "response magic");
  if (magic != kStreamResponseMagic) {
    throw std::runtime_error("invalid scorer response magic");
  }
  const std::uint32_t response_source_index =
      read_binary<std::uint32_t>(in, "response source index");
  const std::uint32_t adjutant_count =
      read_binary<std::uint32_t>(in, "response adjutant count");
  const std::uint32_t top_k = read_binary<std::uint32_t>(in, "response top-k");
  if (response_source_index != static_cast<std::uint32_t>(source_index) ||
      adjutant_count != kAdjutantCandidateCount ||
      top_k != static_cast<std::uint32_t>(expected_top_k)) {
    throw std::runtime_error("scorer response shape mismatch");
  }
  std::vector<std::uint32_t> indices(
      static_cast<std::size_t>(kAdjutantCandidateCount * expected_top_k));
  in.read(
      reinterpret_cast<char*>(indices.data()),
      static_cast<std::streamsize>(indices.size() * sizeof(std::uint32_t)));
  if (!in) {
    throw std::runtime_error("failed to read scorer top indices");
  }
  return indices;
}

void send_scorer_request(
    std::ostream& out,
    int source_index,
    const std::vector<float>& compact396_inputs) {
  write_binary(out, kStreamRequestMagic);
  write_binary(out, static_cast<std::uint32_t>(source_index));
  write_binary(out, static_cast<std::uint32_t>(kAdjutantCandidateCount));
  write_binary(out, static_cast<std::uint32_t>(kExchangeDiscardCombinationCount));
  write_binary(out, static_cast<std::uint32_t>(kExchangeCompactValueInputFeatureCount));
  out.write(
      reinterpret_cast<const char*>(compact396_inputs.data()),
      static_cast<std::streamsize>(compact396_inputs.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("failed to write scorer request matrix");
  }
  out.flush();
}

int find_discard_index(
    const std::vector<std::vector<Card>>& combinations,
    const std::vector<Card>& discard) {
  const std::string target = cards_key(discard);
  for (std::size_t index = 0; index < combinations.size(); ++index) {
    if (cards_key(combinations[index]) == target) {
      return static_cast<int>(index);
    }
  }
  throw std::runtime_error("discard combination not found");
}

std::vector<int> deterministic_diversity_indices(
    int state_index,
    int adjutant_index,
    const std::unordered_set<int>& excluded,
    int count) {
  std::vector<int> candidates;
  candidates.reserve(kExchangeDiscardCombinationCount);
  for (int index = 0; index < kExchangeDiscardCombinationCount; ++index) {
    if (excluded.count(index) == 0) {
      candidates.push_back(index);
    }
  }
  std::sort(candidates.begin(), candidates.end(), [&](int left, int right) {
    const std::uint32_t left_key =
        static_cast<std::uint32_t>((left + 1) * 2654435761U) ^
        static_cast<std::uint32_t>((state_index + 17) * 2246822519U) ^
        static_cast<std::uint32_t>((adjutant_index + 31) * 3266489917U);
    const std::uint32_t right_key =
        static_cast<std::uint32_t>((right + 1) * 2654435761U) ^
        static_cast<std::uint32_t>((state_index + 17) * 2246822519U) ^
        static_cast<std::uint32_t>((adjutant_index + 31) * 3266489917U);
    if (left_key != right_key) {
      return left_key < right_key;
    }
    return left < right;
  });
  if (count < static_cast<int>(candidates.size())) {
    candidates.resize(static_cast<std::size_t>(count));
  }
  return candidates;
}

std::vector<int> proposal_indices_for_adjutant(
    const std::vector<std::uint32_t>& top_indices,
    int adjutant_index,
    int scorer_top_k,
    int proposal_top_k,
    int rb_discard_index,
    int diversity_count,
    int state_index) {
  std::vector<int> proposal;
  std::unordered_set<int> seen;
  const int base = adjutant_index * scorer_top_k;
  for (int index = 0; index < std::min(proposal_top_k, scorer_top_k); ++index) {
    const int candidate = static_cast<int>(top_indices[static_cast<std::size_t>(base + index)]);
    if (candidate < 0 || candidate >= kExchangeDiscardCombinationCount) {
      throw std::runtime_error("scorer top index out of range");
    }
    if (seen.insert(candidate).second) {
      proposal.push_back(candidate);
    }
  }
  if (seen.insert(rb_discard_index).second) {
    proposal.push_back(rb_discard_index);
  }
  for (int candidate : deterministic_diversity_indices(
           state_index,
           adjutant_index,
           seen,
           diversity_count)) {
    if (seen.insert(candidate).second) {
      proposal.push_back(candidate);
    }
  }
  return proposal;
}

bool topk_contains(
    const std::vector<std::uint32_t>& top_indices,
    int adjutant_index,
    int scorer_top_k,
    int k,
    int target_index) {
  const int base = adjutant_index * scorer_top_k;
  for (int offset = 0; offset < std::min(k, scorer_top_k); ++offset) {
    if (static_cast<int>(top_indices[static_cast<std::size_t>(base + offset)]) == target_index) {
      return true;
    }
  }
  return false;
}

void write_stream_manifest(
    const std::filesystem::path& output_dir,
    const AdjutantValueStreamOptions& options,
    const std::vector<StreamSourceDiagnostic>& diagnostics,
    int sample_count,
    int terminal_rollout_count) {
  std::ostringstream out;
  out << "{\"datasetSchemaVersion\":1"
      << ",\"sampleType\":\"adjutant-joint-value-v1\""
      << ",\"teacherId\":\"issue446-compact396-proposal-joint-teacher-v1\""
      << ",\"mode\":";
  json_escape(out, options.mode);
  out << ",\"featureCount\":" << kAdjutantCompactValueInputFeatureCount
      << ",\"stateFeatureCount\":" << kAdjutantCompactStateFeatureCount
      << ",\"candidateCountPerState\":" << kAdjutantCandidateCount
      << ",\"sourceStateCount\":" << diagnostics.size()
      << ",\"sampleCount\":" << sample_count
      << ",\"terminalRolloutCount\":" << terminal_rollout_count
      << ",\"runtimeOrder\":[\"adjutant\",\"kitty-pickup\",\"exchange\",\"playing\"]"
      << ",\"compact290Audit\":" << compact290_audit_json()
      << ",\"proposal\":{\"compact396TopK\":" << options.proposal_top_k
      << ",\"scorerReturnedTopK\":" << options.scorer_top_k
      << ",\"ruleBasedExchange\":true"
      << ",\"diversityRandomCount\":" << options.diversity_count << "}"
      << ",\"files\":{\"features\":\"features.f32\",\"contractMargin\":\"contract-margin.f32\","
      << "\"relativeReward\":\"relative-reward.f32\",\"stateIndex\":\"state-index.u32\","
      << "\"candidateCard\":\"candidate-card.u8\"}"
      << ",\"sourceDiagnostics\":[";
  for (std::size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const StreamSourceDiagnostic& diagnostic = diagnostics[index];
    out << "{\"sourceIndex\":" << diagnostic.source_index
        << ",\"seed\":" << diagnostic.seed
        << ",\"napoleonSeatIndex\":" << diagnostic.napoleon_index
        << ",\"contractSuit\":";
    json_escape(out, suit_id(diagnostic.contract_suit));
    out << ",\"contractTarget\":" << diagnostic.contract_target
        << ",\"ruleBasedAdjutant\":";
    json_escape(out, card_id(diagnostic.rule_based_adjutant));
    out << ",\"ruleBasedAdjutantIndex\":" << diagnostic.rule_based_adjutant_index
        << ",\"rbAdjRbExchangeMargin\":" << diagnostic.rb_adj_rb_exchange_margin
        << ",\"rbAdjOptimizedExchangeMargin\":" << diagnostic.rb_adj_optimized_exchange_margin
        << ",\"bestAdjutantIndex\":" << diagnostic.best_adjutant_index
        << ",\"bestAdjutantMargin\":" << diagnostic.best_adjutant_margin
        << ",\"terminalRollouts\":" << diagnostic.terminal_rollouts
        << ",\"proposalGoldContainment\":{\"top4\":" << diagnostic.proposal_gold_containment_top4
        << ",\"top8\":" << diagnostic.proposal_gold_containment_top8
        << ",\"top16\":" << diagnostic.proposal_gold_containment_top16
        << ",\"top32\":" << diagnostic.proposal_gold_containment_top32
        << ",\"top64\":" << diagnostic.proposal_gold_containment_top64 << "}"
        << ",\"proposalBestRegretSum\":" << diagnostic.proposal_best_regret_sum
        << '}';
  }
  out << "]}";

  std::ofstream manifest(output_dir / "manifest.json");
  if (!manifest) {
    throw std::runtime_error("failed to open stream manifest");
  }
  manifest << out.str() << '\n';
}

AdjutantValueStreamReport run_stream_teacher_impl(
    const AdjutantValueStreamOptions& options,
    std::istream& scorer_response,
    std::ostream& scorer_request) {
  if (options.mode != "proposal" && options.mode != "full-gold") {
    throw std::runtime_error("--mode must be proposal or full-gold");
  }
  if (options.output_directory.empty()) {
    throw std::runtime_error("output directory is required");
  }
  if (options.requested_source_states <= 0 || options.max_deal_attempts <= 0 ||
      options.proposal_top_k <= 0 || options.diversity_count < 0 ||
      options.scorer_top_k < options.proposal_top_k || options.scorer_top_k > 286) {
    throw std::runtime_error("invalid stream teacher options");
  }

  const std::filesystem::path output_dir(options.output_directory);
  std::filesystem::create_directories(output_dir);
  std::ofstream features(output_dir / "features.f32", std::ios::binary);
  std::ofstream margins(output_dir / "contract-margin.f32", std::ios::binary);
  std::ofstream rewards(output_dir / "relative-reward.f32", std::ios::binary);
  std::ofstream state_indices(output_dir / "state-index.u32", std::ios::binary);
  std::ofstream candidate_cards(output_dir / "candidate-card.u8", std::ios::binary);
  if (!features || !margins || !rewards || !state_indices || !candidate_cards) {
    throw std::runtime_error("failed to open stream dataset output files");
  }

  std::vector<StreamSourceDiagnostic> diagnostics;
  int sample_count = 0;
  int terminal_rollout_count = 0;

  for (int attempt = 0; attempt < options.max_deal_attempts &&
       static_cast<int>(diagnostics.size()) < options.requested_source_states; ++attempt) {
    const std::uint32_t seed = options.start_seed + static_cast<std::uint32_t>(attempt);
    GameState source = create_choosing_source_state(
        seed,
        options.agent_seed + static_cast<std::uint32_t>(attempt));
    if (source.phase != Phase::ChoosingAdjutant || !source.contract.has_value()) {
      continue;
    }
    const int source_index = static_cast<int>(diagnostics.size());
    const int napoleon = source.contract->napoleon_player_index;
    SeededRandom rb_adj_rng(options.agent_seed ^ seed);
    const Action rb_adjutant_action =
        select_rule_based_action(source, source.current_player_index, rb_adj_rng);
    if (rb_adjutant_action.type != Action::Type::ChooseAdjutant) {
      throw std::runtime_error("RuleBased adjutant selection failed");
    }
    const int rb_adjutant_index = static_cast<int>(rb_adjutant_action.card.id);

    std::vector<GameState> exchange_states;
    std::vector<std::vector<std::vector<Card>>> discard_combinations_by_adjutant;
    std::vector<float> compact396_inputs;
    compact396_inputs.reserve(
        static_cast<std::size_t>(
            kAdjutantCandidateCount * kExchangeDiscardCombinationCount *
            kExchangeCompactValueInputFeatureCount));

    for (Card adjutant : create_deck()) {
      GameState exchange_state = source;
      apply_action(exchange_state, choose_adjutant_action(napoleon, adjutant));
      const auto& hand = exchange_state.hands[static_cast<std::size_t>(napoleon)];
      if (exchange_state.phase != Phase::Exchanging || hand.size() != 13) {
        throw std::runtime_error("adjutant fixed state did not reach exchange");
      }
      std::vector<std::vector<Card>> combinations = enumerate_discard_combinations(hand);
      if (static_cast<int>(combinations.size()) != kExchangeDiscardCombinationCount) {
        throw std::runtime_error("expected 286 exchange candidates in stream teacher");
      }
      for (const std::vector<Card>& discard : combinations) {
        std::vector<float> input = create_exchange_compact396_input(source, exchange_state, discard);
        compact396_inputs.insert(compact396_inputs.end(), input.begin(), input.end());
      }
      exchange_states.push_back(exchange_state);
      discard_combinations_by_adjutant.push_back(std::move(combinations));
    }

    send_scorer_request(scorer_request, source_index, compact396_inputs);
    const std::vector<std::uint32_t> top_indices =
        read_top_indices_response(scorer_response, source_index, options.scorer_top_k);

    StreamSourceDiagnostic diagnostic;
    diagnostic.seed = seed;
    diagnostic.source_index = source_index;
    diagnostic.napoleon_index = napoleon;
    diagnostic.contract_suit = source.contract->trump_suit;
    diagnostic.contract_target = source.contract->target_point_cards;
    diagnostic.rule_based_adjutant = rb_adjutant_action.card;
    diagnostic.rule_based_adjutant_index = rb_adjutant_index;
    diagnostic.best_adjutant_margin = -std::numeric_limits<double>::infinity();

    for (int adjutant_index = 0; adjutant_index < kAdjutantCandidateCount; ++adjutant_index) {
      Card adjutant{static_cast<std::uint8_t>(adjutant_index)};
      const GameState& exchange_state = exchange_states[static_cast<std::size_t>(adjutant_index)];
      const auto& combinations =
          discard_combinations_by_adjutant[static_cast<std::size_t>(adjutant_index)];

      SeededRandom rb_ex_rng(seed ^ static_cast<std::uint32_t>(adjutant_index * 7919));
      const Action rb_discard_action = select_rule_based_action(exchange_state, napoleon, rb_ex_rng);
      if (rb_discard_action.type != Action::Type::DiscardCards) {
        throw std::runtime_error("RuleBased exchange selection failed");
      }
      const int rb_discard_index = find_discard_index(combinations, rb_discard_action.cards);
      const std::vector<int> proposal_indices = proposal_indices_for_adjutant(
          top_indices,
          adjutant_index,
          options.scorer_top_k,
          options.proposal_top_k,
          rb_discard_index,
          options.diversity_count,
          source_index);

      std::optional<ExchangeEvaluation> gold_best;
      std::optional<ExchangeEvaluation> proposal_best;
      std::optional<ExchangeEvaluation> rb_exchange_eval;
      double exchange_spread = 0.0;

      if (options.mode == "full-gold") {
        gold_best = best_exchange_exhaustive(
            source,
            exchange_state,
            seed + static_cast<std::uint32_t>(adjutant_index * 1009),
            terminal_rollout_count,
            exchange_spread);
        diagnostic.terminal_rollouts += kExchangeDiscardCombinationCount;
        const int gold_index = gold_best->candidate_index;
        diagnostic.proposal_gold_containment_top4 +=
            topk_contains(top_indices, adjutant_index, options.scorer_top_k, 4, gold_index) ? 1 : 0;
        diagnostic.proposal_gold_containment_top8 +=
            topk_contains(top_indices, adjutant_index, options.scorer_top_k, 8, gold_index) ? 1 : 0;
        diagnostic.proposal_gold_containment_top16 +=
            topk_contains(top_indices, adjutant_index, options.scorer_top_k, 16, gold_index) ? 1 : 0;
        diagnostic.proposal_gold_containment_top32 +=
            topk_contains(top_indices, adjutant_index, options.scorer_top_k, 32, gold_index) ? 1 : 0;
        diagnostic.proposal_gold_containment_top64 +=
            topk_contains(top_indices, adjutant_index, options.scorer_top_k, 64, gold_index) ? 1 : 0;
      }

      proposal_best = best_exchange_from_candidates(
          source,
          exchange_state,
          [&]() {
            std::vector<std::vector<Card>> selected;
            selected.reserve(proposal_indices.size());
            for (int index : proposal_indices) {
              selected.push_back(combinations[static_cast<std::size_t>(index)]);
            }
            return selected;
          }(),
          seed ^ static_cast<std::uint32_t>(adjutant_index * 3571 + 17),
          terminal_rollout_count);
      diagnostic.terminal_rollouts += static_cast<int>(proposal_indices.size());

      rb_exchange_eval = evaluate_discard(
          source,
          exchange_state,
          rb_discard_action.cards,
          rb_discard_index,
          seed ^ static_cast<std::uint32_t>(adjutant_index * 3571 + 23));
      ++terminal_rollout_count;
      ++diagnostic.terminal_rollouts;

      const RolloutValue label =
          options.mode == "full-gold" ? gold_best->value : proposal_best->value;
      if (options.mode == "full-gold") {
        diagnostic.proposal_best_regret_sum +=
            gold_best->value.margin - proposal_best->value.margin;
      }

      if (adjutant_index == rb_adjutant_index) {
        diagnostic.rb_adj_rb_exchange_margin = rb_exchange_eval->value.margin;
        diagnostic.rb_adj_optimized_exchange_margin =
            options.mode == "full-gold" ? gold_best->value.margin : proposal_best->value.margin;
      }
      if (label.margin > diagnostic.best_adjutant_margin + kEpsilon) {
        diagnostic.best_adjutant_margin = label.margin;
        diagnostic.best_adjutant_index = adjutant_index;
      }

      const std::vector<float> compact290 = create_adjutant_compact290_input(source, adjutant);
      write_float_vector(features, compact290);
      write_float_value(margins, static_cast<float>(label.margin));
      write_float_value(rewards, static_cast<float>(label.relative_reward));
      write_uint32_value(state_indices, static_cast<std::uint32_t>(source_index));
      write_uint8_value(candidate_cards, static_cast<std::uint8_t>(adjutant_index));
      ++sample_count;
    }
    diagnostics.push_back(diagnostic);
    std::cerr << "[adjutant-stream] mode=" << options.mode
              << " states=" << diagnostics.size() << "/" << options.requested_source_states
              << " samples=" << sample_count
              << " terminalRollouts=" << terminal_rollout_count << '\n';
  }

  write_binary(scorer_request, kStreamDoneMagic);
  scorer_request.flush();
  write_stream_manifest(
      output_dir,
      options,
      diagnostics,
      sample_count,
      terminal_rollout_count);
  std::ifstream manifest_in(output_dir / "manifest.json");
  std::ostringstream manifest_buffer;
  manifest_buffer << manifest_in.rdbuf();
  return AdjutantValueStreamReport{
      static_cast<int>(diagnostics.size()),
      sample_count,
      terminal_rollout_count,
      manifest_buffer.str()};
}

}  // namespace

JointTeacherReport run_joint_teacher_diagnostic(const JointTeacherOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  const std::vector<SourceDiagnostic> diagnostics = collect_sources_and_evaluate(options);
  const auto end = std::chrono::steady_clock::now();
  const long long elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  int terminal_rollouts = 0;
  for (const SourceDiagnostic& diagnostic : diagnostics) {
    terminal_rollouts += diagnostic.terminal_rollout_count;
  }
  return JointTeacherReport{
      render_report(options, diagnostics, elapsed_ms),
      static_cast<int>(diagnostics.size()),
      static_cast<int>(diagnostics.size()),
      terminal_rollouts};
}

AdjutantValueStreamReport run_adjutant_value_stream_teacher(
    const AdjutantValueStreamOptions& options,
    std::istream& scorer_response,
    std::ostream& scorer_request) {
  return run_stream_teacher_impl(options, scorer_response, scorer_request);
}

std::string compact290_audit_json() {
  std::ostringstream out;
  out << "{\"featureCount\":" << kAdjutantCompactValueInputFeatureCount
      << ",\"stateFeatureCount\":" << kAdjutantCompactStateFeatureCount
      << ",\"layout\":["
      << "{\"name\":\"originalHandMask53\",\"start\":0,\"stop\":53,\"knownAtAdjutantChoice\":true},"
      << "{\"name\":\"contractSuit4\",\"start\":53,\"stop\":57,\"knownAtAdjutantChoice\":true},"
      << "{\"name\":\"contractTarget7\",\"start\":57,\"stop\":64,\"knownAtAdjutantChoice\":true},"
      << "{\"name\":\"biddingStarterRelative5\",\"start\":64,\"stop\":69,\"knownAtAdjutantChoice\":true},"
      << "{\"name\":\"bidOwnerTable168\",\"start\":69,\"stop\":237,\"knownAtAdjutantChoice\":true},"
      << "{\"name\":\"candidateAdjutantCard53\",\"start\":237,\"stop\":290,\"knownAtAdjutantChoice\":true}"
      << "],\"excludedKnownLater\":[\"kittyPickup3\",\"pickupHand13\",\"discardCandidate\",\"adjutantOwnerAfterCall\"],"
      << "\"audit\":\"No known adjutant-choice information needed by the proposed compact state is missing; kitty is intentionally absent.\"}";
  return out.str();
}

std::string joint_teacher_definition_json() {
  return "{\"exchange\":\"Q_exchange(exchangeVisibleStateCompact343, discardMask53) -> downstream value after discard then playing\","
         "\"adjutant\":\"Q_adjutant(adjutantVisibleStateCompact237, adjutantCard53) = E[ value after adjutant fixed -> kitty pickup -> best/high-quality exchange -> playing ]\","
         "\"runtime\":\"sequential adjutant action, automatic kitty pickup, sequential exchange action, then playing\","
         "\"notRuntimeJointAction\":true}";
}

}  // namespace napoleon::joint_teacher
