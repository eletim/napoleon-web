#include "napoleon_observation.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace napoleon::observation {
namespace {

constexpr int kEmptyCardIndex = -1;
constexpr int kEmptyPlayerIndex = -1;
constexpr int kEmptyBiddingActionType = -1;
constexpr int kEmptyBiddingSuitIndex = -1;
constexpr int kBiddingActionTypePass = 0;
constexpr int kBiddingActionTypeBid = 1;
constexpr int kMinBiddingTargetPointCards = 13;
constexpr int kBiddingTargetPointCardsClassCount = 7;
constexpr int kConsecutivePassCountClassCount = kPlayerCount + 1;
constexpr int kCompletedTrickCardSlotCount = kTrickCount * kCardsPerTrick;
constexpr int kSelfRoleCount = 4;

std::string player_id(int player_index) {
  return "player-" + std::to_string(player_index);
}

bool is_joker(Card card) {
  return card.id == 52;
}

Rank card_rank(Card card) {
  if (is_joker(card)) {
    throw std::runtime_error("joker has no rank");
  }
  return static_cast<Rank>(card.id % 13);
}

bool is_point_card(Card card) {
  if (is_joker(card)) {
    return false;
  }
  const Rank rank = card_rank(card);
  return rank == Rank::Ten || rank == Rank::Jack || rank == Rank::Queen ||
         rank == Rank::King || rank == Rank::Ace;
}

int card_model_index(Card card) {
  if (is_joker(card)) {
    return 52;
  }

  int rank_index = 0;
  switch (card_rank(card)) {
    case Rank::Ace:
      rank_index = 0;
      break;
    case Rank::King:
      rank_index = 1;
      break;
    case Rank::Queen:
      rank_index = 2;
      break;
    case Rank::Jack:
      rank_index = 3;
      break;
    case Rank::Ten:
      rank_index = 4;
      break;
    case Rank::Nine:
      rank_index = 5;
      break;
    case Rank::Eight:
      rank_index = 6;
      break;
    case Rank::Seven:
      rank_index = 7;
      break;
    case Rank::Six:
      rank_index = 8;
      break;
    case Rank::Five:
      rank_index = 9;
      break;
    case Rank::Four:
      rank_index = 10;
      break;
    case Rank::Three:
      rank_index = 11;
      break;
    case Rank::Two:
      rank_index = 12;
      break;
  }

  return static_cast<int>(card.id / 13) * 13 + rank_index;
}

Card card_from_model_index(int index) {
  if (index < 0 || index >= kCardCount) {
    throw std::runtime_error("card model index out of range");
  }
  if (index == 52) {
    return Card{52};
  }

  const int suit_index = index / 13;
  const int rank_model_index = index % 13;
  int core_rank_index = 0;
  switch (rank_model_index) {
    case 0:
      core_rank_index = static_cast<int>(Rank::Ace);
      break;
    case 1:
      core_rank_index = static_cast<int>(Rank::King);
      break;
    case 2:
      core_rank_index = static_cast<int>(Rank::Queen);
      break;
    case 3:
      core_rank_index = static_cast<int>(Rank::Jack);
      break;
    case 4:
      core_rank_index = static_cast<int>(Rank::Ten);
      break;
    case 5:
      core_rank_index = static_cast<int>(Rank::Nine);
      break;
    case 6:
      core_rank_index = static_cast<int>(Rank::Eight);
      break;
    case 7:
      core_rank_index = static_cast<int>(Rank::Seven);
      break;
    case 8:
      core_rank_index = static_cast<int>(Rank::Six);
      break;
    case 9:
      core_rank_index = static_cast<int>(Rank::Five);
      break;
    case 10:
      core_rank_index = static_cast<int>(Rank::Four);
      break;
    case 11:
      core_rank_index = static_cast<int>(Rank::Three);
      break;
    case 12:
      core_rank_index = static_cast<int>(Rank::Two);
      break;
    default:
      throw std::runtime_error("card model index out of range");
  }

  return Card{static_cast<std::uint8_t>(suit_index * 13 + core_rank_index)};
}

int relative_player_index(int self_player_index, int player_index) {
  return (player_index - self_player_index + kPlayerCount) % kPlayerCount;
}

int absolute_player_index(int self_player_index, int relative_index) {
  return (self_player_index + relative_index) % kPlayerCount;
}

int bidding_suit_index(Suit suit) {
  return static_cast<int>(suit);
}

Card sei_jack_card(Suit trump_suit) {
  return Card{static_cast<std::uint8_t>(static_cast<int>(trump_suit) * 13 + 10)};
}

Card ura_jack_card(Suit trump_suit) {
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

void set_card_mask(std::array<int, kCardCount>& mask, Card card) {
  mask[static_cast<std::size_t>(card_model_index(card))] = 1;
}

template <std::size_t Size>
void append_values(std::vector<float>& target, const std::array<int, Size>& values) {
  for (int value : values) {
    target.push_back(static_cast<float>(value));
  }
}

template <std::size_t OuterSize, std::size_t InnerSize>
void append_matrix(
    std::vector<float>& target,
    const std::array<std::array<int, InnerSize>, OuterSize>& values) {
  for (const auto& row : values) {
    append_values(target, row);
  }
}

template <typename Iterator, std::size_t EmptySize>
void append_one_hot_range(
    std::vector<float>& target,
    Iterator begin,
    Iterator end,
    int slot_count,
    int class_count,
    int min_value,
    const std::array<int, EmptySize>& empty_values) {
  if (static_cast<int>(std::distance(begin, end)) != slot_count) {
    throw std::runtime_error("one-hot index field has an invalid slot count");
  }

  for (Iterator it = begin; it != end; ++it) {
    const int index_value = *it;
    const int class_index = index_value - min_value;
    const bool is_empty =
        std::find(empty_values.begin(), empty_values.end(), index_value) != empty_values.end();
    if ((class_index < 0 || class_index >= class_count) && !is_empty) {
      throw std::runtime_error("one-hot index out of range");
    }

    for (int candidate = 0; candidate < class_count; ++candidate) {
      target.push_back(class_index == candidate ? 1.0F : 0.0F);
    }
  }
}

template <std::size_t Size, std::size_t EmptySize>
void append_one_hot_array(
    std::vector<float>& target,
    const std::array<int, Size>& indices,
    int class_count,
    int min_value,
    const std::array<int, EmptySize>& empty_values) {
  append_one_hot_range(
      target,
      indices.begin(),
      indices.end(),
      static_cast<int>(indices.size()),
      class_count,
      min_value,
      empty_values);
}

template <std::size_t EmptySize>
void append_one_hot_value(
    std::vector<float>& target,
    int index,
    int class_count,
    int min_value,
    const std::array<int, EmptySize>& empty_values) {
  const std::array<int, 1> indices{index};
  append_one_hot_array(target, indices, class_count, min_value, empty_values);
}

EncodedBiddingHistory encode_bidding_history(
    const GameState& state,
    const std::array<int, kPlayerCount>& relative_player_indices) {
  if (state.public_bidding_history.size() > kMaxBiddingActionCount) {
    throw std::runtime_error("bidding history exceeds model input capacity");
  }

  EncodedBiddingHistory history;
  history.action_type_indices.fill(kEmptyBiddingActionType);
  history.player_indices.fill(kEmptyPlayerIndex);
  history.suit_indices.fill(kEmptyBiddingSuitIndex);
  history.target_point_cards.fill(0);
  history.action_mask.fill(0);

  for (std::size_t index = 0; index < state.public_bidding_history.size(); ++index) {
    const BiddingHistoryEntry& entry = state.public_bidding_history[index];
    const auto relative_position = std::find(
        relative_player_indices.begin(), relative_player_indices.end(), entry.player_index);
    if (relative_position == relative_player_indices.end()) {
      throw std::runtime_error("bidding history player is not in relative player order");
    }

    history.action_type_indices[index] =
        entry.is_bid ? kBiddingActionTypeBid : kBiddingActionTypePass;
    history.player_indices[index] = static_cast<int>(
        std::distance(relative_player_indices.begin(), relative_position));
    history.action_mask[index] = 1;

    if (entry.is_bid) {
      history.suit_indices[index] = bidding_suit_index(*entry.suit);
      history.target_point_cards[index] = *entry.target_point_cards;
    }
  }

  return history;
}

std::array<float, kPlayingModelInputFeatureCount> encode_model_input(
    const EncodedPlayingObservation& observation) {
  std::vector<float> values;
  values.reserve(kPlayingModelInputFeatureCount);

  append_values(values, observation.trump_suit_one_hot);
  append_values(values, observation.napoleon_player_one_hot);
  append_values(values, observation.revealed_adjutant_player_one_hot);
  append_values(values, observation.called_adjutant_card_mask);
  append_values(values, observation.self_hand_mask);
  append_values(values, observation.legal_play_mask);
  append_values(values, observation.hand_count_by_player);
  append_matrix(values, observation.captured_point_card_mask_by_player);
  append_values(values, observation.current_trick_slot_mask);
  append_values(values, observation.completed_trick_slot_mask);
  append_values(values, observation.completed_trick_mask);
  append_values(values, observation.bidding_history.action_mask);
  append_values(values, observation.latest_buried_event_point_card_mask);
  values.push_back(static_cast<float>(observation.trick_number));
  values.push_back(static_cast<float>(observation.completed_trick_count));
  values.push_back(static_cast<float>(observation.contract_target_point_cards));
  values.push_back(static_cast<float>(observation.latest_buried_event_hidden_non_point_count));
  values.push_back(static_cast<float>(observation.latest_buried_event_present));

  if (static_cast<int>(values.size()) != kFlatObservationFeatureCount) {
    throw std::runtime_error("flat observation feature count drift");
  }

  append_one_hot_array(
      values,
      observation.special_card_indices,
      kCardCount,
      0,
      std::array<int, 1>{kEmptyCardIndex});
  append_one_hot_array(
      values,
      observation.current_trick_card_indices,
      kCardCount,
      0,
      std::array<int, 1>{kEmptyCardIndex});
  append_one_hot_array(
      values,
      observation.completed_trick_card_indices,
      kCardCount,
      0,
      std::array<int, 1>{kEmptyCardIndex});
  append_one_hot_array(
      values,
      observation.current_trick_player_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_array(
      values,
      observation.completed_trick_player_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_array(
      values,
      observation.completed_trick_winner_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_array(
      values,
      observation.bidding_history.action_type_indices,
      2,
      0,
      std::array<int, 1>{kEmptyBiddingActionType});
  append_one_hot_array(
      values,
      observation.bidding_history.player_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_array(
      values,
      observation.bidding_history.suit_indices,
      4,
      0,
      std::array<int, 1>{kEmptyBiddingSuitIndex});
  append_one_hot_array(
      values,
      observation.bidding_history.target_point_cards,
      kBiddingTargetPointCardsClassCount,
      kMinBiddingTargetPointCards,
      std::array<int, 1>{0});
  append_values(values, observation.self_role_one_hot);

  if (static_cast<int>(values.size()) != kPlayingModelInputFeatureCount) {
    throw std::runtime_error("playing model input feature count drift");
  }

  std::array<float, kPlayingModelInputFeatureCount> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

std::array<float, kBiddingModelInputFeatureCount> encode_bidding_model_input(
    const BiddingModelInput& input,
    const EncodedBiddingHistory& bidding_history,
    const GameState& state) {
  std::vector<float> values;
  values.reserve(kBiddingModelInputFeatureCount);

  const int player_index = input.player_index;
  std::array<int, kCardCount> self_hand_mask{};
  for (Card card : state.hands[static_cast<std::size_t>(player_index)]) {
    set_card_mask(self_hand_mask, card);
  }

  append_values(values, self_hand_mask);
  append_values(values, input.legal_bid_mask);
  append_one_hot_value(
      values,
      relative_player_index(player_index, state.bidding->starter_player_index),
      kPlayerCount,
      0,
      std::array<int, 0>{});

  const bool has_highest_bid = state.bidding->highest_bid.has_value();
  values.push_back(has_highest_bid ? 1.0F : 0.0F);
  append_one_hot_value(
      values,
      has_highest_bid ? relative_player_index(player_index, state.bidding->highest_bid->player_index)
                      : kEmptyPlayerIndex,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_value(
      values,
      has_highest_bid ? bidding_suit_index(state.bidding->highest_bid->suit)
                      : kEmptyBiddingSuitIndex,
      4,
      0,
      std::array<int, 1>{kEmptyBiddingSuitIndex});
  append_one_hot_value(
      values,
      has_highest_bid ? state.bidding->highest_bid->target_point_cards : 0,
      kBiddingTargetPointCardsClassCount,
      kMinBiddingTargetPointCards,
      std::array<int, 1>{0});
  append_one_hot_value(
      values,
      state.bidding->consecutive_pass_count,
      kConsecutivePassCountClassCount,
      0,
      std::array<int, 0>{});

  append_values(values, bidding_history.action_mask);
  append_one_hot_array(
      values,
      bidding_history.action_type_indices,
      2,
      0,
      std::array<int, 1>{kEmptyBiddingActionType});
  append_one_hot_array(
      values,
      bidding_history.player_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  append_one_hot_array(
      values,
      bidding_history.suit_indices,
      4,
      0,
      std::array<int, 1>{kEmptyBiddingSuitIndex});
  append_one_hot_array(
      values,
      bidding_history.target_point_cards,
      kBiddingTargetPointCardsClassCount,
      kMinBiddingTargetPointCards,
      std::array<int, 1>{0});

  if (static_cast<int>(values.size()) != kBiddingModelInputFeatureCount) {
    throw std::runtime_error("bidding model input feature count drift");
  }

  std::array<float, kBiddingModelInputFeatureCount> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

std::vector<float> encode_complete_info_model_input(
    const EncodedPlayingObservation& observation,
    const std::array<int, kCardCount>& card_owner_class_by_card,
    const std::array<int, kPlayerCount>& captured_point_card_count_by_player) {
  std::vector<float> values;
  values.reserve(kCompleteInfoPlayingModelInputFeatureCount);

  append_one_hot_array(
      values,
      card_owner_class_by_card,
      kCompleteInfoOwnerClassCount,
      0,
      std::array<int, 0>{});
  append_values(values, captured_point_card_count_by_player);
  append_values(values, observation.trump_suit_one_hot);
  values.push_back(static_cast<float>(observation.contract_target_point_cards));
  append_values(values, observation.napoleon_player_one_hot);
  append_values(values, observation.revealed_adjutant_player_one_hot);
  append_values(values, observation.self_role_one_hot);
  const auto called_it = std::find(
      observation.called_adjutant_card_mask.begin(),
      observation.called_adjutant_card_mask.end(),
      1);
  if (called_it == observation.called_adjutant_card_mask.end()) {
    throw std::runtime_error("complete-info observation has no called adjutant card");
  }
  values.push_back(static_cast<float>(
      std::distance(observation.called_adjutant_card_mask.begin(), called_it)));
  append_values(values, observation.special_card_indices);
  append_values(values, observation.current_trick_slot_mask);
  append_values(values, observation.current_trick_card_indices);
  append_one_hot_array(
      values,
      observation.current_trick_player_indices,
      kPlayerCount,
      0,
      std::array<int, 1>{kEmptyPlayerIndex});
  values.push_back(static_cast<float>(observation.trick_number));
  values.push_back(static_cast<float>(observation.completed_trick_count));

  if (static_cast<int>(values.size()) != kCompleteInfoPlayingModelInputFeatureCount) {
    throw std::runtime_error("complete-info playing model input feature count drift");
  }
  return values;
}

std::array<int, kCardCount> complete_info_card_owner_classes(
    const GameState& state,
    const EncodedPlayingObservation& observation) {
  std::array<int, kCardCount> owner_classes{};
  owner_classes.fill(kNotInHandOwnerClassIndex);

  for (int relative_index = 0; relative_index < kPlayerCount; ++relative_index) {
    const int absolute_index = observation.relative_player_indices[static_cast<std::size_t>(relative_index)];
    for (Card card : state.hands[static_cast<std::size_t>(absolute_index)]) {
      owner_classes[static_cast<std::size_t>(card_model_index(card))] = relative_index;
    }
  }
  return owner_classes;
}

std::array<int, kPlayerCount> awarded_point_card_counts_by_player(
    const GameState& state,
    const EncodedPlayingObservation& observation) {
  std::array<int, kPlayerCount> counts{};
  for (const AwardedPointCards& award : state.awarded_point_cards) {
    const int relative_index = relative_player_index(
        observation.relative_player_indices[0],
        award.player_index);
    counts[static_cast<std::size_t>(relative_index)] += static_cast<int>(award.cards.size());
  }
  return counts;
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

template <typename T, std::size_t Size>
void write_array(std::ostream& out, const std::array<T, Size>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << values[index];
  }
  out << ']';
}

template <std::size_t OuterSize, std::size_t InnerSize>
void write_matrix(
    std::ostream& out,
    const std::array<std::array<int, InnerSize>, OuterSize>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    write_array(out, values[index]);
  }
  out << ']';
}

void write_relative_player_ids(
    std::ostream& out,
    const std::array<int, kPlayerCount>& relative_player_indices) {
  out << '[';
  for (std::size_t index = 0; index < relative_player_indices.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    json_escape(out, player_id(relative_player_indices[index]));
  }
  out << ']';
}

void write_bidding_history(std::ostream& out, const EncodedBiddingHistory& history) {
  out << "{\"actionTypeIndices\":";
  write_array(out, history.action_type_indices);
  out << ",\"playerIndices\":";
  write_array(out, history.player_indices);
  out << ",\"suitIndices\":";
  write_array(out, history.suit_indices);
  out << ",\"targetPointCards\":";
  write_array(out, history.target_point_cards);
  out << ",\"actionMask\":";
  write_array(out, history.action_mask);
  out << '}';
}

void write_observation(std::ostream& out, const EncodedPlayingObservation& observation) {
  out << "{\"schemaVersion\":" << observation.schema_version;
  out << ",\"relativePlayerIds\":";
  write_relative_player_ids(out, observation.relative_player_indices);
  out << ",\"trickNumber\":" << observation.trick_number;
  out << ",\"completedTrickCount\":" << observation.completed_trick_count;
  out << ",\"contractTargetPointCards\":" << observation.contract_target_point_cards;
  out << ",\"trumpSuitOneHot\":";
  write_array(out, observation.trump_suit_one_hot);
  out << ",\"napoleonPlayerOneHot\":";
  write_array(out, observation.napoleon_player_one_hot);
  out << ",\"revealedAdjutantPlayerOneHot\":";
  write_array(out, observation.revealed_adjutant_player_one_hot);
  out << ",\"selfRoleOneHot\":";
  write_array(out, observation.self_role_one_hot);
  out << ",\"calledAdjutantCardMask\":";
  write_array(out, observation.called_adjutant_card_mask);
  out << ",\"selfHandMask\":";
  write_array(out, observation.self_hand_mask);
  out << ",\"legalPlayMask\":";
  write_array(out, observation.legal_play_mask);
  out << ",\"handCountByPlayer\":";
  write_array(out, observation.hand_count_by_player);
  out << ",\"capturedPointCardMaskByPlayer\":";
  write_matrix(out, observation.captured_point_card_mask_by_player);
  out << ",\"specialCardIndices\":{\"oruma\":" << observation.special_card_indices[0]
      << ",\"yoromeki\":" << observation.special_card_indices[1]
      << ",\"seiJack\":" << observation.special_card_indices[2]
      << ",\"uraJack\":" << observation.special_card_indices[3] << '}';
  out << ",\"currentTrickCardIndices\":";
  write_array(out, observation.current_trick_card_indices);
  out << ",\"currentTrickPlayerIndices\":";
  write_array(out, observation.current_trick_player_indices);
  out << ",\"currentTrickSlotMask\":";
  write_array(out, observation.current_trick_slot_mask);
  out << ",\"completedTrickCardIndices\":";
  write_array(out, observation.completed_trick_card_indices);
  out << ",\"completedTrickPlayerIndices\":";
  write_array(out, observation.completed_trick_player_indices);
  out << ",\"completedTrickSlotMask\":";
  write_array(out, observation.completed_trick_slot_mask);
  out << ",\"completedTrickWinnerIndices\":";
  write_array(out, observation.completed_trick_winner_indices);
  out << ",\"completedTrickMask\":";
  write_array(out, observation.completed_trick_mask);
  out << ",\"biddingHistory\":";
  write_bidding_history(out, observation.bidding_history);
  out << ",\"latestBuriedEventPointCardMask\":";
  write_array(out, observation.latest_buried_event_point_card_mask);
  out << ",\"latestBuriedEventHiddenNonPointCount\":"
      << observation.latest_buried_event_hidden_non_point_count;
  out << ",\"latestBuriedEventPresent\":" << observation.latest_buried_event_present << '}';
}

}  // namespace

PlayingModelInput create_playing_model_input(const GameState& state, int player_index) {
  if (state.phase != Phase::Playing || state.is_trick_complete) {
    throw std::runtime_error("playing model input requires an active playing decision state");
  }
  if (!state.trump_suit.has_value() || !state.contract.has_value() || !state.adjutant.has_value()) {
    throw std::runtime_error("playing model input requires trump, contract, and adjutant");
  }
  if (player_index < 0 || player_index >= kPlayerCount) {
    throw std::runtime_error("player index out of range");
  }

  PlayingModelInput result;
  result.player_index = player_index;
  EncodedPlayingObservation& encoded = result.observation;
  encoded.current_trick_card_indices.fill(kEmptyCardIndex);
  encoded.current_trick_player_indices.fill(kEmptyPlayerIndex);
  encoded.completed_trick_card_indices.fill(kEmptyCardIndex);
  encoded.completed_trick_player_indices.fill(kEmptyPlayerIndex);
  encoded.completed_trick_winner_indices.fill(kEmptyPlayerIndex);

  for (int relative_index = 0; relative_index < kPlayerCount; ++relative_index) {
    encoded.relative_player_indices[static_cast<std::size_t>(relative_index)] =
        absolute_player_index(player_index, relative_index);
  }

  encoded.trick_number = state.trick_number;
  encoded.completed_trick_count = static_cast<int>(state.completed_tricks.size());
  encoded.contract_target_point_cards = state.contract->target_point_cards;
  encoded.trump_suit_one_hot[static_cast<std::size_t>(*state.trump_suit)] = 1;
  encoded.napoleon_player_one_hot[static_cast<std::size_t>(relative_player_index(
      player_index, state.contract->napoleon_player_index))] = 1;

  if (state.adjutant->revealed && state.adjutant->player_index.has_value()) {
    encoded.revealed_adjutant_player_one_hot[static_cast<std::size_t>(relative_player_index(
        player_index, *state.adjutant->player_index))] = 1;
  } else {
    encoded.revealed_adjutant_player_one_hot[static_cast<std::size_t>(kPlayerCount)] = 1;
  }

  int role_index = 2;
  if (player_index == state.contract->napoleon_player_index) {
    role_index = state.adjutant->player_index.has_value() ? 0 : 3;
  } else if (state.adjutant->player_index.has_value() &&
             player_index == *state.adjutant->player_index) {
    role_index = 1;
  }
  encoded.self_role_one_hot[static_cast<std::size_t>(role_index)] = 1;

  set_card_mask(encoded.called_adjutant_card_mask, state.adjutant->called_card);
  for (Card card : state.hands[static_cast<std::size_t>(player_index)]) {
    set_card_mask(encoded.self_hand_mask, card);
  }
  for (const Action& action : get_legal_actions(state, player_index)) {
    if (action.type == Action::Type::PlayCard) {
      set_card_mask(encoded.legal_play_mask, action.card);
    }
  }
  result.legal_play_mask = encoded.legal_play_mask;

  for (int relative_index = 0; relative_index < kPlayerCount; ++relative_index) {
    const int absolute_index = encoded.relative_player_indices[static_cast<std::size_t>(relative_index)];
    encoded.hand_count_by_player[static_cast<std::size_t>(relative_index)] =
        static_cast<int>(state.hands[static_cast<std::size_t>(absolute_index)].size());
  }

  for (const CompletedTrick& trick : state.completed_tricks) {
    const int winner_relative = relative_player_index(player_index, trick.winner_index);
    auto& mask = encoded.captured_point_card_mask_by_player[static_cast<std::size_t>(winner_relative)];
    for (const PlayedCard& played : trick.cards) {
      if (is_point_card(played.card)) {
        set_card_mask(mask, played.card);
      }
    }
  }
  for (const AwardedPointCards& award : state.awarded_point_cards) {
    const int award_relative = relative_player_index(player_index, award.player_index);
    auto& mask = encoded.captured_point_card_mask_by_player[static_cast<std::size_t>(award_relative)];
    for (Card card : award.cards) {
      set_card_mask(mask, card);
    }
  }

  encoded.special_card_indices = {
      card_model_index(parse_card_id("spades-A")),
      card_model_index(parse_card_id("hearts-Q")),
      card_model_index(sei_jack_card(*state.trump_suit)),
      card_model_index(ura_jack_card(*state.trump_suit))};

  for (std::size_t index = 0; index < state.current_trick.size(); ++index) {
    const PlayedCard& played = state.current_trick[index];
    encoded.current_trick_card_indices[index] = card_model_index(played.card);
    encoded.current_trick_player_indices[index] =
        relative_player_index(player_index, played.player_index);
    encoded.current_trick_slot_mask[index] = 1;
  }

  for (std::size_t trick_index = 0; trick_index < state.completed_tricks.size(); ++trick_index) {
    const CompletedTrick& trick = state.completed_tricks[trick_index];
    encoded.completed_trick_winner_indices[trick_index] =
        relative_player_index(player_index, trick.winner_index);
    encoded.completed_trick_mask[trick_index] = 1;
    for (std::size_t card_offset = 0; card_offset < trick.cards.size(); ++card_offset) {
      const std::size_t slot_index = trick_index * kCardsPerTrick + card_offset;
      encoded.completed_trick_card_indices[slot_index] = card_model_index(trick.cards[card_offset].card);
      encoded.completed_trick_player_indices[slot_index] =
          relative_player_index(player_index, trick.cards[card_offset].player_index);
      encoded.completed_trick_slot_mask[slot_index] = 1;
    }
  }

  encoded.bidding_history = encode_bidding_history(state, encoded.relative_player_indices);

  if (state.latest_event.has_value()) {
    for (Card card : state.latest_event->awarded_point_cards) {
      set_card_mask(encoded.latest_buried_event_point_card_mask, card);
    }
    encoded.latest_buried_event_hidden_non_point_count =
        state.latest_event->hidden_non_point_card_count;
    encoded.latest_buried_event_present = 1;
  }

  result.model_input = encode_model_input(encoded);
  return result;
}

VariantPlayingModelInput create_playing_model_input(
    const GameState& state,
    int player_index,
    PlayingObservationVariant variant) {
  const PlayingModelInput public_input = create_playing_model_input(state, player_index);

  VariantPlayingModelInput result;
  result.variant = variant;
  result.encoder_schema_version = playing_encoder_schema_version(variant);
  result.model_input_schema_version = playing_model_input_schema_version(variant);
  result.model_input_feature_count = playing_model_input_feature_count(variant);
  result.player_index = player_index;
  result.legal_play_mask = public_input.legal_play_mask;

  if (variant == PlayingObservationVariant::Public) {
    result.model_input.assign(public_input.model_input.begin(), public_input.model_input.end());
    return result;
  }

  const std::array<int, kCardCount> owner_classes =
      complete_info_card_owner_classes(state, public_input.observation);
  const std::array<int, kPlayerCount> captured_counts =
      awarded_point_card_counts_by_player(state, public_input.observation);
  result.model_input =
      encode_complete_info_model_input(public_input.observation, owner_classes, captured_counts);
  return result;
}

BiddingModelInput create_bidding_model_input(const GameState& state, int player_index) {
  if (state.phase != Phase::Bidding || !state.bidding.has_value()) {
    throw std::runtime_error("bidding model input requires an active bidding state");
  }
  if (player_index < 0 || player_index >= kPlayerCount) {
    throw std::runtime_error("player index out of range");
  }

  BiddingModelInput result;
  result.player_index = player_index;
  for (int relative_index = 0; relative_index < kPlayerCount; ++relative_index) {
    result.relative_player_indices[static_cast<std::size_t>(relative_index)] =
        absolute_player_index(player_index, relative_index);
  }
  for (const Action& action : get_legal_actions(state, player_index)) {
    if (action.type == Action::Type::Bid || action.type == Action::Type::Pass) {
      result.legal_bid_mask[static_cast<std::size_t>(bidding_action_index(action))] = 1;
    }
  }
  const EncodedBiddingHistory bidding_history =
      encode_bidding_history(state, result.relative_player_indices);
  result.model_input = encode_bidding_model_input(result, bidding_history, state);
  return result;
}

int playing_card_model_index(Card card) {
  return card_model_index(card);
}

Card card_from_playing_model_index(int index) {
  return card_from_model_index(index);
}

int bidding_action_index(const Action& action) {
  if (action.type == Action::Type::Pass) {
    return 0;
  }
  if (action.type != Action::Type::Bid || !action.suit.has_value()) {
    throw std::runtime_error("bidding action index requires pass or bid");
  }
  if (action.target_point_cards < kMinBiddingTargetPointCards ||
      action.target_point_cards > kMinBiddingTargetPointCards + kBiddingTargetPointCardsClassCount - 1) {
    throw std::runtime_error("bidding target out of range");
  }
  return 1 + (action.target_point_cards - kMinBiddingTargetPointCards) * 4 +
         static_cast<int>(*action.suit);
}

Action bidding_action_from_index(int action_index, int player_index) {
  if (action_index < 0 || action_index >= kBiddingActionCount) {
    throw std::runtime_error("bidding action index out of range");
  }
  Action action;
  action.player_index = player_index;
  if (action_index == 0) {
    action.type = Action::Type::Pass;
    return action;
  }
  const int offset = action_index - 1;
  action.type = Action::Type::Bid;
  action.target_point_cards = kMinBiddingTargetPointCards + offset / 4;
  action.suit = static_cast<Suit>(offset % 4);
  return action;
}

PlayingObservationVariant parse_playing_observation_variant(const std::string& value) {
  if (value == "public") {
    return PlayingObservationVariant::Public;
  }
  if (value == "complete-info-compact") {
    return PlayingObservationVariant::CompleteInfoCompact;
  }
  throw std::runtime_error("playing observation variant must be public or complete-info-compact");
}

std::string playing_observation_variant_id(PlayingObservationVariant variant) {
  switch (variant) {
    case PlayingObservationVariant::Public:
      return "public";
    case PlayingObservationVariant::CompleteInfoCompact:
      return "complete-info-compact";
  }
  throw std::runtime_error("invalid playing observation variant");
}

int playing_encoder_schema_version(PlayingObservationVariant variant) {
  switch (variant) {
    case PlayingObservationVariant::Public:
      return kPlayingEncoderSchemaVersion;
    case PlayingObservationVariant::CompleteInfoCompact:
      return kCompleteInfoPlayingEncoderSchemaVersion;
  }
  throw std::runtime_error("invalid playing observation variant");
}

int playing_model_input_schema_version(PlayingObservationVariant variant) {
  switch (variant) {
    case PlayingObservationVariant::Public:
      return kPlayingModelInputSchemaVersion;
    case PlayingObservationVariant::CompleteInfoCompact:
      return kCompleteInfoPlayingModelInputSchemaVersion;
  }
  throw std::runtime_error("invalid playing observation variant");
}

int playing_model_input_feature_count(PlayingObservationVariant variant) {
  switch (variant) {
    case PlayingObservationVariant::Public:
      return kPlayingModelInputFeatureCount;
    case PlayingObservationVariant::CompleteInfoCompact:
      return kCompleteInfoPlayingModelInputFeatureCount;
  }
  throw std::runtime_error("invalid playing observation variant");
}

std::optional<PlayingModelInput> create_current_player_playing_model_input(const GameState& state) {
  if (state.phase != Phase::Playing || state.is_trick_complete) {
    return std::nullopt;
  }
  return create_playing_model_input(state, state.current_player_index);
}

std::string current_player_playing_model_input_json(const GameState& state) {
  const std::optional<PlayingModelInput> model_input =
      create_current_player_playing_model_input(state);
  if (!model_input.has_value()) {
    return "null";
  }

  std::ostringstream out;
  out << "{\"modelInputSchemaVersion\":" << model_input->model_input_schema_version;
  out << ",\"modelInputFeatureCount\":" << model_input->model_input_feature_count;
  out << ",\"playerId\":";
  json_escape(out, player_id(model_input->player_index));
  out << ",\"observation\":";
  write_observation(out, model_input->observation);
  out << ",\"legalPlayMask\":";
  write_array(out, model_input->legal_play_mask);
  out << ",\"modelInput\":";
  write_array(out, model_input->model_input);
  out << '}';
  return out.str();
}

std::string current_player_bidding_model_input_json(const GameState& state) {
  if (state.phase != Phase::Bidding) {
    return "null";
  }

  const BiddingModelInput model_input =
      create_bidding_model_input(state, state.current_player_index);
  std::ostringstream out;
  out << "{\"encoderSchemaVersion\":" << model_input.encoder_schema_version;
  out << ",\"modelInputSchemaVersion\":" << model_input.model_input_schema_version;
  out << ",\"modelInputFeatureCount\":" << model_input.model_input_feature_count;
  out << ",\"playerId\":";
  json_escape(out, player_id(model_input.player_index));
  out << ",\"relativePlayerIds\":[";
  for (std::size_t index = 0; index < model_input.relative_player_indices.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    json_escape(out, player_id(model_input.relative_player_indices[index]));
  }
  out << "],\"legalBidMask\":";
  write_array(out, model_input.legal_bid_mask);
  out << ",\"modelInput\":";
  write_array(out, model_input.model_input);
  out << '}';
  return out.str();
}

std::string canonical_snapshot_with_current_player_playing_model_input_json(const GameState& state) {
  std::string snapshot = canonical_snapshot_json(state);
  if (snapshot.empty() || snapshot.back() != '}') {
    throw std::runtime_error("canonical snapshot must be a JSON object");
  }

  snapshot.pop_back();
  snapshot += ",\"playingModelInput\":";
  snapshot += current_player_playing_model_input_json(state);
  snapshot += ",\"biddingModelInput\":";
  snapshot += current_player_bidding_model_input_json(state);
  snapshot += '}';
  return snapshot;
}

}  // namespace napoleon::observation
