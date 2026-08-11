#pragma once

#include "napoleon_core.hpp"

#include <array>
#include <optional>
#include <string>

namespace napoleon::observation {

constexpr int kPlayingEncoderSchemaVersion = 2;
constexpr int kPlayingModelInputSchemaVersion = 2;
constexpr int kCardCount = 53;
constexpr int kTrickCount = 10;
constexpr int kCardsPerTrick = 5;
constexpr int kMaxBiddingActionCount = 117;
constexpr int kPlayingModelInputFeatureCount = 6246;
constexpr int kFlatObservationFeatureCount = 684;

struct EncodedBiddingHistory {
  std::array<int, kMaxBiddingActionCount> action_type_indices{};
  std::array<int, kMaxBiddingActionCount> player_indices{};
  std::array<int, kMaxBiddingActionCount> suit_indices{};
  std::array<int, kMaxBiddingActionCount> target_point_cards{};
  std::array<int, kMaxBiddingActionCount> action_mask{};
};

struct EncodedPlayingObservation {
  int schema_version = kPlayingEncoderSchemaVersion;
  std::array<int, kPlayerCount> relative_player_indices{};
  int trick_number = 1;
  int completed_trick_count = 0;
  int contract_target_point_cards = 12;
  std::array<int, 4> trump_suit_one_hot{};
  std::array<int, kPlayerCount> napoleon_player_one_hot{};
  std::array<int, kPlayerCount + 1> revealed_adjutant_player_one_hot{};
  std::array<int, 4> self_role_one_hot{};
  std::array<int, kCardCount> called_adjutant_card_mask{};
  std::array<int, kCardCount> self_hand_mask{};
  std::array<int, kCardCount> legal_play_mask{};
  std::array<int, kPlayerCount> hand_count_by_player{};
  std::array<std::array<int, kCardCount>, kPlayerCount> captured_point_card_mask_by_player{};
  std::array<int, 4> special_card_indices{};
  std::array<int, kCardsPerTrick> current_trick_card_indices{};
  std::array<int, kCardsPerTrick> current_trick_player_indices{};
  std::array<int, kCardsPerTrick> current_trick_slot_mask{};
  std::array<int, kTrickCount * kCardsPerTrick> completed_trick_card_indices{};
  std::array<int, kTrickCount * kCardsPerTrick> completed_trick_player_indices{};
  std::array<int, kTrickCount * kCardsPerTrick> completed_trick_slot_mask{};
  std::array<int, kTrickCount> completed_trick_winner_indices{};
  std::array<int, kTrickCount> completed_trick_mask{};
  EncodedBiddingHistory bidding_history;
  std::array<int, kCardCount> latest_buried_event_point_card_mask{};
  int latest_buried_event_hidden_non_point_count = 0;
  int latest_buried_event_present = 0;
};

struct PlayingModelInput {
  int model_input_schema_version = kPlayingModelInputSchemaVersion;
  int model_input_feature_count = kPlayingModelInputFeatureCount;
  int player_index = 0;
  EncodedPlayingObservation observation;
  std::array<float, kPlayingModelInputFeatureCount> model_input{};
  std::array<int, kCardCount> legal_play_mask{};
};

std::optional<PlayingModelInput> create_current_player_playing_model_input(const GameState& state);
PlayingModelInput create_playing_model_input(const GameState& state, int player_index);
std::string current_player_playing_model_input_json(const GameState& state);
std::string canonical_snapshot_with_current_player_playing_model_input_json(const GameState& state);

}  // namespace napoleon::observation
