#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace napoleon {

constexpr int kPlayerCount = 5;
constexpr int kCardsPerPlayer = 10;

enum class Suit : std::uint8_t {
  Spades = 0,
  Hearts = 1,
  Diamonds = 2,
  Clubs = 3
};

enum class Rank : std::uint8_t {
  Ace = 0,
  Two = 1,
  Three = 2,
  Four = 3,
  Five = 4,
  Six = 5,
  Seven = 6,
  Eight = 7,
  Nine = 8,
  Ten = 9,
  Jack = 10,
  Queen = 11,
  King = 12
};

enum class Phase : std::uint8_t {
  Bidding,
  Exchanging,
  ChoosingAdjutant,
  Playing,
  Finished
};

struct Card {
  std::uint8_t id = 0;
};

struct PlayedCard {
  int player_index = 0;
  Card card;
};

struct CompletedTrick {
  int trick_number = 1;
  int winner_index = 0;
  std::vector<PlayedCard> cards;
};

struct Bid {
  int player_index = 0;
  Suit suit = Suit::Spades;
  int target_point_cards = 13;
};

struct Contract {
  int napoleon_player_index = 0;
  Suit trump_suit = Suit::Spades;
  int target_point_cards = 13;
};

struct AdjutantState {
  Card called_card;
  std::optional<int> player_index;
  bool revealed = false;
};

struct BiddingHistoryEntry {
  bool is_bid = false;
  int player_index = 0;
  std::optional<Suit> suit;
  std::optional<int> target_point_cards;
};

struct BiddingState {
  int starter_player_index = 0;
  std::optional<Bid> highest_bid;
  int consecutive_pass_count = 0;
  std::vector<BiddingHistoryEntry> history;
};

struct AwardedPointCards {
  int player_index = 0;
  std::vector<Card> cards;
};

struct BuriedCardsResolvedEvent {
  int napoleon_player_index = 0;
  std::vector<Card> awarded_point_cards;
  int hidden_non_point_card_count = 0;
};

struct GameResult {
  std::string winner;
  int napoleon_team_point_cards = 0;
  int alliance_point_cards = 0;
  int target_point_cards = 0;
  int napoleon_player_index = 0;
  std::optional<int> adjutant_player_index;
};

struct GameState {
  std::array<std::vector<Card>, kPlayerCount> hands;
  Phase phase = Phase::Bidding;
  int current_player_index = 0;
  std::vector<PlayedCard> current_trick;
  std::vector<CompletedTrick> completed_tricks;
  std::optional<Suit> trump_suit;
  std::optional<Contract> contract;
  std::optional<AdjutantState> adjutant;
  std::optional<BiddingState> bidding;
  std::vector<AwardedPointCards> awarded_point_cards;
  std::vector<Card> excluded_cards;
  std::optional<BuriedCardsResolvedEvent> latest_event;
  std::optional<GameResult> result;
  int trick_number = 1;
  bool is_trick_complete = false;
  bool is_game_over = false;
  std::vector<Card> unused_cards;
};

struct Action {
  enum class Type {
    Bid,
    Pass,
    ChooseAdjutant,
    DiscardCards,
    PlayCard,
    AdvanceToNextTrick
  };

  Type type = Type::Pass;
  int player_index = 0;
  std::optional<Suit> suit;
  int target_point_cards = 0;
  Card card;
  std::vector<Card> cards;
};

class SeededRandom {
 public:
  explicit SeededRandom(std::uint32_t seed);
  double next();

 private:
  std::uint32_t state_;
};

std::vector<Card> create_deck();
GameState create_initial_game(std::uint32_t seed);
std::vector<Action> get_legal_actions(const GameState& state, int player_index);
void apply_action(GameState& state, const Action& action);

std::string canonical_snapshot_json(const GameState& state);
std::string action_json(const Action& action);
Action parse_action_line(const std::string& line);

std::string card_id(Card card);
Card parse_card_id(const std::string& value);
std::string suit_id(Suit suit);
Suit parse_suit(const std::string& value);
std::string phase_id(Phase phase);

}  // namespace napoleon
