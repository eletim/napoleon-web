#include "napoleon_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace napoleon {
namespace {

constexpr std::array<const char*, 4> kSuitIds = {"spades", "hearts", "diamonds", "clubs"};
constexpr std::array<const char*, 13> kRankIds = {
    "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

std::string player_id(int player_index) {
  return "player-" + std::to_string(player_index);
}

int parse_player_id(const std::string& value) {
  constexpr const char* prefix = "player-";
  if (value.rfind(prefix, 0) != 0) {
    throw std::runtime_error("invalid player id: " + value);
  }

  const int player_index = std::stoi(value.substr(7));
  if (player_index < 0 || player_index >= kPlayerCount) {
    throw std::runtime_error("player id out of range: " + value);
  }

  return player_index;
}

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

bool is_point_card(Card card) {
  if (is_joker(card)) {
    return false;
  }

  const Rank rank = card_rank(card);
  return rank == Rank::Ten || rank == Rank::Jack || rank == Rank::Queen ||
         rank == Rank::King || rank == Rank::Ace;
}

bool is_oruma(Card card) {
  return card_id(card) == "spades-A";
}

bool is_yoromeki(Card card) {
  return card_id(card) == "hearts-Q";
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

int next_player_index(int player_index) {
  return (player_index + 1) % kPlayerCount;
}

int bidding_suit_priority(Suit suit) {
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

  throw std::runtime_error("invalid suit");
}

int compare_bids(const Bid& left, const Bid& right) {
  if (left.target_point_cards != right.target_point_cards) {
    return left.target_point_cards - right.target_point_cards;
  }

  return bidding_suit_priority(left.suit) - bidding_suit_priority(right.suit);
}

std::optional<Suit> lead_suit(const std::vector<PlayedCard>& trick, std::optional<Suit> trump_suit) {
  if (trick.empty()) {
    return std::nullopt;
  }

  const Card lead_card = trick.front().card;
  if (!is_joker(lead_card)) {
    return card_suit(lead_card);
  }

  if (!trump_suit.has_value()) {
    throw std::runtime_error("trump suit required when joker leads");
  }

  return trump_suit;
}

bool can_follow(Card card, Suit suit) {
  return !is_joker(card) && card_suit(card) == suit;
}

std::vector<Card> playable_cards(
    const std::vector<Card>& hand,
    const std::vector<PlayedCard>& trick,
    std::optional<Suit> trump_suit) {
  const std::optional<Suit> lead = lead_suit(trick, trump_suit);
  if (!lead.has_value()) {
    return hand;
  }

  const bool has_follow = std::any_of(hand.begin(), hand.end(), [&](Card card) {
    return can_follow(card, *lead);
  });
  if (!has_follow) {
    return hand;
  }

  std::vector<Card> result;
  for (Card card : hand) {
    if (is_joker(card) || can_follow(card, *lead)) {
      result.push_back(card);
    }
  }
  return result;
}

std::optional<int> find_same_two_winner(
    const std::vector<PlayedCard>& trick,
    std::optional<Suit> trump_suit,
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
    if (card_suit(played.card) != same_two_suit || is_oruma(played.card)) {
      return std::nullopt;
    }

    if (!trump_suit.has_value() && card_rank(played.card) == Rank::Jack) {
      return std::nullopt;
    }

    if (trump_suit.has_value() &&
        (played.card.id == sei_jack(*trump_suit).id || played.card.id == ura_jack(*trump_suit).id)) {
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
    std::optional<Suit> trump_suit) {
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
  if (!trump_suit.has_value()) {
    throw std::runtime_error("trump suit required to evaluate jacks");
  }

  const Card sei = sei_jack(*trump_suit);
  auto sei_play = std::find_if(trick.begin(), trick.end(), [&](const PlayedCard& played) {
    return played.card.id == sei.id;
  });
  if (sei_play != trick.end()) {
    return sei_play->player_index;
  }

  const Card ura = ura_jack(*trump_suit);
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

Strength card_strength(Card card, Suit lead, std::optional<Suit> trump_suit, bool is_lead_card) {
  int category_priority = 0;
  if (is_joker(card)) {
    category_priority = is_lead_card ? 2 : 1;
  } else if (trump_suit.has_value() && card_suit(card) == *trump_suit) {
    category_priority = 2;
  } else if (card_suit(card) == lead) {
    category_priority = 1;
  }

  return Strength{category_priority, is_joker(card) ? 1 : rank_value(card_rank(card))};
}

int determine_trick_winner(
    const std::vector<PlayedCard>& trick,
    std::optional<Suit> trump_suit,
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

void ensure_turn(const GameState& state, int player_index) {
  if (state.current_player_index != player_index) {
    throw std::runtime_error("not player's turn");
  }
}

void ensure_bidding_allowed(const GameState& state, int player_index) {
  if (state.phase == Phase::Finished || state.is_game_over) {
    throw std::runtime_error("game over");
  }
  if (state.phase != Phase::Bidding) {
    throw std::runtime_error("action is only valid during bidding");
  }
  ensure_turn(state, player_index);
}

void complete_bidding(GameState& state, const Contract& contract) {
  state.phase = Phase::ChoosingAdjutant;
  state.current_player_index = contract.napoleon_player_index;
  state.trump_suit = contract.trump_suit;
  state.contract = contract;
  state.bidding = std::nullopt;
  state.awarded_point_cards.clear();
  state.excluded_cards.clear();
  state.latest_event = std::nullopt;
  state.adjutant = std::nullopt;
  state.result = std::nullopt;
}

void append_awarded_point_cards(GameState& state, int player_index, const std::vector<Card>& cards) {
  if (cards.empty()) {
    return;
  }

  for (AwardedPointCards& award : state.awarded_point_cards) {
    if (award.player_index == player_index) {
      award.cards.insert(award.cards.end(), cards.begin(), cards.end());
      return;
    }
  }

  state.awarded_point_cards.push_back(AwardedPointCards{player_index, cards});
}

int resolved_buried_card_count(const GameState& state) {
  int count = static_cast<int>(state.excluded_cards.size());
  for (const AwardedPointCards& award : state.awarded_point_cards) {
    count += static_cast<int>(award.cards.size());
  }
  return count;
}

std::optional<int> resolve_adjutant_player_index(const GameState& state, Card card) {
  if (!state.contract.has_value()) {
    throw std::runtime_error("contract required to resolve adjutant");
  }

  for (int player_index = 0; player_index < kPlayerCount; ++player_index) {
    const auto& hand = state.hands[static_cast<std::size_t>(player_index)];
    if (std::any_of(hand.begin(), hand.end(), [&](Card candidate) {
          return candidate.id == card.id;
        })) {
      return player_index == state.contract->napoleon_player_index
                 ? std::optional<int>{}
                 : std::optional<int>{player_index};
    }
  }

  auto contains = [&](const std::vector<Card>& cards) {
    return std::any_of(cards.begin(), cards.end(), [&](Card candidate) {
      return candidate.id == card.id;
    });
  };
  if (contains(state.unused_cards) || contains(state.excluded_cards)) {
    return std::nullopt;
  }
  for (const AwardedPointCards& award : state.awarded_point_cards) {
    if (contains(award.cards)) {
      return std::nullopt;
    }
  }

  throw std::runtime_error("adjutant card is not in the game state");
}

void reveal_adjutant_if_needed(GameState& state, int player_index, Card card) {
  if (!state.adjutant.has_value() || state.adjutant->revealed ||
      state.adjutant->called_card.id != card.id) {
    return;
  }

  if (!state.adjutant->player_index.has_value()) {
    return;
  }

  if (*state.adjutant->player_index != player_index) {
    throw std::runtime_error("called adjutant owner is inconsistent");
  }

  state.adjutant->revealed = true;
}

GameResult calculate_result(const GameState& state) {
  if (!state.contract.has_value() || !state.adjutant.has_value()) {
    throw std::runtime_error("contract and adjutant required for result");
  }

  std::unordered_set<int> napoleon_team{state.contract->napoleon_player_index};
  if (state.adjutant->player_index.has_value()) {
    napoleon_team.insert(*state.adjutant->player_index);
  }

  int napoleon_points = 0;
  int alliance_points = 0;
  for (const CompletedTrick& trick : state.completed_tricks) {
    int trick_points = 0;
    for (const PlayedCard& played : trick.cards) {
      if (is_point_card(played.card)) {
        ++trick_points;
      }
    }

    if (napoleon_team.count(trick.winner_index) != 0) {
      napoleon_points += trick_points;
    } else {
      alliance_points += trick_points;
    }
  }

  for (const AwardedPointCards& award : state.awarded_point_cards) {
    if (napoleon_team.count(award.player_index) != 0) {
      napoleon_points += static_cast<int>(award.cards.size());
    } else {
      alliance_points += static_cast<int>(award.cards.size());
    }
  }

  return GameResult{
      napoleon_points >= state.contract->target_point_cards ? "napoleon-team" : "alliance",
      napoleon_points,
      alliance_points,
      state.contract->target_point_cards,
      state.contract->napoleon_player_index,
      state.adjutant->player_index};
}

void erase_card(std::vector<Card>& hand, Card card) {
  const auto position = std::find_if(hand.begin(), hand.end(), [&](Card candidate) {
    return candidate.id == card.id;
  });
  if (position == hand.end()) {
    throw std::runtime_error("card not in hand");
  }
  hand.erase(position);
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

void write_card(std::ostream& out, Card card) {
  json_escape(out, card_id(card));
}

void write_played_card(std::ostream& out, const PlayedCard& played) {
  out << "{\"playerId\":";
  json_escape(out, player_id(played.player_index));
  out << ",\"cardId\":";
  write_card(out, played.card);
  out << '}';
}

void write_action(std::ostream& out, const Action& action) {
  out << "{\"type\":";
  switch (action.type) {
    case Action::Type::Bid:
      json_escape(out, "bid");
      out << ",\"playerId\":";
      json_escape(out, player_id(action.player_index));
      out << ",\"suit\":";
      json_escape(out, suit_id(*action.suit));
      out << ",\"targetPointCards\":" << action.target_point_cards;
      break;
    case Action::Type::Pass:
      json_escape(out, "pass");
      out << ",\"playerId\":";
      json_escape(out, player_id(action.player_index));
      break;
    case Action::Type::ChooseAdjutant:
      json_escape(out, "choose-adjutant");
      out << ",\"playerId\":";
      json_escape(out, player_id(action.player_index));
      out << ",\"cardId\":";
      write_card(out, action.card);
      break;
    case Action::Type::DiscardCards:
      json_escape(out, "discard-cards");
      out << ",\"playerId\":";
      json_escape(out, player_id(action.player_index));
      out << ",\"cardIds\":";
      write_array(out, action.cards, write_card);
      break;
    case Action::Type::PlayCard:
      json_escape(out, "play-card");
      out << ",\"playerId\":";
      json_escape(out, player_id(action.player_index));
      out << ",\"cardId\":";
      write_card(out, action.card);
      break;
    case Action::Type::AdvanceToNextTrick:
      json_escape(out, "next-trick");
      break;
  }
  out << '}';
}

void write_completed_trick(std::ostream& out, const CompletedTrick& trick) {
  out << "{\"trickNumber\":" << trick.trick_number << ",\"winnerId\":";
  json_escape(out, player_id(trick.winner_index));
  out << ",\"cards\":";
  write_array(out, trick.cards, write_played_card);
  out << '}';
}

void write_contract(std::ostream& out, const std::optional<Contract>& contract) {
  if (!contract.has_value()) {
    out << "null";
    return;
  }

  out << "{\"napoleonPlayerId\":";
  json_escape(out, player_id(contract->napoleon_player_index));
  out << ",\"trumpSuit\":";
  json_escape(out, suit_id(contract->trump_suit));
  out << ",\"targetPointCards\":" << contract->target_point_cards << '}';
}

void write_adjutant(std::ostream& out, const std::optional<AdjutantState>& adjutant) {
  if (!adjutant.has_value()) {
    out << "null";
    return;
  }

  out << "{\"calledCardId\":";
  write_card(out, adjutant->called_card);
  out << ",\"playerId\":";
  if (adjutant->player_index.has_value()) {
    json_escape(out, player_id(*adjutant->player_index));
  } else {
    out << "null";
  }
  out << ",\"revealed\":" << (adjutant->revealed ? "true" : "false") << '}';
}

void write_bidding(std::ostream& out, const std::optional<BiddingState>& bidding) {
  if (!bidding.has_value()) {
    out << "null";
    return;
  }

  out << "{\"starterPlayerId\":";
  json_escape(out, player_id(bidding->starter_player_index));
  out << ",\"highestBid\":";
  if (bidding->highest_bid.has_value()) {
    out << "{\"playerId\":";
    json_escape(out, player_id(bidding->highest_bid->player_index));
    out << ",\"suit\":";
    json_escape(out, suit_id(bidding->highest_bid->suit));
    out << ",\"targetPointCards\":" << bidding->highest_bid->target_point_cards << '}';
  } else {
    out << "null";
  }
  out << ",\"consecutivePassCount\":" << bidding->consecutive_pass_count;
  out << ",\"history\":";
  write_array(out, bidding->history, [](std::ostream& history_out, const BiddingHistoryEntry& entry) {
    history_out << "{\"type\":";
    json_escape(history_out, entry.is_bid ? "bid" : "pass");
    history_out << ",\"playerId\":";
    json_escape(history_out, player_id(entry.player_index));
    if (entry.is_bid) {
      history_out << ",\"suit\":";
      json_escape(history_out, suit_id(*entry.suit));
      history_out << ",\"targetPointCards\":" << *entry.target_point_cards;
    }
    history_out << '}';
  });
  out << '}';
}

void write_awards(std::ostream& out, const std::vector<AwardedPointCards>& awards) {
  write_array(out, awards, [](std::ostream& award_out, const AwardedPointCards& award) {
    award_out << "{\"playerId\":";
    json_escape(award_out, player_id(award.player_index));
    award_out << ",\"cardIds\":";
    write_array(award_out, award.cards, write_card);
    award_out << '}';
  });
}

void write_result(std::ostream& out, const std::optional<GameResult>& result) {
  if (!result.has_value()) {
    out << "null";
    return;
  }

  out << "{\"winner\":";
  json_escape(out, result->winner);
  out << ",\"napoleonTeamPointCards\":" << result->napoleon_team_point_cards;
  out << ",\"alliancePointCards\":" << result->alliance_point_cards;
  out << ",\"targetPointCards\":" << result->target_point_cards;
  out << ",\"napoleonPlayerId\":";
  json_escape(out, player_id(result->napoleon_player_index));
  out << ",\"adjutantPlayerId\":";
  if (result->adjutant_player_index.has_value()) {
    json_escape(out, player_id(*result->adjutant_player_index));
  } else {
    out << "null";
  }
  out << '}';
}

void write_latest_event(std::ostream& out, const std::optional<BuriedCardsResolvedEvent>& event) {
  if (!event.has_value()) {
    out << "null";
    return;
  }

  out << "{\"type\":\"buried-cards-resolved\",\"napoleonPlayerId\":";
  json_escape(out, player_id(event->napoleon_player_index));
  out << ",\"awardedPointCardIds\":";
  write_array(out, event->awarded_point_cards, write_card);
  out << ",\"hiddenNonPointCardCount\":" << event->hidden_non_point_card_count << '}';
}

}  // namespace

SeededRandom::SeededRandom(std::uint32_t seed) : state_(seed) {}

double SeededRandom::next() {
  state_ = state_ + 0x6d2b79f5u;
  std::uint32_t value = state_;
  value = static_cast<std::uint32_t>((value ^ (value >> 15)) * (value | 1u));
  value ^= value + static_cast<std::uint32_t>((value ^ (value >> 7)) * (value | 61u));
  return static_cast<double>((value ^ (value >> 14)) & 0xffffffffu) / 4294967296.0;
}

std::vector<Card> create_deck() {
  std::vector<Card> deck;
  deck.reserve(53);
  for (std::uint8_t id = 0; id < 53; ++id) {
    deck.push_back(Card{id});
  }
  return deck;
}

GameState create_initial_game(std::uint32_t seed) {
  std::vector<Card> deck = create_deck();
  SeededRandom rng(seed);
  for (int index = static_cast<int>(deck.size()) - 1; index > 0; --index) {
    const int swap_index = static_cast<int>(std::floor(rng.next() * (index + 1)));
    std::swap(deck[static_cast<std::size_t>(index)], deck[static_cast<std::size_t>(swap_index)]);
  }

  GameState state;
  for (int player_index = 0; player_index < kPlayerCount; ++player_index) {
    const int start = player_index * kCardsPerPlayer;
    state.hands[static_cast<std::size_t>(player_index)] = std::vector<Card>(
        deck.begin() + start, deck.begin() + start + kCardsPerPlayer);
  }
  state.unused_cards = std::vector<Card>(deck.begin() + kPlayerCount * kCardsPerPlayer, deck.end());
  state.bidding = BiddingState{};
  return state;
}

std::vector<Action> get_legal_actions(const GameState& state, int player_index) {
  if (state.phase == Phase::Finished || state.is_game_over || state.current_player_index != player_index) {
    return {};
  }

  std::vector<Action> actions;
  if (state.phase == Phase::Bidding) {
    Action pass_action;
    pass_action.type = Action::Type::Pass;
    pass_action.player_index = player_index;
    actions.push_back(pass_action);
    for (int target = 13; target <= 19; ++target) {
      for (Suit suit : {Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades}) {
        Bid candidate{player_index, suit, target};
        if (!state.bidding->highest_bid.has_value() ||
            compare_bids(candidate, *state.bidding->highest_bid) > 0) {
          Action action;
          action.type = Action::Type::Bid;
          action.player_index = player_index;
          action.suit = suit;
          action.target_point_cards = target;
          actions.push_back(action);
        }
      }
    }
    return actions;
  }

  if (state.phase == Phase::ChoosingAdjutant) {
    for (Card card : create_deck()) {
      Action action;
      action.type = Action::Type::ChooseAdjutant;
      action.player_index = player_index;
      action.card = card;
      actions.push_back(action);
    }
    return actions;
  }

  if (state.phase == Phase::Playing && !state.is_trick_complete) {
    for (Card card : playable_cards(
             state.hands[static_cast<std::size_t>(player_index)],
             state.current_trick,
             state.trump_suit)) {
      Action action;
      action.type = Action::Type::PlayCard;
      action.player_index = player_index;
      action.card = card;
      actions.push_back(action);
    }
  }

  return actions;
}

void apply_action(GameState& state, const Action& action) {
  switch (action.type) {
    case Action::Type::Bid: {
      ensure_bidding_allowed(state, action.player_index);
      if (!action.suit.has_value()) {
        throw std::runtime_error("bid requires suit");
      }
      if (action.target_point_cards < 13 || action.target_point_cards > 19) {
        throw std::runtime_error("bid target out of range");
      }

      Bid candidate{action.player_index, *action.suit, action.target_point_cards};
      if (state.bidding->highest_bid.has_value() &&
          compare_bids(candidate, *state.bidding->highest_bid) <= 0) {
        throw std::runtime_error("bid too low");
      }

      state.current_player_index = next_player_index(action.player_index);
      state.bidding->highest_bid = candidate;
      state.bidding->consecutive_pass_count = 0;
      state.bidding->history.push_back(
          BiddingHistoryEntry{true, action.player_index, action.suit, action.target_point_cards});
      return;
    }
    case Action::Type::Pass: {
      ensure_bidding_allowed(state, action.player_index);
      state.bidding->consecutive_pass_count += 1;
      BiddingHistoryEntry history_entry;
      history_entry.is_bid = false;
      history_entry.player_index = action.player_index;
      state.bidding->history.push_back(history_entry);

      if (state.bidding->highest_bid.has_value() &&
          state.bidding->consecutive_pass_count == kPlayerCount - 1) {
        const Bid highest = *state.bidding->highest_bid;
        complete_bidding(
            state,
            Contract{highest.player_index, highest.suit, highest.target_point_cards});
        return;
      }

      if (!state.bidding->highest_bid.has_value() &&
          state.bidding->consecutive_pass_count == kPlayerCount) {
        complete_bidding(state, Contract{state.bidding->starter_player_index, Suit::Spades, 12});
        return;
      }

      state.current_player_index = next_player_index(action.player_index);
      return;
    }
    case Action::Type::ChooseAdjutant: {
      if (state.phase != Phase::ChoosingAdjutant || !state.contract.has_value()) {
        throw std::runtime_error("choose-adjutant is invalid in this state");
      }
      if (action.player_index != state.contract->napoleon_player_index) {
        throw std::runtime_error("only Napoleon can choose adjutant");
      }
      ensure_turn(state, action.player_index);
      if (state.adjutant.has_value()) {
        throw std::runtime_error("adjutant already chosen");
      }
      if (!state.current_trick.empty() || !state.completed_tricks.empty() || state.trick_number != 1 ||
          resolved_buried_card_count(state) != 0 || state.unused_cards.size() != 3 ||
          state.hands[static_cast<std::size_t>(action.player_index)].size() != kCardsPerPlayer) {
        throw std::runtime_error("invalid adjutant state");
      }

      state.adjutant =
          AdjutantState{action.card, resolve_adjutant_player_index(state, action.card), false};
      auto& napoleon_hand = state.hands[static_cast<std::size_t>(action.player_index)];
      napoleon_hand.insert(napoleon_hand.end(), state.unused_cards.begin(), state.unused_cards.end());
      state.unused_cards.clear();
      state.phase = Phase::Exchanging;
      state.current_player_index = action.player_index;
      return;
    }
    case Action::Type::DiscardCards: {
      if (state.phase != Phase::Exchanging || !state.contract.has_value() || !state.adjutant.has_value()) {
        throw std::runtime_error("discard is invalid in this state");
      }
      if (action.player_index != state.contract->napoleon_player_index) {
        throw std::runtime_error("only Napoleon can discard");
      }
      ensure_turn(state, action.player_index);
      if (resolved_buried_card_count(state) != 0 || action.cards.size() != 3) {
        throw std::runtime_error("invalid discard count");
      }

      std::unordered_set<std::uint8_t> ids;
      for (Card card : action.cards) {
        if (!ids.insert(card.id).second) {
          throw std::runtime_error("duplicate discarded card");
        }
      }

      auto& hand = state.hands[static_cast<std::size_t>(action.player_index)];
      if (hand.size() != 13) {
        throw std::runtime_error("Napoleon must have 13 cards before exchange");
      }

      std::vector<Card> awarded;
      std::vector<Card> excluded;
      for (Card card : action.cards) {
        if (std::none_of(hand.begin(), hand.end(), [&](Card candidate) { return candidate.id == card.id; })) {
          throw std::runtime_error("discarded card not in hand");
        }
        if (is_point_card(card)) {
          awarded.push_back(card);
        } else {
          excluded.push_back(card);
        }
      }
      for (Card card : action.cards) {
        erase_card(hand, card);
      }

      append_awarded_point_cards(state, action.player_index, awarded);
      state.excluded_cards.insert(state.excluded_cards.end(), excluded.begin(), excluded.end());
      state.latest_event = BuriedCardsResolvedEvent{
          action.player_index, awarded, static_cast<int>(excluded.size())};
      state.phase = Phase::Playing;
      state.current_player_index = action.player_index;
      state.current_trick.clear();
      state.trick_number = 1;
      state.is_trick_complete = false;
      state.is_game_over = false;
      state.result = std::nullopt;
      return;
    }
    case Action::Type::PlayCard: {
      if (state.phase != Phase::Playing || state.is_game_over) {
        throw std::runtime_error("play-card is invalid in this state");
      }
      if (state.is_trick_complete) {
        throw std::runtime_error("trick already complete");
      }
      if (!state.trump_suit.has_value()) {
        throw std::runtime_error("trump suit required");
      }
      ensure_turn(state, action.player_index);

      auto& hand = state.hands[static_cast<std::size_t>(action.player_index)];
      if (std::none_of(hand.begin(), hand.end(), [&](Card candidate) { return candidate.id == action.card.id; })) {
        throw std::runtime_error("card not in hand");
      }

      const std::vector<Card> playable = playable_cards(hand, state.current_trick, state.trump_suit);
      if (std::none_of(playable.begin(), playable.end(), [&](Card candidate) { return candidate.id == action.card.id; })) {
        throw std::runtime_error("must follow suit");
      }

      erase_card(hand, action.card);
      state.current_trick.push_back(PlayedCard{action.player_index, action.card});
      reveal_adjutant_if_needed(state, action.player_index, action.card);

      const bool trick_complete = state.current_trick.size() == kPlayerCount;
      const int winner = trick_complete
                             ? determine_trick_winner(state.current_trick, state.trump_suit, state.trick_number)
                             : next_player_index(action.player_index);
      if (trick_complete) {
        state.completed_tricks.push_back(
            CompletedTrick{state.trick_number, winner, state.current_trick});
      }

      state.is_trick_complete = trick_complete;
      state.current_player_index = winner;
      state.is_game_over = trick_complete && std::all_of(state.hands.begin(), state.hands.end(), [](const auto& candidate) {
        return candidate.empty();
      });
      if (state.is_game_over) {
        state.phase = Phase::Finished;
        if (state.adjutant.has_value() && state.adjutant->player_index.has_value()) {
          state.adjutant->revealed = true;
        }
        state.result = calculate_result(state);
      }
      return;
    }
    case Action::Type::AdvanceToNextTrick: {
      if (state.phase != Phase::Playing || !state.is_trick_complete || state.is_game_over) {
        throw std::runtime_error("cannot advance trick in this state");
      }
      state.current_trick.clear();
      state.trick_number += 1;
      state.is_trick_complete = false;
      return;
    }
  }
}

std::string canonical_snapshot_json(const GameState& state) {
  std::ostringstream out;
  out << "{\"phase\":";
  json_escape(out, phase_id(state.phase));
  out << ",\"currentPlayerId\":";
  json_escape(out, player_id(state.current_player_index));
  out << ",\"players\":[";
  for (int player_index = 0; player_index < kPlayerCount; ++player_index) {
    if (player_index != 0) {
      out << ',';
    }
    out << "{\"id\":";
    json_escape(out, player_id(player_index));
    out << ",\"handIds\":";
    write_array(out, state.hands[static_cast<std::size_t>(player_index)], write_card);
    out << '}';
  }
  out << "],\"currentTrick\":";
  write_array(out, state.current_trick, write_played_card);
  out << ",\"currentPlayerLegalActions\":";
  write_array(out, get_legal_actions(state, state.current_player_index), write_action);
  out << ",\"completedTricks\":";
  write_array(out, state.completed_tricks, write_completed_trick);
  out << ",\"trumpSuit\":";
  if (state.trump_suit.has_value()) {
    json_escape(out, suit_id(*state.trump_suit));
  } else {
    out << "null";
  }
  out << ",\"contract\":";
  write_contract(out, state.contract);
  out << ",\"adjutant\":";
  write_adjutant(out, state.adjutant);
  out << ",\"bidding\":";
  write_bidding(out, state.bidding);
  out << ",\"awardedPointCards\":";
  write_awards(out, state.awarded_point_cards);
  out << ",\"excludedCardIds\":";
  write_array(out, state.excluded_cards, write_card);
  out << ",\"unusedCardIds\":";
  write_array(out, state.unused_cards, write_card);
  out << ",\"latestEvent\":";
  write_latest_event(out, state.latest_event);
  out << ",\"result\":";
  write_result(out, state.result);
  out << ",\"trickNumber\":" << state.trick_number;
  out << ",\"isTrickComplete\":" << (state.is_trick_complete ? "true" : "false");
  out << ",\"isGameOver\":" << (state.is_game_over ? "true" : "false") << '}';
  return out.str();
}

Action parse_action_line(const std::string& line) {
  std::istringstream input(line);
  std::string type;
  input >> type;
  if (type.empty() || type[0] == '#') {
    throw std::runtime_error("empty action line");
  }

  if (type == "bid") {
    std::string player;
    std::string suit;
    int target = 0;
    input >> player >> suit >> target;
    Action action;
    action.type = Action::Type::Bid;
    action.player_index = parse_player_id(player);
    action.suit = parse_suit(suit);
    action.target_point_cards = target;
    return action;
  }

  if (type == "pass") {
    std::string player;
    input >> player;
    Action action;
    action.type = Action::Type::Pass;
    action.player_index = parse_player_id(player);
    return action;
  }

  if (type == "choose-adjutant") {
    std::string player;
    std::string card;
    input >> player >> card;
    Action action;
    action.type = Action::Type::ChooseAdjutant;
    action.player_index = parse_player_id(player);
    action.card = parse_card_id(card);
    return action;
  }

  if (type == "discard") {
    std::string player;
    input >> player;
    Action action;
    action.type = Action::Type::DiscardCards;
    action.player_index = parse_player_id(player);
    for (std::string card; input >> card;) {
      action.cards.push_back(parse_card_id(card));
    }
    return action;
  }

  if (type == "play") {
    std::string player;
    std::string card;
    input >> player >> card;
    Action action;
    action.type = Action::Type::PlayCard;
    action.player_index = parse_player_id(player);
    action.card = parse_card_id(card);
    return action;
  }

  if (type == "next-trick") {
    Action action;
    action.type = Action::Type::AdvanceToNextTrick;
    return action;
  }

  throw std::runtime_error("unknown action type: " + type);
}

std::string card_id(Card card) {
  if (is_joker(card)) {
    return "joker";
  }

  return std::string(kSuitIds[card.id / 13]) + "-" + kRankIds[card.id % 13];
}

Card parse_card_id(const std::string& value) {
  if (value == "joker") {
    return Card{52};
  }

  const std::size_t dash = value.find('-');
  if (dash == std::string::npos) {
    throw std::runtime_error("invalid card id: " + value);
  }

  const std::string suit_part = value.substr(0, dash);
  const std::string rank_part = value.substr(dash + 1);
  int suit_index = -1;
  int rank_index = -1;
  for (std::size_t index = 0; index < kSuitIds.size(); ++index) {
    if (suit_part == kSuitIds[index]) {
      suit_index = static_cast<int>(index);
      break;
    }
  }
  for (std::size_t index = 0; index < kRankIds.size(); ++index) {
    if (rank_part == kRankIds[index]) {
      rank_index = static_cast<int>(index);
      break;
    }
  }
  if (suit_index < 0 || rank_index < 0) {
    throw std::runtime_error("invalid card id: " + value);
  }

  return Card{static_cast<std::uint8_t>(suit_index * 13 + rank_index)};
}

std::string suit_id(Suit suit) {
  return kSuitIds[static_cast<std::size_t>(suit)];
}

Suit parse_suit(const std::string& value) {
  for (std::size_t index = 0; index < kSuitIds.size(); ++index) {
    if (value == kSuitIds[index]) {
      return static_cast<Suit>(index);
    }
  }
  throw std::runtime_error("invalid suit: " + value);
}

std::string phase_id(Phase phase) {
  switch (phase) {
    case Phase::Bidding:
      return "bidding";
    case Phase::Exchanging:
      return "exchanging";
    case Phase::ChoosingAdjutant:
      return "choosing-adjutant";
    case Phase::Playing:
      return "playing";
    case Phase::Finished:
      return "finished";
  }

  throw std::runtime_error("invalid phase");
}

}  // namespace napoleon
