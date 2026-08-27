#include "napoleon_parameterized_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace napoleon::parameterized_policy {
namespace {

constexpr double kEpsilon = 1e-12;

constexpr double flag(bool value) { return value ? 1.0 : 0.0; }

bool is_joker(Card card) { return card.id == 52; }
bool is_standard(Card card) { return card.id < 52; }
int suit_index(Card card) { return is_standard(card) ? card.id / 13 : -1; }
int rank_index(Card card) { return is_standard(card) ? card.id % 13 : -1; }
bool is_oruma(Card card) { return card.id == 0; }
bool is_yoromeki(Card card) { return card.id == 13 + 11; }
Card sei_jack(Suit trump) {
  return Card{static_cast<std::uint8_t>(static_cast<int>(trump) * 13 + 10)};
}
Card ura_jack(Suit trump) {
  switch (trump) {
    case Suit::Spades: return Card{static_cast<std::uint8_t>(Suit::Clubs) * 13 + 10};
    case Suit::Hearts: return Card{static_cast<std::uint8_t>(Suit::Diamonds) * 13 + 10};
    case Suit::Diamonds: return Card{static_cast<std::uint8_t>(Suit::Hearts) * 13 + 10};
    case Suit::Clubs: return Card{static_cast<std::uint8_t>(Suit::Spades) * 13 + 10};
  }
  throw std::runtime_error("invalid trump suit");
}
bool is_point(Card card) {
  const int rank = rank_index(card);
  return rank == 0 || rank >= 9;
}
bool is_high(Card card) {
  const int rank = rank_index(card);
  return rank == 0 || rank >= 10;
}
double ai_rank(Card card) {
  if (!is_standard(card)) return 0.0;
  static constexpr std::array<double, 13> values = {
      20.0, 12.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0};
  return values[static_cast<std::size_t>(rank_index(card))];
}
bool is_trump(Card card, Suit trump) {
  return is_standard(card) && suit_index(card) == static_cast<int>(trump);
}
bool contains_id(const std::vector<Card>& cards, std::uint8_t id) {
  return std::any_of(cards.begin(), cards.end(), [id](Card card) { return card.id == id; });
}

std::array<int, 4> suit_lengths(const std::vector<Card>& cards) {
  std::array<int, 4> result{};
  for (Card card : cards) {
    if (is_standard(card)) ++result[static_cast<std::size_t>(suit_index(card))];
  }
  return result;
}

double dot(const std::vector<double>& features, const std::vector<double>& weights, std::size_t offset) {
  double result = 0.0;
  for (std::size_t index = 0; index < features.size(); ++index) {
    result += features[index] * weights[offset + index];
  }
  return result;
}

void enumerate_combinations(
    const std::vector<Card>& hand,
    std::size_t start,
    std::vector<Card>& current,
    std::vector<std::vector<Card>>& output) {
  if (current.size() == 3) {
    output.push_back(current);
    return;
  }
  const std::size_t needed = 3 - current.size();
  for (std::size_t index = start; index + needed <= hand.size(); ++index) {
    current.push_back(hand[index]);
    enumerate_combinations(hand, index + 1, current, output);
    current.pop_back();
  }
}

void json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (char ch : value) {
    if (ch == '"' || ch == '\\') out << '\\';
    out << ch;
  }
  out << '"';
}

}  // namespace

const std::vector<FeatureDefinition>& feature_schema() {
  static const std::vector<FeatureDefinition> schema = {
      {"adjutant", "bias", 1, "candidate-independent intercept"},
      {"adjutant", "is_oruma", 1, "called card is spades ace"},
      {"adjutant", "is_sei_jack", 1, "called card is the regular jack"},
      {"adjutant", "is_ura_jack", 1, "called card is the reverse jack"},
      {"adjutant", "is_joker", 1, "called card is joker"},
      {"adjutant", "is_yoromeki", 1, "called card is hearts queen"},
      {"adjutant", "is_trump", 1, "called card belongs to trump suit"},
      {"adjutant", "is_point_card", 1, "called card is a point card"},
      {"adjutant", "is_self_held", 1, "Napoleon visibly holds the called card"},
      {"adjutant", "rule_based_eligible", 1, "existing RuleBased candidate eligibility"},
      {"adjutant", "eligible_generic_trump", 1, "eligible trump excluding three top specials"},
      {"adjutant", "candidate_rank", 20, "AI rank value divided by 20"},
      {"adjutant", "eligible_generic_rank", 20, "eligible generic AI rank divided by 20"},
      {"adjutant", "candidate_suit_length", 10, "own length in candidate suit divided by 10"},
      {"adjutant", "candidate_suit_point_count", 5, "own point cards in candidate suit divided by 5"},
      {"adjutant", "candidate_suit_high_count", 4, "own A/K/Q/J count in candidate suit divided by 4"},
      {"adjutant", "candidate_suit_has_ace", 1, "own hand contains candidate-suit ace"},
      {"adjutant", "candidate_suit_has_king", 1, "own hand contains candidate-suit king"},
      {"adjutant", "candidate_suit_void", 1, "own hand is void in candidate suit"},
      {"adjutant", "candidate_suit_singleton", 1, "own hand is singleton in candidate suit"},
      {"adjutant", "candidate_suit_doubleton", 1, "own hand is doubleton in candidate suit"},
      {"adjutant", "trump_length", 10, "own trump length divided by 10"},
      {"adjutant", "hand_point_count", 10, "own point-card count divided by 10"},
      {"adjutant", "hand_high_count", 10, "own A/K/Q/J count divided by 10"},
      {"adjutant", "hand_void_count", 4, "own void suit count divided by 4"},
      {"adjutant", "hand_singleton_count", 4, "own singleton suit count divided by 4"},
      {"adjutant", "target_level", 6, "contract target minus 13, divided by 6"},
      {"adjutant", "target_x_oruma", 6, "target level times oruma flag"},
      {"adjutant", "target_x_sei_jack", 6, "target level times regular-jack flag"},
      {"adjutant", "target_x_ura_jack", 6, "target level times reverse-jack flag"},
      {"adjutant", "target_x_trump", 6, "target level times trump flag"},
      {"adjutant", "ura_jack_x_trump_length", 10, "reverse-jack flag times normalized trump length"},
      {"adjutant", "public_bid_count", 10, "public bidding bid count divided by 10"},
      {"adjutant", "self_bid_count", 4, "Napoleon public bid count divided by 4"},
      {"adjutant", "target_x_trump_length", 60, "target level times normalized trump length"},

      {"exchange", "bias", 1, "discard-set intercept"},
      {"exchange", "buried_point_count", 3, "buried point-card count divided by 3"},
      {"exchange", "buried_trump_count", 3, "buried trump count divided by 3"},
      {"exchange", "remaining_trump_count", 10, "retained trump count divided by 10"},
      {"exchange", "buried_rank_sum", 60, "buried AI rank sum divided by 60"},
      {"exchange", "buried_ace_count", 3, "buried ace count divided by 3"},
      {"exchange", "buried_ten_count", 3, "buried ten count divided by 3"},
      {"exchange", "buried_jack_count", 3, "buried jack count divided by 3"},
      {"exchange", "buried_queen_count", 3, "buried queen count divided by 3"},
      {"exchange", "buried_king_count", 3, "buried king count divided by 3"},
      {"exchange", "buried_joker", 1, "joker is buried"},
      {"exchange", "buried_oruma", 1, "oruma is buried"},
      {"exchange", "buried_sei_jack", 1, "regular jack is buried"},
      {"exchange", "buried_ura_jack", 1, "reverse jack is buried"},
      {"exchange", "buried_yoromeki", 1, "yoromeki is buried"},
      {"exchange", "buried_called_adjutant", 1, "called adjutant card is buried"},
      {"exchange", "retained_point_count", 10, "retained point-card count divided by 10"},
      {"exchange", "retained_high_count", 10, "retained A/K/Q/J count divided by 10"},
      {"exchange", "retained_trump_count", 10, "retained trump count divided by 10"},
      {"exchange", "retained_spades_length", 10, "retained spades divided by 10"},
      {"exchange", "retained_hearts_length", 10, "retained hearts divided by 10"},
      {"exchange", "retained_diamonds_length", 10, "retained diamonds divided by 10"},
      {"exchange", "retained_clubs_length", 10, "retained clubs divided by 10"},
      {"exchange", "retained_spades_void", 1, "retained hand is void in spades"},
      {"exchange", "retained_hearts_void", 1, "retained hand is void in hearts"},
      {"exchange", "retained_diamonds_void", 1, "retained hand is void in diamonds"},
      {"exchange", "retained_clubs_void", 1, "retained hand is void in clubs"},
      {"exchange", "retained_spades_singleton", 1, "retained spades singleton"},
      {"exchange", "retained_hearts_singleton", 1, "retained hearts singleton"},
      {"exchange", "retained_diamonds_singleton", 1, "retained diamonds singleton"},
      {"exchange", "retained_clubs_singleton", 1, "retained clubs singleton"},
      {"exchange", "retained_spades_doubleton", 1, "retained spades doubleton"},
      {"exchange", "retained_hearts_doubleton", 1, "retained hearts doubleton"},
      {"exchange", "retained_diamonds_doubleton", 1, "retained diamonds doubleton"},
      {"exchange", "retained_clubs_doubleton", 1, "retained clubs doubleton"},
      {"exchange", "retained_void_count", 4, "retained void count divided by 4"},
      {"exchange", "retained_singleton_count", 4, "retained singleton count divided by 4"},
      {"exchange", "retained_doubleton_count", 4, "retained doubleton count divided by 4"},
      {"exchange", "retained_longest_suit", 10, "longest retained suit divided by 10"},
      {"exchange", "retained_shortest_nonzero_suit", 10, "shortest nonzero retained suit divided by 10"},
      {"exchange", "retained_nontrump_void_count", 3, "non-trump void count divided by 3"},
      {"exchange", "discard_same_suit_pair_count", 3, "same-suit pairs among discards divided by 3"},
      {"exchange", "discard_all_same_suit", 1, "all standard discards share a suit"},
      {"exchange", "kitty_buried_count", 3, "known kitty-origin buried count divided by 3"},
      {"exchange", "original_buried_count", 3, "original-hand buried count divided by 3"},
      {"exchange", "target_level", 6, "contract target minus 13, divided by 6"},
      {"exchange", "buried_point_x_target", 18, "buried point count times target level"},
      {"exchange", "buried_trump_x_target", 18, "buried trump count times target level"},
      {"exchange", "remaining_trump_x_target", 60, "remaining trump count times target level"},
      {"exchange", "buried_called_x_target", 6, "called-card burial times target level"},
      {"exchange", "kitty_buried_x_point", 9, "kitty burial count times point burial count"},
      {"exchange", "kitty_buried_x_trump", 9, "kitty burial count times trump burial count"},
      {"exchange", "rule_based_regular_rank_sum", 60, "RuleBased non-special buried rank sum divided by 60"},
      {"exchange", "rule_based_regular_trump_count", 3, "RuleBased non-special buried trump count divided by 3"},
      {"exchange", "target_x_oruma", 6, "target level times oruma burial"},
      {"exchange", "target_x_sei_jack", 6, "target level times regular-jack burial"},
      {"exchange", "target_x_ura_jack", 6, "target level times reverse-jack burial"},
      {"exchange", "target_x_joker", 6, "target level times joker burial"},
      {"exchange", "target_x_yoromeki", 6, "target level times yoromeki burial"},
      {"exchange", "buried_yoromeki_trump", 1, "buried yoromeki belongs to trump suit"},
  };
  if (schema.size() != static_cast<std::size_t>(kParameterCount)) {
    throw std::runtime_error("parameterized feature schema dimension mismatch");
  }
  return schema;
}

Parameters initial_rule_based_parameters() {
  Parameters result;
  result.values.assign(kParameterCount, 0.0);
  result.values[1] = 60.0;
  result.values[2] = 55.0;
  result.values[3] = 50.5;
  result.values[9] = 100.0;
  result.values[10] = 30.0;
  result.values[12] = 20.0;
  result.values[31] = 10.0;
  const std::size_t offset = kAdjutantFeatureCount;
  result.values[offset + 10] = -20.0;
  result.values[offset + 11] = -60.0;
  result.values[offset + 12] = -55.0;
  result.values[offset + 13] = -50.0;
  result.values[offset + 14] = -30.0;
  result.values[offset + 52] = -60.0;
  result.values[offset + 53] = -90.0;
  result.values[offset + 59] = -15.0;
  return result;
}

void validate_parameters(const Parameters& parameters) {
  if (parameters.values.size() != static_cast<std::size_t>(kParameterCount)) {
    throw std::runtime_error("expected " + std::to_string(kParameterCount) + " parameters");
  }
  if (std::any_of(parameters.values.begin(), parameters.values.end(), [](double value) {
        return !std::isfinite(value);
      })) {
    throw std::runtime_error("parameters must be finite");
  }
}

std::vector<double> extract_adjutant_features(
    const GameState& state,
    int player_index,
    Card candidate) {
  if (state.phase != Phase::ChoosingAdjutant || !state.contract.has_value()) {
    throw std::runtime_error("adjutant features require choosing-adjutant state");
  }
  const auto& hand = state.hands.at(static_cast<std::size_t>(player_index));
  const Suit trump = state.contract->trump_suit;
  const bool oruma = is_oruma(candidate);
  const bool sei = candidate.id == sei_jack(trump).id;
  const bool ura = candidate.id == ura_jack(trump).id;
  const bool held = contains_id(hand, candidate.id);
  const bool eligible = is_standard(candidate) && !held && !is_yoromeki(candidate) &&
      (oruma || sei || ura || is_trump(candidate, trump));
  const bool generic = eligible && !oruma && !sei && !ura;
  const auto lengths = suit_lengths(hand);
  const int candidate_suit = suit_index(candidate);
  const int candidate_length = candidate_suit >= 0 ? lengths[static_cast<std::size_t>(candidate_suit)] : 0;
  int candidate_points = 0;
  int candidate_highs = 0;
  bool has_ace = false;
  bool has_king = false;
  int hand_points = 0;
  int hand_highs = 0;
  for (Card card : hand) {
    hand_points += is_point(card) ? 1 : 0;
    hand_highs += is_high(card) ? 1 : 0;
    if (suit_index(card) == candidate_suit) {
      candidate_points += is_point(card) ? 1 : 0;
      candidate_highs += is_high(card) ? 1 : 0;
      has_ace = has_ace || rank_index(card) == 0;
      has_king = has_king || rank_index(card) == 12;
    }
  }
  const int voids = static_cast<int>(std::count(lengths.begin(), lengths.end(), 0));
  const int singletons = static_cast<int>(std::count(lengths.begin(), lengths.end(), 1));
  const int trump_length = lengths[static_cast<std::size_t>(trump)];
  const double target = (state.contract->target_point_cards - 13) / 6.0;
  int bid_count = 0;
  int self_bid_count = 0;
  for (const auto& entry : state.public_bidding_history) {
    if (entry.is_bid) {
      ++bid_count;
      self_bid_count += entry.player_index == player_index ? 1 : 0;
    }
  }
  return {
      1.0, flag(oruma), flag(sei), flag(ura), flag(is_joker(candidate)),
      flag(is_yoromeki(candidate)), flag(is_trump(candidate, trump)),
      flag(is_point(candidate)), flag(held), flag(eligible), flag(generic),
      ai_rank(candidate) / 20.0, generic ? ai_rank(candidate) / 20.0 : 0.0,
      candidate_length / 10.0, candidate_points / 5.0, candidate_highs / 4.0,
      flag(has_ace), flag(has_king), flag(candidate_suit >= 0 && candidate_length == 0),
      flag(candidate_suit >= 0 && candidate_length == 1),
      flag(candidate_suit >= 0 && candidate_length == 2),
      trump_length / 10.0, hand_points / 10.0, hand_highs / 10.0,
      voids / 4.0, singletons / 4.0, target, target * oruma, target * sei,
      target * ura, target * is_trump(candidate, trump), ura ? trump_length / 10.0 : 0.0,
      bid_count / 10.0, self_bid_count / 4.0, target * trump_length / 10.0};
}

std::vector<double> extract_exchange_features(
    const GameState& state,
    int player_index,
    const std::vector<Card>& discarded,
    const std::vector<std::uint8_t>& kitty_card_ids) {
  if (state.phase != Phase::Exchanging || !state.contract.has_value() || !state.adjutant.has_value() ||
      discarded.size() != 3) {
    throw std::runtime_error("exchange features require exchange state and three discards");
  }
  const Suit trump = state.contract->trump_suit;
  const auto& hand = state.hands.at(static_cast<std::size_t>(player_index));
  std::unordered_set<std::uint8_t> discard_ids;
  for (Card card : discarded) discard_ids.insert(card.id);
  std::vector<Card> retained;
  for (Card card : hand) if (discard_ids.count(card.id) == 0) retained.push_back(card);
  if (retained.size() != 10) throw std::runtime_error("exchange candidate must retain ten cards");

  int buried_points = 0, buried_trumps = 0, retained_points = 0, retained_highs = 0;
  int buried_aces = 0, buried_tens = 0, buried_jacks = 0, buried_queens = 0, buried_kings = 0;
  int kitty_buried = 0, regular_trumps = 0;
  double rank_sum = 0.0, regular_rank_sum = 0.0;
  bool joker = false, oruma = false, sei = false, ura = false, yoro = false, called = false;
  const Card sei_card = sei_jack(trump);
  const Card ura_card = ura_jack(trump);
  for (Card card : discarded) {
    buried_points += is_point(card) ? 1 : 0;
    buried_trumps += is_trump(card, trump) ? 1 : 0;
    rank_sum += ai_rank(card);
    const int rank = rank_index(card);
    buried_aces += rank == 0 ? 1 : 0;
    buried_tens += rank == 9 ? 1 : 0;
    buried_jacks += rank == 10 ? 1 : 0;
    buried_queens += rank == 11 ? 1 : 0;
    buried_kings += rank == 12 ? 1 : 0;
    joker = joker || is_joker(card);
    oruma = oruma || is_oruma(card);
    sei = sei || card.id == sei_card.id;
    ura = ura || card.id == ura_card.id;
    yoro = yoro || is_yoromeki(card);
    called = called || card.id == state.adjutant->called_card.id;
    kitty_buried += std::find(kitty_card_ids.begin(), kitty_card_ids.end(), card.id) != kitty_card_ids.end();
    const bool special = is_joker(card) || is_oruma(card) || card.id == sei_card.id ||
        card.id == ura_card.id || is_yoromeki(card);
    if (!special) {
      regular_rank_sum += ai_rank(card);
      regular_trumps += is_trump(card, trump) ? 1 : 0;
    }
  }
  for (Card card : retained) {
    retained_points += is_point(card) ? 1 : 0;
    retained_highs += is_high(card) ? 1 : 0;
  }
  const auto lengths = suit_lengths(retained);
  const int retained_trumps = lengths[static_cast<std::size_t>(trump)];
  const int voids = static_cast<int>(std::count(lengths.begin(), lengths.end(), 0));
  const int singletons = static_cast<int>(std::count(lengths.begin(), lengths.end(), 1));
  const int doubletons = static_cast<int>(std::count(lengths.begin(), lengths.end(), 2));
  const int longest = *std::max_element(lengths.begin(), lengths.end());
  int shortest_nonzero = 10;
  for (int length : lengths) if (length > 0) shortest_nonzero = std::min(shortest_nonzero, length);
  int nontrump_voids = 0;
  for (int suit = 0; suit < 4; ++suit) {
    if (suit != static_cast<int>(trump) && lengths[static_cast<std::size_t>(suit)] == 0) ++nontrump_voids;
  }
  int same_suit_pairs = 0;
  for (int left = 0; left < 3; ++left) for (int right = left + 1; right < 3; ++right) {
    same_suit_pairs += is_standard(discarded[left]) && is_standard(discarded[right]) &&
        suit_index(discarded[left]) == suit_index(discarded[right]);
  }
  const bool all_same = std::all_of(discarded.begin(), discarded.end(), [&](Card card) {
    return is_standard(card) && suit_index(card) == suit_index(discarded.front());
  });
  const double target = (state.contract->target_point_cards - 13) / 6.0;
  std::vector<double> features = {
      1.0, buried_points / 3.0, buried_trumps / 3.0, retained_trumps / 10.0,
      rank_sum / 60.0, buried_aces / 3.0, buried_tens / 3.0, buried_jacks / 3.0,
      buried_queens / 3.0, buried_kings / 3.0, flag(joker), flag(oruma), flag(sei),
      flag(ura), flag(yoro), flag(called),
      retained_points / 10.0, retained_highs / 10.0, retained_trumps / 10.0};
  for (int length : lengths) features.push_back(length / 10.0);
  for (int length : lengths) features.push_back(length == 0);
  for (int length : lengths) features.push_back(length == 1);
  for (int length : lengths) features.push_back(length == 2);
  features.insert(features.end(), {
      voids / 4.0, singletons / 4.0, doubletons / 4.0, longest / 10.0,
      shortest_nonzero / 10.0, nontrump_voids / 3.0, same_suit_pairs / 3.0, flag(all_same),
      kitty_buried / 3.0, (3 - kitty_buried) / 3.0, target,
      (buried_points / 3.0) * target, (buried_trumps / 3.0) * target,
      (retained_trumps / 10.0) * target, called * target,
      (kitty_buried / 3.0) * (buried_points / 3.0),
      (kitty_buried / 3.0) * (buried_trumps / 3.0),
      regular_rank_sum / 60.0, regular_trumps / 3.0,
      target * oruma, target * sei, target * ura, target * joker, target * yoro,
      flag(yoro && is_trump(Card{static_cast<std::uint8_t>(13 + 11)}, trump))});
  if (features.size() != static_cast<std::size_t>(kExchangeFeatureCount)) {
    throw std::runtime_error("exchange feature dimension mismatch");
  }
  return features;
}

SelectionResult select_adjutant(
    const GameState& state,
    int player_index,
    const Parameters& parameters) {
  validate_parameters(parameters);
  SelectionResult best;
  best.score = -std::numeric_limits<double>::infinity();
  bool found = false;
  for (const Action& action : get_legal_actions(state, player_index)) {
    if (action.type != Action::Type::ChooseAdjutant) continue;
    std::vector<double> features = extract_adjutant_features(state, player_index, action.card);
    const double score = dot(features, parameters.values, 0);
    if (!found || score > best.score + kEpsilon ||
        (std::fabs(score - best.score) <= kEpsilon && action.card.id < best.action.card.id)) {
      best = SelectionResult{action, score, std::move(features), false};
      found = true;
    }
  }
  if (!found) throw std::runtime_error("no legal adjutant action");
  return best;
}

SelectionResult select_exchange(
    const GameState& state,
    int player_index,
    const std::vector<std::uint8_t>& kitty_card_ids,
    const Parameters& parameters) {
  validate_parameters(parameters);
  const auto& hand = state.hands.at(static_cast<std::size_t>(player_index));
  std::vector<std::vector<Card>> combinations;
  std::vector<Card> current;
  enumerate_combinations(hand, 0, current, combinations);
  SelectionResult best;
  best.score = -std::numeric_limits<double>::infinity();
  bool found = false;
  for (const auto& discarded : combinations) {
    std::vector<double> features =
        extract_exchange_features(state, player_index, discarded, kitty_card_ids);
    const double score = dot(features, parameters.values, kAdjutantFeatureCount);
    if (!found || score > best.score + kEpsilon) {
      Action action;
      action.type = Action::Type::DiscardCards;
      action.player_index = player_index;
      action.cards = discarded;
      best = SelectionResult{action, score, std::move(features), false};
      found = true;
    }
  }
  if (!found) throw std::runtime_error("no exchange combination");
  return best;
}

std::string feature_schema_json() {
  std::ostringstream out;
  out << "{\"schemaVersion\":" << kFeatureSchemaVersion
      << ",\"adjutantFeatureCount\":" << kAdjutantFeatureCount
      << ",\"exchangeFeatureCount\":" << kExchangeFeatureCount
      << ",\"parameterCount\":" << kParameterCount << ",\"features\":[";
  const auto& schema = feature_schema();
  for (std::size_t index = 0; index < schema.size(); ++index) {
    if (index) out << ',';
    out << "{\"index\":" << index << ",\"block\":";
    json_string(out, schema[index].block);
    out << ",\"name\":";
    json_string(out, schema[index].name);
    out << ",\"scale\":" << schema[index].scale << ",\"description\":";
    json_string(out, schema[index].description);
    out << '}';
  }
  out << "]}";
  return out.str();
}

std::string parameters_json(const Parameters& parameters) {
  validate_parameters(parameters);
  std::ostringstream out;
  out << std::setprecision(17) << "{\"schemaVersion\":" << kFeatureSchemaVersion
      << ",\"parameterCount\":" << kParameterCount << ",\"weights\":[";
  for (std::size_t index = 0; index < parameters.values.size(); ++index) {
    if (index) out << ',';
    out << parameters.values[index];
  }
  out << "]}";
  return out.str();
}

}  // namespace napoleon::parameterized_policy
