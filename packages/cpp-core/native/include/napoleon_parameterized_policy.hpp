#pragma once

#include "napoleon_core.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace napoleon::parameterized_policy {

constexpr int kAdjutantFeatureCount = 35;
constexpr int kExchangeFeatureCount = 60;
constexpr int kParameterCount = kAdjutantFeatureCount + kExchangeFeatureCount;
constexpr int kFeatureSchemaVersion = 1;

struct FeatureDefinition {
  std::string block;
  std::string name;
  double scale = 1.0;
  std::string description;
};

struct Parameters {
  std::vector<double> values;
};

struct SelectionResult {
  Action action;
  double score = 0.0;
  std::vector<double> features;
  bool fallback = false;
};

const std::vector<FeatureDefinition>& feature_schema();
Parameters initial_rule_based_parameters();
void validate_parameters(const Parameters& parameters);

std::vector<double> extract_adjutant_features(
    const GameState& state,
    int player_index,
    Card candidate);

std::vector<double> extract_exchange_features(
    const GameState& state,
    int player_index,
    const std::vector<Card>& discarded,
    const std::vector<std::uint8_t>& kitty_card_ids);

SelectionResult select_adjutant(
    const GameState& state,
    int player_index,
    const Parameters& parameters);

SelectionResult select_exchange(
    const GameState& state,
    int player_index,
    const std::vector<std::uint8_t>& kitty_card_ids,
    const Parameters& parameters);

std::string feature_schema_json();
std::string parameters_json(const Parameters& parameters);

}  // namespace napoleon::parameterized_policy
