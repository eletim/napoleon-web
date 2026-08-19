#include "napoleon_core.hpp"
#include "napoleon_observation.hpp"
#include "napoleon_onnx_policy.hpp"
#include "napoleon_roster.hpp"
#include "napoleon_simulation_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using napoleon::Action;
using napoleon::AgentIdentity;
using napoleon::AgentType;
using napoleon::FinishedGame;
using napoleon::GameResult;
using napoleon::Phase;
using napoleon::RosterAssignment;

constexpr int kPlayerCount = napoleon::kPlayerCount;
constexpr int kBiddingActionCount = napoleon::observation::kBiddingActionCount;
constexpr int kBiddingFeatureCount = napoleon::observation::kBiddingModelInputFeatureCount;
constexpr int kPlayingFeatureCount = napoleon::observation::kPlayingModelInputFeatureCount;

struct Sha256 {
  std::array<std::uint32_t, 8> state{
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<std::uint8_t, 64> buffer{};
  std::uint64_t bit_length = 0;
  std::size_t buffer_length = 0;

  void update(const std::uint8_t* data, std::size_t length) {
    bit_length += static_cast<std::uint64_t>(length) * 8u;
    for (std::size_t index = 0; index < length; ++index) {
      buffer[buffer_length++] = data[index];
      if (buffer_length == buffer.size()) {
        transform(buffer.data());
        buffer_length = 0;
      }
    }
  }

  void update(const std::string& value) {
    update(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
  }

  std::string hexdigest() {
    const std::uint64_t final_bit_length = bit_length;
    buffer[buffer_length++] = 0x80u;
    if (buffer_length > 56) {
      while (buffer_length < 64) {
        buffer[buffer_length++] = 0;
      }
      transform(buffer.data());
      buffer_length = 0;
    }
    while (buffer_length < 56) {
      buffer[buffer_length++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer[buffer_length++] = static_cast<std::uint8_t>((final_bit_length >> shift) & 0xffu);
    }
    transform(buffer.data());

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state) {
      out << std::setw(8) << word;
    }
    return out.str();
  }

  static std::uint32_t rotr(std::uint32_t value, int shift) {
    return (value >> shift) | (value << (32 - shift));
  }

  void transform(const std::uint8_t* chunk) {
    static constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::array<std::uint32_t, 64> w{};
    for (std::size_t index = 0; index < 16; ++index) {
      w[index] = (static_cast<std::uint32_t>(chunk[index * 4]) << 24) |
                 (static_cast<std::uint32_t>(chunk[index * 4 + 1]) << 16) |
                 (static_cast<std::uint32_t>(chunk[index * 4 + 2]) << 8) |
                 static_cast<std::uint32_t>(chunk[index * 4 + 3]);
    }
    for (std::size_t index = 16; index < 64; ++index) {
      const std::uint32_t s0 =
          rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10);
      w[index] = w[index - 16] + s0 + w[index - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];
    for (std::size_t index = 0; index < 64; ++index) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + k[index] + w[index];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
};

struct Options {
  std::filesystem::path output_directory;
  std::filesystem::path bidding_onnx_path;
  std::filesystem::path bidding_metadata_path;
  std::string bidding_artifact_id = "candidate-bidding";
  std::filesystem::path playing_onnx_path;
  std::filesystem::path playing_metadata_path;
  std::string playing_artifact_id = "ppo-separated-v1000";
  std::uint32_t start_seed = 0;
  std::uint32_t game_count = 1;
  std::uint32_t games_per_shard = 100;
  std::uint32_t max_concurrent_games = 256;
  std::uint32_t inference_max_batch_size = 256;
  std::uint32_t sampling_seed = 0;
  double temperature = 1.0;
  std::string inference_device = "cpu";
  std::string policy_backend = "onnx";
};

struct Draft {
  std::uint32_t seed = 0;
  std::uint64_t step = 0;
  int acting_player_index = 0;
  int candidate_seat_index = 0;
  std::vector<float> model_input;
  std::vector<int> legal_bid_mask;
  int selected_action_index = 0;
  double behavior_log_probability = 0.0;
};

struct Shard {
  std::string file;
  std::uint32_t start_seed = 0;
  std::uint32_t end_seed = 0;
  std::uint32_t game_count = 0;
  std::uint64_t sample_count = 0;
  std::uint64_t byte_length = 0;
  std::string sha256;
};

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

std::string json_string(const std::string& value) {
  std::ostringstream out;
  json_escape(out, value);
  return out.str();
}

std::uint32_t parse_uint32(const std::string& value, const std::string& name) {
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error(name + " must be an integer between 0 and 4294967295");
  }
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(name + " must be an integer between 0 and 4294967295");
  }
  return static_cast<std::uint32_t>(parsed);
}

template <typename T>
void write_array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << values[index];
  }
  out << ']';
}

std::string player_id(int player_index) {
  return "player-" + std::to_string(player_index);
}

std::string sha256_bytes(const std::string& value) {
  Sha256 sha;
  sha.update(value);
  return sha.hexdigest();
}

std::string sha256_file(const std::filesystem::path& path) {
  if (path.empty()) {
    return "";
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open file for sha256: " + path.string());
  }
  Sha256 sha;
  std::array<char, 8192> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      sha.update(
          reinterpret_cast<const std::uint8_t*>(buffer.data()),
          static_cast<std::size_t>(count));
    }
  }
  return sha.hexdigest();
}

void write_card_ids(std::ostream& out) {
  out << '[';
  constexpr std::array<const char*, 4> suits{"spades", "hearts", "diamonds", "clubs"};
  constexpr std::array<const char*, 13> ranks{
      "A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"};
  bool first = true;
  for (const char* suit : suits) {
    for (const char* rank : ranks) {
      if (!first) {
        out << ',';
      }
      first = false;
      json_escape(out, std::string(suit) + "-" + rank);
    }
  }
  if (!first) {
      out << ',';
  }
  json_escape(out, "joker");
  out << ']';
}

void write_policy_artifact(
    std::ostream& out,
    const std::string& type,
    const std::string& artifact_id,
    const std::filesystem::path& onnx_path,
    const std::filesystem::path& metadata_path,
    const std::string& requested_inference_device,
    const std::string& resolved_inference_device,
    const std::string& execution_provider) {
  out << "{\"type\":" << json_string(type)
      << ",\"artifactId\":" << json_string(artifact_id)
      << ",\"onnxPath\":" << json_string(onnx_path.string())
      << ",\"metadataPath\":" << json_string(metadata_path.string())
      << ",\"onnxFileName\":" << json_string(onnx_path.filename().string())
      << ",\"metadataFileName\":" << json_string(metadata_path.filename().string())
      << ",\"onnxSha256\":" << json_string(sha256_file(onnx_path))
      << ",\"metadataSha256\":" << json_string(sha256_file(metadata_path))
      << ",\"requestedInferenceDevice\":" << json_string(requested_inference_device)
      << ",\"resolvedInferenceDevice\":" << json_string(resolved_inference_device)
      << ",\"executionProvider\":" << json_string(execution_provider)
      << ",\"metadata\":{}}";
}

std::uint32_t sha256_first_u32_be(const std::string& value) {
  return static_cast<std::uint32_t>(std::stoul(sha256_bytes(value).substr(0, 8), nullptr, 16));
}

AgentIdentity frozen_bidding_agent(std::uint32_t seed, int candidate_seat, int player_index) {
  const std::string key =
      "per-seat-seeded-rule-based-conservative-50-50-v1:" + std::to_string(seed) + ":" +
      std::to_string(candidate_seat) + ":" + std::to_string(player_index);
  return (sha256_first_u32_be(key) % 2u) == 0u
      ? napoleon::rule_based_agent("rule-based-bidding")
      : napoleon::rule_based_agent("conservative-bidding");
}

void write_bidding_baseline_policy(std::ostream& out, const AgentIdentity& agent) {
  if (agent.id == "rule-based-bidding") {
    out << "{\"type\":\"rule-based-bidding\",\"id\":\"rule-based-bidding-v1\",\"version\":1}";
    return;
  }
  if (agent.id == "conservative-bidding") {
    out << "{\"type\":\"conservative-bidding\",\"id\":\"conservative-bidding-v1\"}";
    return;
  }
  throw std::runtime_error("unknown frozen bidding agent id: " + agent.id);
}

void write_frozen_bidding_mix_metadata(std::ostream& out) {
  out << "{\"type\":\"mixed-frozen-bidding\""
      << ",\"mixingRuleVersion\":\"per-seat-seeded-rule-based-conservative-50-50-v1\""
      << ",\"selectionUnit\":\"game-seat\""
      << ",\"ruleBasedWeight\":0.5"
      << ",\"conservativeWeight\":0.5"
      << ",\"policies\":{\"ruleBased\":{\"type\":\"rule-based-bidding\","
         "\"id\":\"rule-based-bidding-v1\",\"version\":1},"
         "\"conservative\":{\"type\":\"conservative-bidding\","
         "\"id\":\"conservative-bidding-v1\"}}}";
}

void write_frozen_bidding_mix_diagnostics(
    std::ostream& out,
    std::uint32_t start_seed,
    std::uint32_t game_count) {
  int rule_based_count = 0;
  int conservative_count = 0;
  std::vector<std::tuple<std::uint32_t, int, int, AgentIdentity>> assignments;
  assignments.reserve(static_cast<std::size_t>(game_count) * kPlayerCount * (kPlayerCount - 1));
  for (std::uint32_t offset = 0; offset < game_count; ++offset) {
    const std::uint32_t seed = start_seed + offset;
    for (int candidate_seat = 0; candidate_seat < kPlayerCount; ++candidate_seat) {
      for (int player_index = 0; player_index < kPlayerCount; ++player_index) {
        if (player_index == candidate_seat) {
          continue;
        }
        AgentIdentity agent = frozen_bidding_agent(seed, candidate_seat, player_index);
        if (agent.id == "rule-based-bidding") {
          ++rule_based_count;
        } else {
          ++conservative_count;
        }
        assignments.push_back({seed, candidate_seat, player_index, std::move(agent)});
      }
    }
  }

  out << "{\"type\":\"mixed-frozen-bidding\""
      << ",\"mixingRuleVersion\":\"per-seat-seeded-rule-based-conservative-50-50-v1\""
      << ",\"selectionUnit\":\"game-seat\""
      << ",\"ruleBasedWeight\":0.5"
      << ",\"conservativeWeight\":0.5"
      << ",\"ruleBasedSeatCount\":" << rule_based_count
      << ",\"conservativeSeatCount\":" << conservative_count
      << ",\"policies\":{\"ruleBased\":{\"type\":\"rule-based-bidding\","
         "\"id\":\"rule-based-bidding-v1\",\"version\":1},"
         "\"conservative\":{\"type\":\"conservative-bidding\","
         "\"id\":\"conservative-bidding-v1\"}}"
      << ",\"seatAssignments\":[";
  for (std::size_t index = 0; index < assignments.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const auto& [seed, candidate_seat, player_index, agent] = assignments[index];
    out << "{\"seed\":" << seed
        << ",\"candidateSeatIndex\":" << candidate_seat
        << ",\"rotationOffset\":" << candidate_seat
        << ",\"playerIndex\":" << player_index
        << ",\"policy\":";
    write_bidding_baseline_policy(out, agent);
    out << '}';
  }
  out << "]}";
}

RosterAssignment nonplaying_roster(std::uint32_t seed, int candidate_seat) {
  RosterAssignment roster;
  roster.current_seat_index = candidate_seat;
  for (int player_index = 0; player_index < kPlayerCount; ++player_index) {
    roster.agents[static_cast<std::size_t>(player_index)] =
        player_index == candidate_seat
            ? napoleon::current_policy_agent("candidate-bidding")
            : frozen_bidding_agent(seed, candidate_seat, player_index);
  }
  return roster;
}

int raw_reward_for_player(const GameResult& result, int player_index) {
  if (result.result_type == "all-pass") {
    return 0;
  }
  const bool napoleon_won = result.winner == "napoleon-team";
  const int d = result.target_point_cards;
  const bool is_napoleon = player_index == result.napoleon_player_index;
  const bool is_adjutant =
      result.adjutant_player_index.has_value() && player_index == *result.adjutant_player_index;
  const bool is_napoleon_adjutant =
      is_napoleon && (!result.adjutant_player_index.has_value() || is_adjutant);
  if (is_napoleon_adjutant) {
    return napoleon_won ? 3 * d : -5;
  }
  if (is_napoleon) {
    return napoleon_won ? 2 * d : -5;
  }
  if (is_adjutant) {
    return napoleon_won ? d : 0;
  }
  return napoleon_won ? d : 0;
}

std::string role_for_player(const GameResult& result, int player_index) {
  if (result.result_type == "all-pass") {
    return player_index == result.starter_player_index ? "all-pass-starter" : "all-pass-other";
  }
  const bool is_napoleon = player_index == result.napoleon_player_index;
  const bool is_adjutant =
      result.adjutant_player_index.has_value() && player_index == *result.adjutant_player_index;
  const bool is_napoleon_adjutant =
      is_napoleon && (!result.adjutant_player_index.has_value() || is_adjutant);
  if (is_napoleon_adjutant) {
    return "napoleon-adjutant";
  }
  if (is_napoleon) {
    return "napoleon";
  }
  if (is_adjutant) {
    return "adjutant";
  }
  return "citizen";
}

void write_outcome(std::ostream& out, const GameResult& result, int player_index) {
  if (result.result_type == "all-pass") {
    out << "{\"outcomeType\":\"all-pass\",\"starterPlayerId\":"
        << json_string(player_id(result.starter_player_index))
        << ",\"actingPlayerRole\":" << json_string(role_for_player(result, player_index))
        << ",\"actingPlayerPayoff\":0}";
    return;
  }
  out << "{\"outcomeType\":\"standard\",\"winner\":" << json_string(result.winner)
      << ",\"trumpSuit\":\""
      << napoleon::suit_id(result.trump_suit.value_or(static_cast<napoleon::Suit>(0))) << "\"";
  out << ",\"targetPointCards\":" << result.target_point_cards
      << ",\"napoleonPlayerId\":" << json_string(player_id(result.napoleon_player_index))
      << ",\"actingPlayerRole\":" << json_string(role_for_player(result, player_index)) << "}";
}

void write_sample(std::ostream& out, const Draft& draft, const GameResult& result) {
  std::array<int, kPlayerCount> raw_rewards{};
  for (int player = 0; player < kPlayerCount; ++player) {
    raw_rewards[static_cast<std::size_t>(player)] = raw_reward_for_player(result, player);
  }
  const double mean = std::accumulate(raw_rewards.begin(), raw_rewards.end(), 0.0) / kPlayerCount;
  const int raw = raw_rewards[static_cast<std::size_t>(draft.acting_player_index)];

  out << "{\"sampleType\":\"non-playing-bidding-rl-sample\",\"schemaVersion\":4";
  out << ",\"seed\":" << draft.seed;
  out << ",\"step\":" << draft.step;
  out << ",\"phase\":\"bidding\"";
  out << ",\"actingPlayerId\":" << json_string(player_id(draft.acting_player_index));
  out << ",\"actingPlayerIndex\":" << draft.acting_player_index;
  out << ",\"candidateSeatIndex\":" << draft.candidate_seat_index;
  out << ",\"rotationOffset\":" << draft.candidate_seat_index;
  out << ",\"relativePlayerIds\":[";
  for (int offset = 0; offset < kPlayerCount; ++offset) {
    if (offset != 0) {
      out << ',';
    }
    json_escape(out, player_id((draft.acting_player_index + offset) % kPlayerCount));
  }
  out << "],\"modelInput\":";
  write_array(out, draft.model_input);
  out << ",\"legalBidMask\":";
  write_array(out, draft.legal_bid_mask);
  out << ",\"selectedActionIndex\":" << draft.selected_action_index;
  out << ",\"behaviorLogProbability\":" << std::setprecision(17) << draft.behavior_log_probability;
  out << ",\"rawTerminalReward\":" << raw;
  out << ",\"gameMeanRawTerminalReward\":" << mean;
  out << ",\"terminalReward\":" << (static_cast<double>(raw) - mean);
  out << ",\"outcome\":";
  write_outcome(out, result, draft.acting_player_index);
  out << "}\n";
}

std::string shard_name(std::size_t index) {
  std::ostringstream out;
  out << "shard-" << std::setw(5) << std::setfill('0') << index << ".jsonl";
  return out.str();
}

napoleon::onnx_policy::InferenceDevice parse_device(const std::string& value) {
  if (value == "cpu" || value == "auto") {
    return napoleon::onnx_policy::InferenceDevice::Cpu;
  }
  if (value == "cuda") {
    return napoleon::onnx_policy::InferenceDevice::Cuda;
  }
  throw std::runtime_error("--inference-device must be cpu, auto, or cuda");
}

Options parse_args(int argc, char** argv) {
  Options options;
  auto need = [&](int& index, const std::string& name) -> std::string {
    if (index + 1 >= argc) {
      throw std::runtime_error(name + " requires a value");
    }
    return argv[++index];
  };
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--output") options.output_directory = need(index, arg);
    else if (arg == "--bidding-onnx") options.bidding_onnx_path = need(index, arg);
    else if (arg == "--bidding-metadata") options.bidding_metadata_path = need(index, arg);
    else if (arg == "--bidding-artifact-id") options.bidding_artifact_id = need(index, arg);
    else if (arg == "--playing-onnx") options.playing_onnx_path = need(index, arg);
    else if (arg == "--playing-metadata") options.playing_metadata_path = need(index, arg);
    else if (arg == "--playing-artifact-id") options.playing_artifact_id = need(index, arg);
    else if (arg == "--start-seed") options.start_seed = parse_uint32(need(index, arg), "start-seed");
    else if (arg == "--game-count") options.game_count = parse_uint32(need(index, arg), "game-count");
    else if (arg == "--games-per-shard") options.games_per_shard = parse_uint32(need(index, arg), "games-per-shard");
    else if (arg == "--max-concurrent-games") options.max_concurrent_games = parse_uint32(need(index, arg), "max-concurrent-games");
    else if (arg == "--inference-max-batch-size") options.inference_max_batch_size = parse_uint32(need(index, arg), "inference-max-batch-size");
    else if (arg == "--sampling-seed") options.sampling_seed = parse_uint32(need(index, arg), "sampling-seed");
    else if (arg == "--temperature") options.temperature = std::stod(need(index, arg));
    else if (arg == "--inference-device") options.inference_device = need(index, arg);
    else if (arg == "--policy-backend") options.policy_backend = need(index, arg);
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (options.output_directory.empty()) {
    throw std::runtime_error("--output is required");
  }
  if (options.game_count == 0) {
    throw std::runtime_error("--game-count must be positive");
  }
  if (options.games_per_shard == 0) {
    throw std::runtime_error("--games-per-shard must be positive");
  }
  if (options.max_concurrent_games == 0) {
    throw std::runtime_error("--max-concurrent-games must be positive");
  }
  if (options.inference_max_batch_size == 0) {
    throw std::runtime_error("--inference-max-batch-size must be positive");
  }
  if (options.policy_backend != "onnx" && options.policy_backend != "deterministic") {
    throw std::runtime_error("--policy-backend must be onnx or deterministic");
  }
  if (options.start_seed > std::numeric_limits<std::uint32_t>::max() - options.game_count + 1u) {
    throw std::runtime_error("start-seed + game-count exceeds uint32 range");
  }
  if (options.policy_backend == "onnx" &&
      (options.bidding_onnx_path.empty() || options.bidding_metadata_path.empty() ||
       options.playing_onnx_path.empty() || options.playing_metadata_path.empty())) {
    throw std::runtime_error(
        "--bidding-onnx, --bidding-metadata, --playing-onnx, and --playing-metadata "
        "are required for --policy-backend onnx");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);
    if (std::filesystem::exists(options.output_directory)) {
      throw std::runtime_error("output directory already exists");
    }
    std::filesystem::create_directories(options.output_directory);

    napoleon::onnx_policy::BatchedPolicyExecutor executor({
        options.inference_max_batch_size,
        options.temperature,
        options.sampling_seed});
    if (options.policy_backend == "onnx") {
      executor.add_policy(
          napoleon::onnx_policy::PolicyKey{AgentType::CurrentPolicy, "candidate-bidding"},
          napoleon::onnx_policy::create_onnxruntime_policy_session({
              {AgentType::CurrentPolicy, "candidate-bidding"},
              options.bidding_onnx_path.string(),
              "model_input",
              "logits",
              parse_device(options.inference_device),
              kBiddingFeatureCount,
              kBiddingActionCount}));
      executor.add_policy(
          napoleon::onnx_policy::PolicyKey{AgentType::FrozenPolicy, "ppo-separated-v1000"},
          napoleon::onnx_policy::create_onnxruntime_policy_session({
              {AgentType::FrozenPolicy, "ppo-separated-v1000"},
              options.playing_onnx_path.string(),
              "model_input",
              "logits",
              parse_device(options.inference_device),
              kPlayingFeatureCount,
              napoleon::observation::kCardCount}));
    } else {
      executor.add_policy(
          napoleon::onnx_policy::PolicyKey{AgentType::CurrentPolicy, "candidate-bidding"},
          std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>(
              napoleon::onnx_policy::DeterministicPolicySession::default_logits(),
              napoleon::onnx_policy::ExecutionProvider::Cpu,
              kBiddingFeatureCount,
              kBiddingActionCount));
      executor.add_policy(
          napoleon::onnx_policy::PolicyKey{AgentType::FrozenPolicy, "ppo-separated-v1000"},
          std::make_unique<napoleon::onnx_policy::DeterministicPolicySession>());
    }

    napoleon::SimulationRuntime runtime({
        napoleon::fixed_roster({napoleon::rule_based_agent(), napoleon::rule_based_agent(),
                                napoleon::rule_based_agent(), napoleon::rule_based_agent(),
                                napoleon::rule_based_agent()}),
        options.start_seed,
        0,
        options.max_concurrent_games,
        [](const napoleon::GameState& state, int player_index, napoleon::AgentRequest& request) {
          napoleon::onnx_policy::attach_bidding_model_input(state, player_index, request);
          napoleon::onnx_policy::attach_playing_model_input(state, player_index, request);
        },
        [](const napoleon::GameState& state, int player_index, const RosterAssignment& roster) {
          if (state.phase == Phase::Bidding) {
            return roster.agents[static_cast<std::size_t>(player_index)];
          }
          if (state.phase == Phase::Playing && !state.is_trick_complete) {
            return napoleon::frozen_policy_agent("ppo-separated-v1000");
          }
          return napoleon::rule_based_agent("rule-based");
        }});

    std::vector<napoleon::ScheduledGame> schedule;
    schedule.reserve(options.game_count * kPlayerCount);
    for (std::uint32_t offset = 0; offset < options.game_count; ++offset) {
      const std::uint32_t seed = options.start_seed + offset;
      for (int candidate_seat = 0; candidate_seat < kPlayerCount; ++candidate_seat) {
        schedule.push_back({seed, nonplaying_roster(seed, candidate_seat)});
      }
    }

    std::unordered_map<std::uint32_t, std::vector<Draft>> drafts_by_game;
    std::map<std::uint32_t, std::vector<std::string>> samples_by_seed;

    const auto started = std::chrono::steady_clock::now();
    std::size_t next_schedule = 0;
    std::uint64_t finished_games = 0;
    const std::uint64_t total_games = schedule.size();
    while (finished_games < total_games) {
      const std::size_t active_games = runtime.active_game_count();
      if (next_schedule < schedule.size() && active_games < options.max_concurrent_games) {
        const std::size_t open_slots = options.max_concurrent_games - active_games;
        const std::size_t batch_count = std::min(open_slots, schedule.size() - next_schedule);
        std::vector<napoleon::ScheduledGame> batch(
            schedule.begin() + static_cast<std::ptrdiff_t>(next_schedule),
            schedule.begin() + static_cast<std::ptrdiff_t>(next_schedule + batch_count));
        runtime.add_scheduled_games(batch);
        next_schedule += batch_count;
      }
      runtime.advance_runnable_games();
      const std::vector<napoleon::AgentRequest> requests = runtime.collect_agent_requests();
      std::unordered_map<std::uint64_t, napoleon::AgentRequest> request_by_id;
      for (const auto& request : requests) {
        request_by_id.emplace(request.request_id, request);
      }
      std::vector<napoleon::onnx_policy::PolicyActionResult> policy_results = executor.run(requests);
      std::vector<napoleon::AgentResult> results;
      results.reserve(policy_results.size());
      for (const auto& policy_result : policy_results) {
        const napoleon::AgentRequest& request = request_by_id.at(policy_result.result.request_id);
        if (request.phase == Phase::Bidding && request.agent.type == AgentType::CurrentPolicy) {
          Draft draft;
          draft.seed = request.seed;
          draft.step = request.game_decision_count;
          draft.acting_player_index = request.player_index;
          draft.candidate_seat_index = request.player_index;
          draft.model_input = request.bidding_model_input;
          draft.legal_bid_mask = request.legal_bid_mask;
          draft.selected_action_index = policy_result.result.selected_action_index;
          draft.behavior_log_probability = policy_result.result.behavior_log_probability;
          drafts_by_game[request.game_id].push_back(std::move(draft));
        }
        results.push_back(policy_result.result);
      }
      runtime.submit_agent_results(results);
      for (const FinishedGame& finished : runtime.collect_finished_games()) {
        auto& seed_samples = samples_by_seed[finished.seed];
        for (const Draft& draft : drafts_by_game[finished.game_id]) {
          std::ostringstream sample;
          write_sample(sample, draft, finished.result);
          seed_samples.push_back(sample.str());
        }
        drafts_by_game.erase(finished.game_id);
        ++finished_games;
      }
    }

    const auto serialization_started = std::chrono::steady_clock::now();
    std::vector<Shard> shards;
    std::uint64_t sample_count = 0;
    for (std::uint32_t offset = 0; offset < options.game_count;) {
      const std::uint32_t logical_game_count =
          std::min(options.games_per_shard, options.game_count - offset);
      const std::uint32_t shard_start_seed = options.start_seed + offset;
      const std::uint32_t shard_end_seed = shard_start_seed + logical_game_count - 1;
      const std::string file = shard_name(shards.size());
      std::ostringstream shard_content;

      std::uint64_t shard_sample_count = 0;
      for (std::uint32_t seed_offset = 0; seed_offset < logical_game_count; ++seed_offset) {
        const std::uint32_t seed = shard_start_seed + seed_offset;
        const auto found = samples_by_seed.find(seed);
        if (found == samples_by_seed.end()) {
          continue;
        }
        for (const std::string& sample : found->second) {
          shard_content << sample;
          ++shard_sample_count;
        }
      }
      const std::string shard_data = shard_content.str();
      std::ofstream shard(options.output_directory / file, std::ios::binary);
      if (!shard) {
        throw std::runtime_error("failed to open shard");
      }
      shard << shard_data;
      shard.close();
      shards.push_back({
          file,
          shard_start_seed,
          shard_end_seed,
          logical_game_count,
          shard_sample_count,
          static_cast<std::uint64_t>(shard_data.size()),
          sha256_bytes(shard_data)});
      sample_count += shard_sample_count;
      offset += logical_game_count;
    }
    const auto ended = std::chrono::steady_clock::now();
    const double serialization_elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            ended - serialization_started)
            .count();
    const auto runtime_metrics = runtime.metrics();
    const auto policy_stats = executor.stats();
    const double total_elapsed_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(ended - started)
            .count();
    const double total_elapsed_seconds = total_elapsed_ms / 1000.0;

    std::ofstream manifest(options.output_directory / "manifest.json");
    const std::string resolved_inference_device = options.inference_device == "cuda" ? "cuda" : "cpu";
    manifest << "{\n";
    manifest << "  \"datasetSchemaVersion\":4,\n";
    manifest << "  \"generatorVersion\":5,\n";
    manifest << "  \"format\":\"jsonl\",\n";
    manifest << "  \"sampleType\":\"non-playing-bidding-rl-sample\",\n";
    manifest << "  \"sampleSchemaVersion\":4,\n";
    manifest << "  \"phaseScope\":\"bidding-only\",\n";
    manifest << "  \"learnedPhases\":[\"bidding\"],\n";
    manifest << "  \"ruleBasedPhases\":[\"choosing-adjutant\",\"exchanging\"],\n";
    manifest << "  \"fixedPhases\":[\"playing\"],\n";
    manifest << "  \"rolloutPolicyTopology\":\"candidate-x1-frozen-x4-v1\",\n";
    manifest << "  \"gameCountUnit\":\"logical-seeds\",\n";
    manifest << "  \"logicalSeedCount\":" << options.game_count << ",\n";
    manifest << "  \"actualGameCount\":" << total_games << ",\n";
    manifest << "  \"rotationOffsets\":[0,1,2,3,4],\n";
    manifest << "  \"startSeed\":" << options.start_seed << ",\n";
    manifest << "  \"endSeed\":" << (options.start_seed + options.game_count - 1) << ",\n";
    manifest << "  \"gameCount\":" << options.game_count << ",\n";
    manifest << "  \"sampleCount\":" << sample_count << ",\n";
    manifest << "  \"gamesPerShard\":" << options.games_per_shard << ",\n";
    manifest << "  \"shardCount\":" << shards.size() << ",\n";
    manifest << "  \"playerCount\":5,\n";
    manifest << "  \"cardCount\":53,\n";
    manifest << "  \"cardIds\":";
    write_card_ids(manifest);
    manifest << ",\n";
    manifest << "  \"cardIdsSha256\":\"7ea0fdb58078f835bc5f7e6307a2a0c869430db343dd9e50ed1226f0452aaf38\",\n";
    manifest << "  \"biddingEncoderSchemaVersion\":1,\n";
    manifest << "  \"biddingModelInputSchemaVersion\":1,\n";
    manifest << "  \"biddingModelInputFeatureCount\":" << kBiddingFeatureCount << ",\n";
    manifest << "  \"playingModelInputSchemaVersion\":2,\n";
    manifest << "  \"playingModelInputFeatureCount\":" << kPlayingFeatureCount << ",\n";
    manifest << "  \"actionCount\":" << kBiddingActionCount << ",\n";
    manifest << "  \"behaviorPolicy\":";
    write_policy_artifact(
        manifest,
        "bidding-onnx",
        options.bidding_artifact_id,
        options.bidding_onnx_path,
        options.bidding_metadata_path,
        options.inference_device,
        resolved_inference_device,
        resolved_inference_device);
    manifest << ",\n";
    manifest << "  \"fixedPlayingPolicy\":";
    write_policy_artifact(
        manifest,
        "playing-onnx",
        options.playing_artifact_id,
        options.playing_onnx_path,
        options.playing_metadata_path,
        options.inference_device,
        resolved_inference_device,
        resolved_inference_device);
    manifest << ",\n";
    manifest << "  \"samplingAlgorithm\":\"masked-categorical\",\n";
    manifest << "  \"temperature\":" << options.temperature << ",\n";
    manifest << "  \"reward\":{\"type\":\"non-playing-terminal-role-reward\",\"version\":3,\"id\":\"non-playing-terminal-role-reward-v3\"},\n";
    manifest << "  \"allPassRule\":{\"id\":\"all-pass-immediate-zero-raw-terminal-reward-v1\",\"starterPayoff\":0,\"otherPayoff\":0},\n";
    manifest << "  \"terminalRewardTransform\":{\"type\":\"raw-reward-minus-game-player-mean\",\"version\":1,\"id\":\"non-playing-terminal-role-reward-v3-minus-game-player-mean-v1\",\"sourceRewardId\":\"non-playing-terminal-role-reward-v3\",\"baseline\":\"meanRawRewardAllPlayers\",\"formula\":\"relative_reward_i = raw_reward_i - mean(raw_reward_all_players)\"},\n";
    manifest << "  \"nonLearningAgents\":{\"bidding\":";
    write_frozen_bidding_mix_metadata(manifest);
    manifest << ",\"choosingAdjutant\":{\"type\":\"rule-based\",\"version\":1}"
             << ",\"exchanging\":{\"type\":\"rule-based\",\"version\":1}"
             << ",\"playing\":";
    write_policy_artifact(
        manifest,
        "playing-onnx",
        options.playing_artifact_id,
        options.playing_onnx_path,
        options.playing_metadata_path,
        options.inference_device,
        resolved_inference_device,
        resolved_inference_device);
    manifest << "},\n";
    manifest << "  \"diagnostics\":{\"simulationBackend\":\"cpp\",\"actualGameCount\":" << total_games
             << ",\"logicalSeedCount\":" << options.game_count
             << ",\"candidateSeatCount\":1,\"frozenSeatCount\":4,\"candidateRotationSeatCount\":5"
             << ",\"fallbackCount\":0,\"illegalActionCount\":0,\"rotationOffsets\":[0,1,2,3,4]"
             << ",\"frozenBiddingOpponentMix\":";
    write_frozen_bidding_mix_diagnostics(manifest, options.start_seed, options.game_count);
    manifest << "},\n";
    manifest << "  \"benchmark\":{\"totalRolloutElapsedMs\":"
             << total_elapsed_ms
             << ",\"actualGamesPerSecond\":"
             << (total_elapsed_seconds > 0.0 ? total_games / total_elapsed_seconds : 0.0)
             << ",\"biddingDecisionCount\":" << sample_count
             << ",\"biddingDecisionsPerSecond\":"
             << (total_elapsed_seconds > 0.0 ? sample_count / total_elapsed_seconds : 0.0)
             << ",\"simulationCpuElapsedMs\":" << (runtime_metrics.cpu_elapsed_ns / 1000000.0)
             << ",\"inferenceElapsedMs\":" << (policy_stats.inference_elapsed_ns / 1000000.0)
             << ",\"inferenceBatchCount\":" << policy_stats.session_run_count
             << ",\"inferenceRequestCount\":" << policy_stats.request_count
             << ",\"meanBatchSize\":" << policy_stats.mean_batch_size
             << ",\"maxBatchSize\":" << policy_stats.max_observed_batch_size
             << ",\"serializationElapsedMs\":" << serialization_elapsed_ms << "},\n";
    manifest << "  \"shards\":[";
    for (std::size_t index = 0; index < shards.size(); ++index) {
      if (index != 0) manifest << ',';
      manifest << "{\"file\":" << json_string(shards[index].file)
               << ",\"startSeed\":" << shards[index].start_seed
               << ",\"endSeed\":" << shards[index].end_seed
               << ",\"gameCount\":" << shards[index].game_count
               << ",\"sampleCount\":" << shards[index].sample_count
               << ",\"byteLength\":" << shards[index].byte_length
               << ",\"sha256\":" << json_string(shards[index].sha256) << "}";
    }
    manifest << "]\n";
    manifest << "}\n";

    std::cout << "{\"outputDirectory\":" << json_string(options.output_directory.string())
              << ",\"sampleCount\":" << sample_count
              << ",\"actualGameCount\":" << total_games
              << ",\"totalRolloutElapsedMs\":" << total_elapsed_ms
              << ",\"actualGamesPerSecond\":"
              << (total_elapsed_seconds > 0.0 ? total_games / total_elapsed_seconds : 0.0)
              << ",\"biddingDecisionCount\":" << sample_count
              << ",\"biddingDecisionsPerSecond\":"
              << (total_elapsed_seconds > 0.0 ? sample_count / total_elapsed_seconds : 0.0)
              << ",\"simulationCpuElapsedMs\":" << (runtime_metrics.cpu_elapsed_ns / 1000000.0)
              << ",\"inferenceElapsedMs\":" << (policy_stats.inference_elapsed_ns / 1000000.0)
              << ",\"inferenceBatchCount\":" << policy_stats.session_run_count
              << ",\"inferenceRequestCount\":" << policy_stats.request_count
              << ",\"meanBatchSize\":" << policy_stats.mean_batch_size
              << ",\"maxBatchSize\":" << policy_stats.max_observed_batch_size
              << ",\"serializationElapsedMs\":" << serialization_elapsed_ms << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
