#include "napoleon_joint_teacher.hpp"
#include "napoleon_parameterized_policy.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t parse_uint32(const std::string& name, const std::string& value) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(name + " must be a uint32");
  }
  return static_cast<std::uint32_t>(parsed);
}

int parse_positive_int(const std::string& name, const std::string& value) {
  const std::uint32_t parsed = parse_uint32(name, value);
  if (parsed == 0 || parsed > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(name + " must be a positive int");
  }
  return static_cast<int>(parsed);
}

std::vector<std::uint32_t> read_seeds(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("failed to open seed file: " + path);
  std::vector<std::uint32_t> seeds;
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty()) seeds.push_back(parse_uint32("seed", line));
  }
  return seeds;
}

void usage() {
  throw std::runtime_error(
      "usage: napoleon_parameterized_eval_cli --schema | --initial-parameters | "
      "--discover-start <seed> --discover-count <N> [--max-attempts <N>] | "
      "--seeds <path> --server [policy path options]");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    napoleon::joint_teacher::ParameterizedPolicyEvaluationOptions options;
    bool schema = false;
    bool initial = false;
    bool server = false;
    bool discover = false;
    std::uint32_t discover_start = 0;
    int discover_count = 0;
    int max_attempts = 0;
    std::string seed_path;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--schema") schema = true;
      else if (arg == "--initial-parameters") initial = true;
      else if (arg == "--server") server = true;
      else if (arg == "--discover-start" && index + 1 < argc) {
        discover = true;
        discover_start = parse_uint32(arg, argv[++index]);
      } else if (arg == "--discover-count" && index + 1 < argc) {
        discover_count = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--max-attempts" && index + 1 < argc) {
        max_attempts = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--seeds" && index + 1 < argc) {
        seed_path = argv[++index];
      } else if (arg == "--bidding-margin-onnx" && index + 1 < argc) {
        options.bidding_margin_onnx_path = argv[++index];
      } else if (arg == "--playing-policy-onnx" && index + 1 < argc) {
        options.playing_policy_onnx_path = argv[++index];
      } else if (arg == "--playing-critic-onnx" && index + 1 < argc) {
        options.playing_critic_onnx_path = argv[++index];
      } else if (arg == "--device" && index + 1 < argc) {
        options.policy_device = argv[++index];
      } else if (arg == "--agent-seed" && index + 1 < argc) {
        options.agent_seed = parse_uint32(arg, argv[++index]);
      } else {
        usage();
      }
    }

    const int modes = static_cast<int>(schema) + static_cast<int>(initial) +
        static_cast<int>(discover) + static_cast<int>(server);
    if (modes != 1) usage();
    if (schema) {
      std::cout << napoleon::parameterized_policy::feature_schema_json() << '\n';
      return 0;
    }
    if (initial) {
      std::cout << napoleon::parameterized_policy::parameters_json(
                       napoleon::parameterized_policy::initial_rule_based_parameters())
                << '\n';
      return 0;
    }
    if (discover) {
      if (discover_count <= 0) usage();
      if (max_attempts == 0) max_attempts = discover_count * 20;
      const auto seeds = napoleon::joint_teacher::discover_parameterized_policy_seeds(
          options, discover_start, discover_count, max_attempts);
      for (std::uint32_t seed : seeds) std::cout << seed << '\n';
      return 0;
    }
    if (seed_path.empty()) usage();
    napoleon::joint_teacher::run_parameterized_policy_evaluation_server(
        options, read_seeds(seed_path), std::cin, std::cout);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
