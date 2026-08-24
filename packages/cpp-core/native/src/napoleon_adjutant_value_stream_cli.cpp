#include "napoleon_joint_teacher.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::uint32_t parse_uint32(const std::string& name, const std::string& value) {
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value);
  } catch (const std::exception&) {
    throw std::runtime_error(name + " must be an unsigned integer");
  }
  if (parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(name + " must be <= 4294967295");
  }
  return static_cast<std::uint32_t>(parsed);
}

int parse_int(const std::string& name, const std::string& value) {
  const std::uint32_t parsed = parse_uint32(name, value);
  if (parsed > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(name + " must fit in int");
  }
  return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    napoleon::joint_teacher::AdjutantValueStreamOptions options;
    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--mode" && index + 1 < argc) {
        options.mode = argv[++index];
      } else if (arg == "--output-directory" && index + 1 < argc) {
        options.output_directory = argv[++index];
      } else if (arg == "--bidding-policy-id" && index + 1 < argc) {
        options.bidding_policy_id = argv[++index];
      } else if (arg == "--bidding-margin-onnx" && index + 1 < argc) {
        options.bidding_margin_onnx_path = argv[++index];
      } else if (arg == "--playing-policy-id" && index + 1 < argc) {
        options.playing_policy_id = argv[++index];
      } else if (arg == "--playing-policy-onnx" && index + 1 < argc) {
        options.playing_policy_onnx_path = argv[++index];
      } else if (arg == "--playing-critic-onnx" && index + 1 < argc) {
        options.playing_critic_onnx_path = argv[++index];
      } else if (arg == "--policy-device" && index + 1 < argc) {
        options.policy_device = argv[++index];
      } else if (arg == "--start-seed" && index + 1 < argc) {
        options.start_seed = parse_uint32(arg, argv[++index]);
      } else if (arg == "--states" && index + 1 < argc) {
        options.requested_source_states = parse_int(arg, argv[++index]);
      } else if (arg == "--max-deal-attempts" && index + 1 < argc) {
        options.max_deal_attempts = parse_int(arg, argv[++index]);
      } else if (arg == "--proposal-top-k" && index + 1 < argc) {
        options.proposal_top_k = parse_int(arg, argv[++index]);
      } else if (arg == "--diversity-count" && index + 1 < argc) {
        options.diversity_count = parse_int(arg, argv[++index]);
      } else if (arg == "--scorer-top-k" && index + 1 < argc) {
        options.scorer_top_k = parse_int(arg, argv[++index]);
      } else if (arg == "--agent-seed" && index + 1 < argc) {
        options.agent_seed = parse_uint32(arg, argv[++index]);
      } else {
        throw std::runtime_error(
            "usage: napoleon_adjutant_value_stream_cli --mode proposal|full-gold "
            "--output-directory <path> --states <N> --start-seed <uint32> "
            "[--bidding-policy-id frozen-raise-v1] [--bidding-margin-onnx <path>] "
            "[--playing-policy-id ppo-separated-v1000] [--playing-policy-onnx <path>] "
            "[--playing-critic-onnx <path>] [--policy-device cpu|cuda] "
            "[--max-deal-attempts <N>] [--proposal-top-k <N>] "
            "[--diversity-count <N>] [--scorer-top-k <N>] [--agent-seed <uint32>]");
      }
    }

    const napoleon::joint_teacher::AdjutantValueStreamReport report =
        napoleon::joint_teacher::run_adjutant_value_stream_teacher(
            options,
            std::cin,
            std::cout);
    std::cerr << "[adjutant-stream] completed sourceStates=" << report.source_state_count
              << " samples=" << report.sample_count
              << " terminalRollouts=" << report.terminal_rollout_count << '\n';
    return report.source_state_count == options.requested_source_states ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
