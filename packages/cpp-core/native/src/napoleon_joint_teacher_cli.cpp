#include "napoleon_joint_teacher.hpp"

#include <cstdint>
#include <fstream>
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

int parse_positive_int(const std::string& name, const std::string& value) {
  const std::uint32_t parsed = parse_uint32(name, value);
  if (parsed == 0 || parsed > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(name + " must be a positive int");
  }
  return static_cast<int>(parsed);
}

void write_output(const std::string& path, const std::string& json) {
  if (path.empty() || path == "-") {
    std::cout << json << '\n';
    return;
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open output path: " + path);
  }
  out << json << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    napoleon::joint_teacher::JointTeacherOptions options;
    std::string output = "-";

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--start-seed" && index + 1 < argc) {
        options.start_seed = parse_uint32(arg, argv[++index]);
      } else if (arg == "--states" && index + 1 < argc) {
        options.requested_source_states = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--max-deal-attempts" && index + 1 < argc) {
        options.max_deal_attempts = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--exhaustive-states" && index + 1 < argc) {
        options.exhaustive_state_count = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--heuristic-top-k" && index + 1 < argc) {
        options.heuristic_top_k = parse_positive_int(arg, argv[++index]);
      } else if (arg == "--agent-seed" && index + 1 < argc) {
        options.agent_seed = parse_uint32(arg, argv[++index]);
      } else if (arg == "--output" && index + 1 < argc) {
        output = argv[++index];
      } else {
        throw std::runtime_error(
            "usage: napoleon_joint_teacher_cli --states <N> --start-seed <uint32> "
            "[--max-deal-attempts <N>] [--exhaustive-states <N>] "
            "[--heuristic-top-k <N>] [--agent-seed <uint32>] [--output <path>|-]");
      }
    }

    const napoleon::joint_teacher::JointTeacherReport report =
        napoleon::joint_teacher::run_joint_teacher_diagnostic(options);
    write_output(output, report.json);
    if (output != "-") {
      std::cerr << "completed sourceStates=" << report.source_state_count
                << " terminalRollouts=" << report.terminal_rollout_count << '\n';
    }
    return report.source_state_count == options.requested_source_states ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
