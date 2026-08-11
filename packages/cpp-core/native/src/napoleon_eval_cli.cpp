#include "napoleon_evaluation.hpp"

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

std::size_t parse_size(const std::string& name, const std::string& value) {
  const std::uint32_t parsed = parse_uint32(name, value);
  if (parsed == 0) {
    throw std::runtime_error(name + " must be positive");
  }
  return parsed;
}

std::string parse_inference_device(const std::string& value) {
  if (value == "cpu" || value == "auto" || value == "cuda") {
    return value;
  }
  throw std::runtime_error("--inference-device must be one of cpu, auto, cuda");
}

std::string parse_policy_backend(const std::string& value) {
  if (value == "deterministic" || value == "onnx") {
    return value;
  }
  throw std::runtime_error("--policy-backend must be one of deterministic, onnx");
}

napoleon::evaluation::EvaluationScenario parse_scenario(const std::string& value) {
  using napoleon::evaluation::EvaluationScenario;
  if (value == "candidate-vs-rule-based" || value == "rule-based-x4") {
    return EvaluationScenario::CandidateVsRuleBased;
  }
  if (value == "candidate-vs-frozen-policy" || value == "frozen-x4" ||
      value == "rl-v740-x4") {
    return EvaluationScenario::CandidateVsFrozen;
  }
  if (value == "candidate-vs-opponent-pool" || value == "opponent-pool") {
    return EvaluationScenario::CandidateVsOpponentPool;
  }
  if (value == "tournament") {
    return EvaluationScenario::Tournament;
  }
  throw std::runtime_error(
      "--scenario must be one of candidate-vs-rule-based, "
      "candidate-vs-frozen-policy, candidate-vs-opponent-pool, tournament");
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
    napoleon::evaluation::EvaluationOptions options;
    std::string output = "-";

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if ((arg == "--scenario" || arg == "--benchmark") && index + 1 < argc) {
        options.scenario = parse_scenario(argv[++index]);
      } else if (arg == "--start-seed" && index + 1 < argc) {
        options.start_seed = parse_uint32(arg, argv[++index]);
      } else if ((arg == "--seed-count" || arg == "--game-count") && index + 1 < argc) {
        options.seed_count = parse_uint32(arg, argv[++index]);
      } else if (arg == "--roster-seed" && index + 1 < argc) {
        options.roster_seed = parse_uint32(arg, argv[++index]);
      } else if (arg == "--max-concurrent-games" && index + 1 < argc) {
        options.max_concurrent_games = parse_size(arg, argv[++index]);
      } else if (arg == "--inference-max-batch-size" && index + 1 < argc) {
        options.inference_max_batch_size = parse_size(arg, argv[++index]);
      } else if (arg == "--candidate-id" && index + 1 < argc) {
        options.candidate_id = argv[++index];
      } else if (arg == "--frozen-policy-id" && index + 1 < argc) {
        options.frozen_id = argv[++index];
      } else if (arg == "--candidate-onnx" && index + 1 < argc) {
        options.candidate_onnx_path = argv[++index];
      } else if (arg == "--candidate-metadata" && index + 1 < argc) {
        options.candidate_metadata_path = argv[++index];
      } else if (arg == "--frozen-onnx" && index + 1 < argc) {
        options.frozen_onnx_path = argv[++index];
      } else if (arg == "--frozen-metadata" && index + 1 < argc) {
        options.frozen_metadata_path = argv[++index];
      } else if (arg == "--inference-device" && index + 1 < argc) {
        options.inference_device = parse_inference_device(argv[++index]);
      } else if (arg == "--policy-backend" && index + 1 < argc) {
        options.policy_backend = parse_policy_backend(argv[++index]);
      } else if (arg == "--output" && index + 1 < argc) {
        output = argv[++index];
      } else {
        throw std::runtime_error(
            "usage: napoleon_eval_cli --scenario <name> --start-seed <uint32> "
            "--seed-count <N> [--roster-seed <uint32>] "
            "[--max-concurrent-games <N>] [--inference-max-batch-size <N>] "
            "[--candidate-id <id>] [--frozen-policy-id <id>] "
            "[--candidate-onnx <path>] [--candidate-metadata <path>] "
            "[--frozen-onnx <path>] [--frozen-metadata <path>] "
            "[--inference-device cpu|auto|cuda] [--policy-backend deterministic|onnx] "
            "[--output <path>|-]");
      }
    }
    if (options.inference_device == "cuda" && options.policy_backend != "onnx") {
      throw std::runtime_error("--inference-device cuda requires --policy-backend onnx");
    }

    const napoleon::evaluation::EvaluationArtifact artifact =
        napoleon::evaluation::run_evaluation(options);
    write_output(output, artifact.json);
    if (output != "-") {
      std::cerr << "completed " << artifact.completed_games << "/"
                << artifact.scheduled_games << '\n';
    }
    if (output != "-") {
      const std::string resolved = options.inference_device == "cuda" ? "cuda" : "cpu";
      std::cout << "{\"scheduledGames\":" << artifact.scheduled_games
                << ",\"completedGames\":" << artifact.completed_games
                << ",\"failedGames\":" << artifact.failed_games
                << ",\"illegalActionCount\":0"
                << ",\"requestedInferenceDevice\":\"" << options.inference_device << "\""
                << ",\"resolvedInferenceDevice\":\"" << resolved << "\""
                << ",\"executionProvider\":\"" << resolved << "\"}\n";
    }
    return artifact.failed_games == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
