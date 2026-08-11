#include "napoleon_onnx_policy.hpp"
#include "napoleon_rule_based.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SmokeOptions {
  std::string current_onnx;
  std::string frozen_onnx;
  napoleon::onnx_policy::InferenceDevice device = napoleon::onnx_policy::InferenceDevice::Cpu;
  std::size_t games = 2;
  std::size_t max_batch_size = 4;
  std::uint32_t base_seed = 9000;
  std::uint32_t roster_seed = 9100;
  std::uint32_t sampling_seed = 9200;
};

std::size_t parse_size(const std::string& value, const std::string& label) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0) {
    throw std::runtime_error(label + " must be positive");
  }
  return static_cast<std::size_t>(parsed);
}

std::uint32_t parse_u32(const std::string& value, const std::string& label) {
  const unsigned long parsed = std::stoul(value);
  if (parsed > 0xffffffffUL) {
    throw std::runtime_error(label + " must fit in uint32");
  }
  return static_cast<std::uint32_t>(parsed);
}

SmokeOptions parse_args(int argc, char** argv) {
  SmokeOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto require_value = [&](const std::string& label) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(label + " requires a value");
      }
      return argv[++index];
    };

    if (arg == "--current-onnx") {
      options.current_onnx = require_value(arg);
    } else if (arg == "--frozen-onnx") {
      options.frozen_onnx = require_value(arg);
    } else if (arg == "--provider") {
      const std::string provider = require_value(arg);
      if (provider == "cpu") {
        options.device = napoleon::onnx_policy::InferenceDevice::Cpu;
      } else if (provider == "cuda") {
        options.device = napoleon::onnx_policy::InferenceDevice::Cuda;
      } else {
        throw std::runtime_error("--provider must be cpu or cuda");
      }
    } else if (arg == "--games") {
      options.games = parse_size(require_value(arg), arg);
    } else if (arg == "--max-batch-size") {
      options.max_batch_size = parse_size(require_value(arg), arg);
    } else if (arg == "--base-seed") {
      options.base_seed = parse_u32(require_value(arg), arg);
    } else if (arg == "--roster-seed") {
      options.roster_seed = parse_u32(require_value(arg), arg);
    } else if (arg == "--sampling-seed") {
      options.sampling_seed = parse_u32(require_value(arg), arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (options.current_onnx.empty()) {
    throw std::runtime_error("--current-onnx is required");
  }
  return options;
}

napoleon::AgentResult first_legal_result(const napoleon::AgentRequest& request) {
  if (request.legal_actions.empty()) {
    throw std::runtime_error("request has no legal actions");
  }
  napoleon::AgentResult result;
  result.request_id = request.request_id;
  result.action = request.legal_actions.front();
  return result;
}

void write_histogram(std::ostream& out, const std::map<std::size_t, std::uint64_t>& histogram) {
  out << '{';
  bool first = true;
  for (const auto& entry : histogram) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << '"' << entry.first << "\":" << entry.second;
  }
  out << '}';
}

void write_policy_stats(std::ostream& out, const napoleon::onnx_policy::PolicyBatchStats& stats) {
  out << "{\"requestCount\":" << stats.request_count;
  out << ",\"sessionRunCount\":" << stats.session_run_count;
  out << ",\"meanBatchSize\":" << stats.mean_batch_size;
  out << ",\"maxObservedBatchSize\":" << stats.max_observed_batch_size;
  out << ",\"batchHistogram\":";
  write_histogram(out, stats.batch_size_histogram);
  out << ",\"inferenceElapsedNs\":" << stats.inference_elapsed_ns << '}';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const SmokeOptions options = parse_args(argc, argv);
    const napoleon::AgentIdentity current = napoleon::current_policy_agent("current");
    const napoleon::AgentIdentity frozen = napoleon::frozen_policy_agent("frozen");
    const napoleon::AgentIdentity rule = napoleon::rule_based_agent("rule");

    napoleon::onnx_policy::BatchedPolicyExecutor executor(
        napoleon::onnx_policy::BatchedPolicyConfig{
            options.max_batch_size,
            1.0,
            options.sampling_seed});
    executor.add_policy(
        napoleon::onnx_policy::policy_key_from_agent(current),
        napoleon::onnx_policy::create_onnxruntime_policy_session(
            napoleon::onnx_policy::PolicySessionConfig{
                napoleon::onnx_policy::policy_key_from_agent(current),
                options.current_onnx,
                "model_input",
                "logits",
                options.device}));

    napoleon::RosterSpec roster = napoleon::self_play_roster(current);
    if (!options.frozen_onnx.empty()) {
      executor.add_policy(
          napoleon::onnx_policy::policy_key_from_agent(frozen),
          napoleon::onnx_policy::create_onnxruntime_policy_session(
              napoleon::onnx_policy::PolicySessionConfig{
                  napoleon::onnx_policy::policy_key_from_agent(frozen),
                  options.frozen_onnx,
                  "model_input",
                  "logits",
                  options.device}));
      roster = napoleon::fixed_roster({current, rule, frozen, rule, current});
    }

    napoleon::SimulationRuntime runtime(napoleon::SimulationRuntimeConfig{
        roster,
        options.base_seed,
        options.roster_seed,
        std::max<std::size_t>(1, options.games),
        napoleon::onnx_policy::attach_playing_model_input});
    runtime.add_games(options.games);

    std::size_t finished_count = 0;
    for (int iteration = 0; iteration < 100000 && finished_count < options.games; ++iteration) {
      runtime.advance_runnable_games();
      std::vector<napoleon::AgentRequest> requests = runtime.collect_agent_requests();
      if (!requests.empty()) {
        std::vector<napoleon::AgentRequest> playing_requests;
        std::vector<napoleon::AgentResult> setup_results;
        for (const napoleon::AgentRequest& request : requests) {
          if (request.phase == napoleon::Phase::Playing) {
            playing_requests.push_back(request);
          } else {
            setup_results.push_back(first_legal_result(request));
          }
        }
        if (!setup_results.empty()) {
          runtime.submit_agent_results(setup_results);
        }
        if (!playing_requests.empty()) {
          const std::vector<napoleon::onnx_policy::PolicyActionResult> policy_results =
              executor.run(playing_requests);
          std::vector<napoleon::AgentResult> results;
          results.reserve(policy_results.size());
          for (const auto& policy_result : policy_results) {
            results.push_back(policy_result.result);
          }
          runtime.submit_agent_results(results);
        }
      }
      finished_count += runtime.collect_finished_games().size();
    }

    if (finished_count != options.games) {
      throw std::runtime_error("smoke runtime did not finish all games");
    }

    const napoleon::RuntimeMetrics runtime_metrics = runtime.metrics();
    const napoleon::onnx_policy::BatchedPolicyStats stats = executor.stats();
    std::cout << "{\"provider\":\""
              << napoleon::onnx_policy::inference_device_id(options.device) << '"';
    std::cout << ",\"games\":" << finished_count;
    std::cout << ",\"runtime\":{\"agentRequestCount\":" << runtime_metrics.agent_request_count;
    std::cout << ",\"submittedAgentResultCount\":"
              << runtime_metrics.submitted_agent_result_count << '}';
    std::cout << ",\"inference\":";
    write_policy_stats(std::cout, napoleon::onnx_policy::PolicyBatchStats{
                                      stats.request_count,
                                      stats.session_run_count,
                                      stats.mean_batch_size,
                                      stats.max_observed_batch_size,
                                      stats.batch_size_histogram,
                                      stats.inference_elapsed_ns});
    std::cout << ",\"policyStats\":{";
    bool first = true;
    for (const auto& entry : stats.policy_stats) {
      if (!first) {
        std::cout << ',';
      }
      first = false;
      std::cout << '"' << entry.first << "\":";
      write_policy_stats(std::cout, entry.second);
    }
    std::cout << "}}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "napoleon_onnx_policy_smoke failed: " << error.what() << "\n";
    return 1;
  }
}
