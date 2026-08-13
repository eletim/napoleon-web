#pragma once

#include "napoleon_observation.hpp"
#include "napoleon_simulation_runtime.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace napoleon::onnx_policy {

constexpr int kPolicyLogitCount = observation::kCardCount;

enum class ExecutionProvider : std::uint8_t {
  Cpu,
  Cuda
};

enum class InferenceDevice : std::uint8_t {
  Cpu,
  Cuda
};

struct PolicyKey {
  AgentType agent_type = AgentType::CurrentPolicy;
  std::string agent_id;
};

struct PolicySessionConfig {
  PolicyKey key;
  std::string onnx_path;
  std::string input_name = "model_input";
  std::string output_name = "logits";
  InferenceDevice inference_device = InferenceDevice::Cpu;
  std::size_t model_input_feature_count = observation::kPlayingModelInputFeatureCount;
};

struct BatchedPolicyConfig {
  std::size_t max_batch_size = 1;
  double temperature = 1.0;
  std::uint32_t sampling_seed = 0;
};

struct PolicyBatchStats {
  std::uint64_t request_count = 0;
  std::uint64_t session_run_count = 0;
  double mean_batch_size = 0.0;
  std::size_t max_observed_batch_size = 0;
  std::map<std::size_t, std::uint64_t> batch_size_histogram;
  std::uint64_t inference_elapsed_ns = 0;
};

struct BatchedPolicyStats {
  std::uint64_t request_count = 0;
  std::uint64_t session_run_count = 0;
  double mean_batch_size = 0.0;
  std::size_t max_observed_batch_size = 0;
  std::map<std::size_t, std::uint64_t> batch_size_histogram;
  std::uint64_t inference_elapsed_ns = 0;
  std::map<std::string, PolicyBatchStats> policy_stats;
};

struct PolicyActionResult {
  AgentResult result;
  std::uint64_t sequence = 0;
  int selected_card_index = -1;
  double behavior_log_probability = 0.0;
  std::string policy_key;
};

class PolicySession {
 public:
  virtual ~PolicySession() = default;
  virtual std::vector<std::array<float, kPolicyLogitCount>> run_logits_batch(
      const std::vector<std::vector<float>>& inputs) = 0;
  virtual std::size_t model_input_feature_count() const = 0;
  virtual ExecutionProvider execution_provider() const = 0;
};

class DeterministicPolicySession final : public PolicySession {
 public:
  explicit DeterministicPolicySession(
      std::array<float, kPolicyLogitCount> logits = default_logits(),
      ExecutionProvider provider = ExecutionProvider::Cpu,
      std::size_t model_input_feature_count = observation::kPlayingModelInputFeatureCount);

  std::vector<std::array<float, kPolicyLogitCount>> run_logits_batch(
      const std::vector<std::vector<float>>& inputs) override;
  std::size_t model_input_feature_count() const override;
  ExecutionProvider execution_provider() const override;
  std::uint64_t session_run_count() const;

  static std::array<float, kPolicyLogitCount> default_logits();

 private:
  std::array<float, kPolicyLogitCount> logits_;
  ExecutionProvider provider_ = ExecutionProvider::Cpu;
  std::size_t model_input_feature_count_ = observation::kPlayingModelInputFeatureCount;
  std::uint64_t session_run_count_ = 0;
};

std::unique_ptr<PolicySession> create_onnxruntime_policy_session(
    const PolicySessionConfig& config);

class BatchedPolicyExecutor {
 public:
  explicit BatchedPolicyExecutor(BatchedPolicyConfig config);
  ~BatchedPolicyExecutor();

  void add_policy(PolicyKey key, std::unique_ptr<PolicySession> session);
  void enqueue(const AgentRequest& request);
  std::vector<PolicyActionResult> flush();
  std::vector<PolicyActionResult> run(const std::vector<AgentRequest>& requests);

  BatchedPolicyStats stats() const;
  void reset_stats();

 private:
  struct QueueItem;
  struct PolicyState;

  BatchedPolicyConfig config_;
  std::map<std::string, std::unique_ptr<PolicyState>> policies_;
  mutable std::mutex mutex_;
};

std::string policy_key_id(const PolicyKey& key);
PolicyKey policy_key_from_agent(const AgentIdentity& agent);
void attach_playing_model_input(
    const GameState& state,
    int player_index,
    AgentRequest& request);
void attach_playing_model_input(
    const GameState& state,
    int player_index,
    observation::PlayingObservationVariant variant,
    AgentRequest& request);
std::string execution_provider_id(ExecutionProvider provider);
std::string inference_device_id(InferenceDevice device);

}  // namespace napoleon::onnx_policy
