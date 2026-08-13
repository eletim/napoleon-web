#include "napoleon_onnx_policy.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef NAPOLEON_ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace napoleon::onnx_policy {
namespace {

constexpr double kDefaultTemperature = 1.0;

std::uint32_t mix_u32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

std::uint32_t stable_hash(const std::string& value) {
  std::uint32_t hash = 2166136261u;
  for (unsigned char ch : value) {
    hash ^= ch;
    hash *= 16777619u;
  }
  return hash;
}

std::uint32_t sampling_seed_for_request(
    std::uint32_t base_seed,
    const AgentRequest& request,
    const std::string& key) {
  std::uint32_t value = base_seed;
  value ^= mix_u32(request.game_index + 0x9e3779b9u);
  value ^= mix_u32(static_cast<std::uint32_t>(request.sequence & 0xffffffffu));
  value ^= mix_u32(static_cast<std::uint32_t>(request.player_index) + 0x85ebca6bu);
  value ^= stable_hash(key);
  return mix_u32(value);
}

std::vector<float> request_model_input(
    const AgentRequest& request,
    std::size_t expected_feature_count) {
  if (request.playing_model_input.size() != expected_feature_count) {
    throw std::runtime_error(
        "playing_model_input feature count mismatch: expected " +
        std::to_string(expected_feature_count) + ", got " +
        std::to_string(request.playing_model_input.size()));
  }
  std::vector<float> input = request.playing_model_input;
  for (float value : input) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("playing_model_input values must be finite");
    }
  }
  return input;
}

std::array<int, observation::kCardCount> request_legal_play_mask(const AgentRequest& request) {
  if (request.legal_play_mask.size() != observation::kCardCount) {
    throw std::runtime_error("legal_play_mask must contain 53 entries");
  }
  std::array<int, observation::kCardCount> mask{};
  std::copy(request.legal_play_mask.begin(), request.legal_play_mask.end(), mask.begin());
  return mask;
}

bool is_legal_mask_value(int value) {
  return value == 1;
}

void validate_temperature(double temperature) {
  if (!std::isfinite(temperature) || temperature <= 0.0) {
    throw std::runtime_error("policy sampling temperature must be finite and greater than 0");
  }
}

struct MaskedDistribution {
  std::vector<int> legal_card_indices;
  std::vector<double> probabilities;
  std::vector<double> log_probabilities;
};

MaskedDistribution create_masked_distribution(
    const std::array<float, kPolicyLogitCount>& logits,
    const std::array<int, observation::kCardCount>& legal_play_mask,
    double temperature) {
  validate_temperature(temperature);

  MaskedDistribution distribution;
  std::vector<double> scaled_logits;
  for (int index = 0; index < kPolicyLogitCount; ++index) {
    const float logit = logits[static_cast<std::size_t>(index)];
    if (!std::isfinite(logit)) {
      throw std::runtime_error("policy logits must be finite");
    }
    const int mask_value = legal_play_mask[static_cast<std::size_t>(index)];
    if (mask_value != 0 && mask_value != 1) {
      throw std::runtime_error("legalPlayMask must contain only 0/1 values");
    }
    if (is_legal_mask_value(mask_value)) {
      distribution.legal_card_indices.push_back(index);
      scaled_logits.push_back(static_cast<double>(logit) / temperature);
    }
  }

  if (distribution.legal_card_indices.empty()) {
    throw std::runtime_error("legalPlayMask must contain at least one legal card");
  }

  if (distribution.legal_card_indices.size() == 1) {
    distribution.probabilities.push_back(1.0);
    distribution.log_probabilities.push_back(0.0);
    return distribution;
  }

  const double max_scaled_logit =
      *std::max_element(scaled_logits.begin(), scaled_logits.end());
  std::vector<double> exp_values;
  exp_values.reserve(scaled_logits.size());
  for (double scaled_logit : scaled_logits) {
    exp_values.push_back(std::exp(scaled_logit - max_scaled_logit));
  }
  const double exp_sum = std::accumulate(exp_values.begin(), exp_values.end(), 0.0);
  if (!std::isfinite(exp_sum) || exp_sum <= 0.0) {
    throw std::runtime_error("masked categorical softmax normalization failed");
  }

  const double log_denominator = max_scaled_logit + std::log(exp_sum);
  for (std::size_t index = 0; index < exp_values.size(); ++index) {
    const double probability = exp_values[index] / exp_sum;
    const double log_probability = scaled_logits[index] - log_denominator;
    if (!std::isfinite(probability) || probability < 0.0 ||
        !std::isfinite(log_probability) || log_probability > 1e-12) {
      throw std::runtime_error("masked categorical distribution contains an invalid probability");
    }
    distribution.probabilities.push_back(probability);
    distribution.log_probabilities.push_back(log_probability);
  }

  return distribution;
}

struct SampledAction {
  int selected_card_index = -1;
  double log_probability = 0.0;
};

SampledAction sample_legal_action(
    const std::array<float, kPolicyLogitCount>& logits,
    const std::array<int, observation::kCardCount>& legal_play_mask,
    double temperature,
    std::uint32_t seed) {
  const MaskedDistribution distribution =
      create_masked_distribution(logits, legal_play_mask, temperature);

  if (distribution.legal_card_indices.size() == 1) {
    return SampledAction{distribution.legal_card_indices.front(), 0.0};
  }

  SeededRandom rng(seed);
  const double random_value = rng.next();
  if (!std::isfinite(random_value) || random_value < 0.0 || random_value >= 1.0) {
    throw std::runtime_error("rng must return a finite value in [0, 1)");
  }

  double cumulative_probability = 0.0;
  for (std::size_t index = 0; index < distribution.legal_card_indices.size(); ++index) {
    cumulative_probability += distribution.probabilities[index];
    if (random_value < cumulative_probability) {
      return SampledAction{
          distribution.legal_card_indices[index],
          distribution.log_probabilities[index]};
    }
  }

  const std::size_t last_index = distribution.legal_card_indices.size() - 1;
  return SampledAction{
      distribution.legal_card_indices[last_index],
      distribution.log_probabilities[last_index]};
}

Action action_from_selected_card_index(const AgentRequest& request, int selected_card_index) {
  const Card selected_card = observation::card_from_playing_model_index(selected_card_index);
  auto action_it = std::find_if(
      request.legal_actions.begin(),
      request.legal_actions.end(),
      [&](const Action& action) {
        return action.type == Action::Type::PlayCard &&
               observation::playing_card_model_index(action.card) == selected_card_index &&
               action.card.id == selected_card.id;
      });
  if (action_it == request.legal_actions.end()) {
    throw std::runtime_error("ONNX policy selected a card outside request legal actions");
  }
  return *action_it;
}

void record_batch(PolicyBatchStats& stats, std::size_t batch_size, std::uint64_t elapsed_ns) {
  stats.request_count += batch_size;
  stats.session_run_count += 1;
  stats.mean_batch_size =
      static_cast<double>(stats.request_count) / static_cast<double>(stats.session_run_count);
  stats.max_observed_batch_size = std::max(stats.max_observed_batch_size, batch_size);
  stats.batch_size_histogram[batch_size] += 1;
  stats.inference_elapsed_ns += elapsed_ns;
}

void merge_policy_stats(BatchedPolicyStats& aggregate, const PolicyBatchStats& stats) {
  aggregate.request_count += stats.request_count;
  aggregate.session_run_count += stats.session_run_count;
  aggregate.max_observed_batch_size =
      std::max(aggregate.max_observed_batch_size, stats.max_observed_batch_size);
  aggregate.inference_elapsed_ns += stats.inference_elapsed_ns;
  for (const auto& entry : stats.batch_size_histogram) {
    aggregate.batch_size_histogram[entry.first] += entry.second;
  }
}

#ifdef NAPOLEON_ENABLE_ONNXRUNTIME

bool has_cuda_provider() {
  const std::vector<std::string> providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
}

class OnnxRuntimePolicySession final : public PolicySession {
 public:
  explicit OnnxRuntimePolicySession(const PolicySessionConfig& config)
      : env_(ORT_LOGGING_LEVEL_WARNING, "napoleon-onnx-policy"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        input_name_(config.input_name),
        output_name_(config.output_name),
        model_input_feature_count_(config.model_input_feature_count),
        provider_(config.inference_device == InferenceDevice::Cuda ? ExecutionProvider::Cuda
                                                                    : ExecutionProvider::Cpu) {
    if (config.onnx_path.empty()) {
      throw std::runtime_error("onnx_path must not be empty");
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (config.inference_device == InferenceDevice::Cuda) {
      if (!has_cuda_provider()) {
        throw std::runtime_error("CUDAExecutionProvider is not available for ONNX Runtime");
      }
      OrtCUDAProviderOptions cuda_options;
      options.AppendExecutionProvider_CUDA(cuda_options);
    }

    session_ = std::make_unique<Ort::Session>(env_, config.onnx_path.c_str(), options);
  }

  std::vector<std::array<float, kPolicyLogitCount>> run_logits_batch(
      const std::vector<std::vector<float>>& inputs) override {
    if (inputs.empty()) {
      throw std::runtime_error("ONNX batch must contain at least one input");
    }

    std::vector<float> batch_input;
    batch_input.reserve(inputs.size() * model_input_feature_count_);
    for (const auto& input : inputs) {
      if (input.size() != model_input_feature_count_) {
        throw std::runtime_error("ONNX policy input feature count mismatch");
      }
      batch_input.insert(batch_input.end(), input.begin(), input.end());
    }

    std::array<std::int64_t, 2> input_shape{
        static_cast<std::int64_t>(inputs.size()),
        static_cast<std::int64_t>(model_input_feature_count_)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        batch_input.data(),
        batch_input.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
      throw std::runtime_error("ONNX Runtime must return one tensor output");
    }

    Ort::TensorTypeAndShapeInfo shape = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> dims = shape.GetShape();
    if (dims.size() != 2 || dims[0] != static_cast<std::int64_t>(inputs.size()) ||
        dims[1] != kPolicyLogitCount) {
      throw std::runtime_error("ONNX output shape must be [batch, 53]");
    }

    const float* output_data = outputs[0].GetTensorData<float>();
    std::vector<std::array<float, kPolicyLogitCount>> result(inputs.size());
    for (std::size_t row = 0; row < inputs.size(); ++row) {
      std::copy(
          output_data + row * kPolicyLogitCount,
          output_data + (row + 1) * kPolicyLogitCount,
          result[row].begin());
    }
    return result;
  }

  ExecutionProvider execution_provider() const override {
    return provider_;
  }

  std::size_t model_input_feature_count() const override {
    return model_input_feature_count_;
  }

 private:
  Ort::Env env_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;
  std::size_t model_input_feature_count_ = observation::kPlayingModelInputFeatureCount;
  ExecutionProvider provider_ = ExecutionProvider::Cpu;
  std::unique_ptr<Ort::Session> session_;
};

#endif

}  // namespace

struct BatchedPolicyExecutor::QueueItem {
  AgentRequest request;
  std::vector<float> model_input;
  std::array<int, observation::kCardCount> legal_play_mask;
};

struct BatchedPolicyExecutor::PolicyState {
  std::unique_ptr<PolicySession> session;
  std::deque<QueueItem> queue;
  PolicyBatchStats stats;
};

DeterministicPolicySession::DeterministicPolicySession(
    std::array<float, kPolicyLogitCount> logits,
    ExecutionProvider provider,
    std::size_t model_input_feature_count)
    : logits_(logits),
      provider_(provider),
      model_input_feature_count_(model_input_feature_count) {
  if (model_input_feature_count_ == 0) {
    throw std::runtime_error("deterministic policy feature count must be positive");
  }
}

std::vector<std::array<float, kPolicyLogitCount>> DeterministicPolicySession::run_logits_batch(
    const std::vector<std::vector<float>>& inputs) {
  if (inputs.empty()) {
    throw std::runtime_error("deterministic policy batch must contain at least one input");
  }
  for (const auto& input : inputs) {
    if (input.size() != model_input_feature_count_) {
      throw std::runtime_error("deterministic policy input feature count mismatch");
    }
  }
  session_run_count_ += 1;
  return std::vector<std::array<float, kPolicyLogitCount>>(inputs.size(), logits_);
}

std::size_t DeterministicPolicySession::model_input_feature_count() const {
  return model_input_feature_count_;
}

ExecutionProvider DeterministicPolicySession::execution_provider() const {
  return provider_;
}

std::uint64_t DeterministicPolicySession::session_run_count() const {
  return session_run_count_;
}

std::array<float, kPolicyLogitCount> DeterministicPolicySession::default_logits() {
  std::array<float, kPolicyLogitCount> logits{};
  for (int index = 0; index < kPolicyLogitCount; ++index) {
    logits[static_cast<std::size_t>(index)] = static_cast<float>(index) / 10.0F;
  }
  return logits;
}

std::unique_ptr<PolicySession> create_onnxruntime_policy_session(
    const PolicySessionConfig& config) {
#ifdef NAPOLEON_ENABLE_ONNXRUNTIME
  return std::make_unique<OnnxRuntimePolicySession>(config);
#else
  (void)config;
  throw std::runtime_error(
      "ONNX Runtime support is not enabled; configure with -DNAPOLEON_ENABLE_ONNXRUNTIME=ON");
#endif
}

BatchedPolicyExecutor::BatchedPolicyExecutor(BatchedPolicyConfig config)
    : config_(std::move(config)) {
  if (config_.max_batch_size == 0) {
    throw std::runtime_error("max_batch_size must be positive");
  }
  if (config_.temperature == 0.0) {
    config_.temperature = kDefaultTemperature;
  }
  validate_temperature(config_.temperature);
}

BatchedPolicyExecutor::~BatchedPolicyExecutor() = default;

void BatchedPolicyExecutor::add_policy(PolicyKey key, std::unique_ptr<PolicySession> session) {
  if (!session) {
    throw std::runtime_error("policy session must not be null");
  }
  const std::string id = policy_key_id(key);
  std::lock_guard<std::mutex> lock(mutex_);
  if (policies_.find(id) != policies_.end()) {
    throw std::runtime_error("duplicate policy session: " + id);
  }
  auto state = std::make_unique<PolicyState>();
  state->session = std::move(session);
  policies_.emplace(id, std::move(state));
}

void BatchedPolicyExecutor::enqueue(const AgentRequest& request) {
  if (request.phase != Phase::Playing) {
    throw std::runtime_error("ONNX policy executor only supports active playing requests");
  }
  if (request.legal_actions.empty()) {
    throw std::runtime_error("agent request has no legal actions");
  }

  const std::string id = policy_key_id(policy_key_from_agent(request.agent));
  std::lock_guard<std::mutex> lock(mutex_);
  auto policy_it = policies_.find(id);
  if (policy_it == policies_.end()) {
    throw std::runtime_error("missing policy session for " + id);
  }

  QueueItem item;
  item.request = request;
  item.model_input =
      request_model_input(request, policy_it->second->session->model_input_feature_count());
  item.legal_play_mask = request_legal_play_mask(request);
  policy_it->second->queue.push_back(std::move(item));
}

std::vector<PolicyActionResult> BatchedPolicyExecutor::flush() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PolicyActionResult> results;

  for (auto& policy_entry : policies_) {
    const std::string& policy_id = policy_entry.first;
    PolicyState& policy = *policy_entry.second;
    while (!policy.queue.empty()) {
      const std::size_t batch_size =
          std::min(config_.max_batch_size, static_cast<std::size_t>(policy.queue.size()));
      std::vector<QueueItem> batch;
      batch.reserve(batch_size);
      std::vector<std::vector<float>> inputs;
      inputs.reserve(batch_size);
      auto queue_it = policy.queue.begin();
      for (std::size_t index = 0; index < batch_size; ++index, ++queue_it) {
        batch.push_back(*queue_it);
        inputs.push_back(batch.back().model_input);
      }

      const auto started = std::chrono::steady_clock::now();
      std::vector<std::array<float, kPolicyLogitCount>> logits =
          policy.session->run_logits_batch(inputs);
      const auto ended = std::chrono::steady_clock::now();
      if (logits.size() != batch.size()) {
        throw std::runtime_error("policy session returned a mismatched batch size");
      }
      const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
      std::vector<PolicyActionResult> batch_results;
      batch_results.reserve(batch.size());

      for (std::size_t index = 0; index < batch.size(); ++index) {
        const AgentRequest& request = batch[index].request;
        const std::uint32_t seed =
            sampling_seed_for_request(config_.sampling_seed, request, policy_id);
        const SampledAction sampled = sample_legal_action(
            logits[index],
            batch[index].legal_play_mask,
            config_.temperature,
            seed);
        Action action = action_from_selected_card_index(request, sampled.selected_card_index);
        AgentResult result;
        result.request_id = request.request_id;
        result.action = action;
        result.selected_card_index = sampled.selected_card_index;
        result.behavior_log_probability = sampled.log_probability;
        result.policy_key = policy_id;
        batch_results.push_back(PolicyActionResult{
            result,
            request.sequence,
            sampled.selected_card_index,
            sampled.log_probability,
            policy_id});
      }

      for (std::size_t index = 0; index < batch.size(); ++index) {
        policy.queue.pop_front();
      }
      record_batch(policy.stats, batch.size(), elapsed_ns);
      results.insert(
          results.end(),
          std::make_move_iterator(batch_results.begin()),
          std::make_move_iterator(batch_results.end()));
    }
  }

  std::sort(
      results.begin(),
      results.end(),
      [](const PolicyActionResult& left, const PolicyActionResult& right) {
        return left.sequence < right.sequence;
      });
  return results;
}

std::vector<PolicyActionResult> BatchedPolicyExecutor::run(
    const std::vector<AgentRequest>& requests) {
  for (const AgentRequest& request : requests) {
    enqueue(request);
  }
  return flush();
}

BatchedPolicyStats BatchedPolicyExecutor::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  BatchedPolicyStats aggregate;
  for (const auto& entry : policies_) {
    aggregate.policy_stats[entry.first] = entry.second->stats;
    merge_policy_stats(aggregate, entry.second->stats);
  }
  if (aggregate.session_run_count > 0) {
    aggregate.mean_batch_size =
        static_cast<double>(aggregate.request_count) /
        static_cast<double>(aggregate.session_run_count);
  }
  return aggregate;
}

void BatchedPolicyExecutor::reset_stats() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& entry : policies_) {
    entry.second->stats = PolicyBatchStats{};
  }
}

std::string policy_key_id(const PolicyKey& key) {
  if (key.agent_id.empty()) {
    throw std::runtime_error("policy agent id must not be empty");
  }
  return agent_type_id(key.agent_type) + ":" + key.agent_id;
}

PolicyKey policy_key_from_agent(const AgentIdentity& agent) {
  if (agent.type == AgentType::RuleBased) {
    throw std::runtime_error("rule-based agent does not have an ONNX policy key");
  }
  return PolicyKey{agent.type, agent.id};
}

void attach_playing_model_input(
    const GameState& state,
    int player_index,
    AgentRequest& request) {
  attach_playing_model_input(state, player_index, observation::PlayingObservationVariant::Public, request);
}

void attach_playing_model_input(
    const GameState& state,
    int player_index,
    observation::PlayingObservationVariant variant,
    AgentRequest& request) {
  if (request.phase != Phase::Playing) {
    return;
  }
  const observation::VariantPlayingModelInput input =
      observation::create_playing_model_input(state, player_index, variant);
  request.playing_model_input = input.model_input;
  request.legal_play_mask.assign(input.legal_play_mask.begin(), input.legal_play_mask.end());
}

std::string execution_provider_id(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider::Cpu:
      return "cpu";
    case ExecutionProvider::Cuda:
      return "cuda";
  }
  throw std::runtime_error("invalid execution provider");
}

std::string inference_device_id(InferenceDevice device) {
  switch (device) {
    case InferenceDevice::Cpu:
      return "cpu";
    case InferenceDevice::Cuda:
      return "cuda";
  }
  throw std::runtime_error("invalid inference device");
}

}  // namespace napoleon::onnx_policy
