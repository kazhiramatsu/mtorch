#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "mtorch/core/tensor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MTORCH_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace {

using mtorch::Device;
using mtorch::DeviceType;
using mtorch::ScalarType;
using mtorch::Tensor;
using mtorch::TensorPtr;
using mtorch::UniqueResult;

PyObject* TensorType = nullptr;
PyObject* GeneratorType = nullptr;

bool is_floating_scalar_type(ScalarType dtype) {
  return dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64;
}

#if defined(MTORCH_USE_ACCELERATE)
bool accelerate_int_ok(int64_t value) {
  return value >= 0 && value <= std::numeric_limits<int>::max();
}
#endif

struct RandomState {
  uint64_t state = 0;
  bool has_spare_normal = false;
  double spare_normal = 0.0;
};

struct PyGenerator {
  PyObject_HEAD
  RandomState rng;
  bool uses_global = false;
  uint64_t initial_seed = 0;
};

RandomState& global_rng() {
  static RandomState rng;
  return rng;
}

uint64_t& global_initial_seed() {
  static uint64_t seed = 0;
  return seed;
}

void seed_random_state(RandomState& rng, uint64_t seed) {
  rng.state = seed;
  rng.has_spare_normal = false;
  rng.spare_normal = 0.0;
}

uint64_t next_random_u64(RandomState& rng = global_rng()) {
  rng.state += 0x9E3779B97F4A7C15ULL;
  uint64_t value = rng.state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

double next_uniform_open(RandomState& rng = global_rng()) {
  constexpr double kScale = 1.0 / 9007199254740992.0;
  const double value = static_cast<double>(next_random_u64(rng) >> 11U) * kScale;
  return value <= 0.0 ? kScale : value;
}

void next_standard_normal_pair(double& first, double& second, RandomState& rng = global_rng()) {
  double u = 0.0;
  double v = 0.0;
  double radius_squared = 0.0;
  do {
    u = 2.0 * next_uniform_open(rng) - 1.0;
    v = 2.0 * next_uniform_open(rng) - 1.0;
    radius_squared = u * u + v * v;
  } while (radius_squared >= 1.0 || radius_squared <= 0.0);
  const double scale = std::sqrt(-2.0 * std::log(radius_squared) / radius_squared);
  first = u * scale;
  second = v * scale;
}

double next_standard_normal(RandomState& rng = global_rng()) {
  if (rng.has_spare_normal) {
    rng.has_spare_normal = false;
    return rng.spare_normal;
  }
  double first = 0.0;
  double second = 0.0;
  next_standard_normal_pair(first, second, rng);
  rng.spare_normal = second;
  rng.has_spare_normal = true;
  return first;
}

template <typename T>
void fill_random_normal_contiguous(
    T* data,
    int64_t elements,
    double mean = 0.0,
    double std = 1.0,
    RandomState& rng = global_rng()) {
  int64_t i = 0;
  if (rng.has_spare_normal && elements > 0) {
    data[i++] = static_cast<T>(mean + std * rng.spare_normal);
    rng.has_spare_normal = false;
  }
  while (i + 1 < elements) {
    double first = 0.0;
    double second = 0.0;
    next_standard_normal_pair(first, second, rng);
    data[i++] = static_cast<T>(mean + std * first);
    data[i++] = static_cast<T>(mean + std * second);
  }
  if (i < elements) {
    double first = 0.0;
    double second = 0.0;
    next_standard_normal_pair(first, second, rng);
    data[i++] = static_cast<T>(mean + std * first);
    rng.spare_normal = second;
    rng.has_spare_normal = true;
  }
}

template <typename T>
void fill_random_uniform_contiguous(
    T* data,
    int64_t elements,
    double from,
    double to,
    RandomState& rng = global_rng()) {
  const double scale = to - from;
  for (int64_t i = 0; i < elements; ++i) {
    data[i] = static_cast<T>(from + scale * next_uniform_open(rng));
  }
}

void fill_randn_result(const TensorPtr& result, RandomState& rng = global_rng()) {
  const int64_t elements = result->numel();
  if (elements == 0) {
    return;
  }
  switch (result->dtype) {
    case ScalarType::Float32:
      fill_random_normal_contiguous(
          reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset, elements, 0.0, 1.0, rng);
      return;
    case ScalarType::Float64:
      fill_random_normal_contiguous(
          reinterpret_cast<double*>(result->storage->bytes.data()) + result->offset, elements, 0.0, 1.0, rng);
      return;
    case ScalarType::Float16:
      for (int64_t i = 0; i < elements; ++i) {
        result->set_at_linear(i, next_standard_normal(rng));
      }
      return;
    case ScalarType::Int32:
    case ScalarType::Int64:
    case ScalarType::Bool:
      throw std::invalid_argument("randn is only implemented for floating point dtypes");
  }
}

int64_t next_random_int64(int64_t low, int64_t high, RandomState& rng = global_rng()) {
  if (high <= low) {
    throw std::invalid_argument("randint expects high to be greater than low");
  }
  const uint64_t range = static_cast<uint64_t>(high - low);
  return low + static_cast<int64_t>(next_random_u64(rng) % range);
}

void fill_randint_result(const TensorPtr& result, int64_t low, int64_t high, RandomState& rng = global_rng()) {
  const int64_t elements = result->numel();
  if (elements == 0) {
    return;
  }
  if (result->dtype == ScalarType::Bool && (low < 0 || high > 2)) {
    throw std::invalid_argument("randint bool dtype expects values in [0, 2)");
  }
  switch (result->dtype) {
    case ScalarType::Int64: {
      auto* data = reinterpret_cast<int64_t*>(result->storage->bytes.data()) + result->offset;
      for (int64_t i = 0; i < elements; ++i) {
        data[i] = next_random_int64(low, high, rng);
      }
      return;
    }
    case ScalarType::Int32: {
      auto* data = reinterpret_cast<int32_t*>(result->storage->bytes.data()) + result->offset;
      for (int64_t i = 0; i < elements; ++i) {
        data[i] = static_cast<int32_t>(next_random_int64(low, high, rng));
      }
      return;
    }
    case ScalarType::Float32: {
      auto* data = reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset;
      for (int64_t i = 0; i < elements; ++i) {
        data[i] = static_cast<float>(next_random_int64(low, high, rng));
      }
      return;
    }
    case ScalarType::Float64: {
      auto* data = reinterpret_cast<double*>(result->storage->bytes.data()) + result->offset;
      for (int64_t i = 0; i < elements; ++i) {
        data[i] = static_cast<double>(next_random_int64(low, high, rng));
      }
      return;
    }
    case ScalarType::Bool: {
      auto* data = reinterpret_cast<uint8_t*>(result->storage->bytes.data()) + result->offset;
      for (int64_t i = 0; i < elements; ++i) {
        data[i] = next_random_int64(low, high, rng) == 0 ? uint8_t{0} : uint8_t{1};
      }
      return;
    }
    case ScalarType::Float16:
      for (int64_t i = 0; i < elements; ++i) {
        result->set_at_linear(i, static_cast<double>(next_random_int64(low, high, rng)));
      }
      return;
  }
}

void normal_inplace(Tensor& tensor, double mean, double std, RandomState& rng = global_rng()) {
  if (!is_floating_scalar_type(tensor.dtype)) {
    throw std::invalid_argument("normal_ is only implemented for floating point tensors");
  }
  const int64_t elements = tensor.numel();
  if (elements == 0) {
    return;
  }
  tensor.mark_storage_modified();
  if (tensor.is_contiguous()) {
    switch (tensor.dtype) {
      case ScalarType::Float32:
        fill_random_normal_contiguous(
            reinterpret_cast<float*>(tensor.storage->bytes.data()) + tensor.offset, elements, mean, std, rng);
        return;
      case ScalarType::Float64:
        fill_random_normal_contiguous(
            reinterpret_cast<double*>(tensor.storage->bytes.data()) + tensor.offset, elements, mean, std, rng);
        return;
      case ScalarType::Float16:
      case ScalarType::Int32:
      case ScalarType::Int64:
      case ScalarType::Bool:
        break;
    }
  }
  for (int64_t i = 0; i < elements; ++i) {
    tensor.set_at_linear(i, mean + std * next_standard_normal(rng));
  }
}

double next_truncated_normal(double mean, double std, double a, double b, RandomState& rng = global_rng()) {
  double value = 0.0;
  do {
    value = mean + std * next_standard_normal(rng);
  } while (value < a || value > b);
  return value;
}

void trunc_normal_inplace(Tensor& tensor, double mean, double std, double a, double b, RandomState& rng = global_rng()) {
  if (!is_floating_scalar_type(tensor.dtype)) {
    throw std::invalid_argument("trunc_normal_ is only implemented for floating point tensors");
  }
  if (a > b) {
    throw std::invalid_argument("trunc_normal_ expects a <= b");
  }
  if (std < 0.0) {
    throw std::invalid_argument("trunc_normal_ expects std >= 0");
  }
  if (std == 0.0) {
    if (mean < a || mean > b) {
      throw std::invalid_argument("trunc_normal_ with std=0 expects mean within [a, b]");
    }
    tensor.fill_inplace(mean);
    return;
  }

  const int64_t elements = tensor.numel();
  if (elements == 0) {
    return;
  }
  tensor.mark_storage_modified();
  if (tensor.is_contiguous()) {
    switch (tensor.dtype) {
      case ScalarType::Float32: {
        auto* data = reinterpret_cast<float*>(tensor.storage->bytes.data()) + tensor.offset;
        for (int64_t i = 0; i < elements; ++i) {
          data[i] = static_cast<float>(next_truncated_normal(mean, std, a, b, rng));
        }
        return;
      }
      case ScalarType::Float64: {
        auto* data = reinterpret_cast<double*>(tensor.storage->bytes.data()) + tensor.offset;
        for (int64_t i = 0; i < elements; ++i) {
          data[i] = next_truncated_normal(mean, std, a, b, rng);
        }
        return;
      }
      case ScalarType::Float16:
      case ScalarType::Int32:
      case ScalarType::Int64:
      case ScalarType::Bool:
        break;
    }
  }
  for (int64_t i = 0; i < elements; ++i) {
    tensor.set_at_linear(i, next_truncated_normal(mean, std, a, b, rng));
  }
}

void uniform_inplace(Tensor& tensor, double from, double to, RandomState& rng = global_rng()) {
  if (!is_floating_scalar_type(tensor.dtype)) {
    throw std::invalid_argument("uniform_ is only implemented for floating point tensors");
  }
  const int64_t elements = tensor.numel();
  if (elements == 0) {
    return;
  }
  tensor.mark_storage_modified();
  if (tensor.is_contiguous()) {
    switch (tensor.dtype) {
      case ScalarType::Float32:
        fill_random_uniform_contiguous(
            reinterpret_cast<float*>(tensor.storage->bytes.data()) + tensor.offset, elements, from, to, rng);
        return;
      case ScalarType::Float64:
        fill_random_uniform_contiguous(
            reinterpret_cast<double*>(tensor.storage->bytes.data()) + tensor.offset, elements, from, to, rng);
        return;
      case ScalarType::Float16:
      case ScalarType::Int32:
      case ScalarType::Int64:
      case ScalarType::Bool:
        break;
    }
  }
  const double scale = to - from;
  for (int64_t i = 0; i < elements; ++i) {
    tensor.set_at_linear(i, from + scale * next_uniform_open(rng));
  }
}

template <typename T>
void fill_bernoulli_contiguous(const T* probabilities, T* target, int64_t elements, RandomState& rng) {
  for (int64_t i = 0; i < elements; ++i) {
    const double probability = static_cast<double>(probabilities[i]);
    if (probability < 0.0 || probability > 1.0) {
      throw std::runtime_error("bernoulli probability tensor must contain values in [0, 1]");
    }
    target[i] = next_uniform_open(rng) < probability ? static_cast<T>(1) : static_cast<T>(0);
  }
}

TensorPtr bernoulli_tensor(const TensorPtr& input, RandomState& rng = global_rng()) {
  if (!is_floating_scalar_type(input->dtype)) {
    throw std::runtime_error("bernoulli is only implemented for floating point tensors");
  }
  auto result = mtorch::zeros(input->sizes, input->dtype, input->device);
  const bool requires_grad = mtorch::is_grad_enabled() && input->requires_grad;
  result->requires_grad = requires_grad;
  const int64_t elements = input->numel();
  if (elements == 0) {
    return result;
  }
  if (input->is_contiguous() && result->is_contiguous()) {
    switch (input->dtype) {
      case ScalarType::Float32:
        fill_bernoulli_contiguous(
            reinterpret_cast<const float*>(input->storage->bytes.data()) + input->offset,
            reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset,
            elements,
            rng);
        break;
      case ScalarType::Float64:
        fill_bernoulli_contiguous(
            reinterpret_cast<const double*>(input->storage->bytes.data()) + input->offset,
            reinterpret_cast<double*>(result->storage->bytes.data()) + result->offset,
            elements,
            rng);
        break;
      case ScalarType::Float16:
      case ScalarType::Int32:
      case ScalarType::Int64:
      case ScalarType::Bool:
        for (int64_t i = 0; i < elements; ++i) {
          const double probability = input->value_at_linear(i);
          if (probability < 0.0 || probability > 1.0) {
            throw std::runtime_error("bernoulli probability tensor must contain values in [0, 1]");
          }
          result->set_at_linear(i, next_uniform_open(rng) < probability ? 1.0 : 0.0);
        }
        break;
    }
  } else {
    for (int64_t i = 0; i < elements; ++i) {
      const double probability = input->value_at_linear(i);
      if (probability < 0.0 || probability > 1.0) {
        throw std::runtime_error("bernoulli probability tensor must contain values in [0, 1]");
      }
      result->set_at_linear(i, next_uniform_open(rng) < probability ? 1.0 : 0.0);
    }
  }
  if (requires_grad) {
    result->parents = {input};
    result->backward_fn = [input](const Tensor&) {
      input->backward_with(*mtorch::zeros(input->sizes, ScalarType::Float32, input->device));
    };
  }
  return result;
}

TensorPtr dropout_tensor(const TensorPtr& input, double p, bool training) {
  if (p < 0.0 || p > 1.0) {
    throw std::invalid_argument("dropout probability has to be between 0 and 1");
  }
  if (!training || p == 0.0) {
    return input;
  }
  if (!is_floating_scalar_type(input->dtype)) {
    throw std::invalid_argument("dropout input must be floating point");
  }

  const int64_t elements = input->numel();
  auto result = mtorch::zeros(input->sizes, input->dtype, input->device);
  result->requires_grad = mtorch::is_grad_enabled() && input->requires_grad;
  TensorPtr mask = result->requires_grad ? mtorch::zeros(input->sizes, ScalarType::Float32, input->device) : nullptr;
  if (elements == 0) {
    return result;
  }

  const double scale = p == 1.0 ? 0.0 : 1.0 / (1.0 - p);
  if (p == 1.0 && !result->requires_grad) {
    return result;
  }
  float* mask_data = mask ? reinterpret_cast<float*>(mask->storage->bytes.data()) + mask->offset : nullptr;
  if (p != 1.0 && input->is_contiguous()) {
    switch (input->dtype) {
      case ScalarType::Float32: {
        const float* input_data = reinterpret_cast<const float*>(input->storage->bytes.data()) + input->offset;
        float* output_data = reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset;
        for (int64_t i = 0; i < elements; ++i) {
          const float factor = next_uniform_open() < p ? 0.0f : static_cast<float>(scale);
          if (mask_data) {
            mask_data[i] = factor;
          }
          output_data[i] = input_data[i] * factor;
        }
        break;
      }
      case ScalarType::Float64: {
        const double* input_data = reinterpret_cast<const double*>(input->storage->bytes.data()) + input->offset;
        double* output_data = reinterpret_cast<double*>(result->storage->bytes.data()) + result->offset;
        for (int64_t i = 0; i < elements; ++i) {
          const float factor = next_uniform_open() < p ? 0.0f : static_cast<float>(scale);
          if (mask_data) {
            mask_data[i] = factor;
          }
          output_data[i] = input_data[i] * static_cast<double>(factor);
        }
        break;
      }
      case ScalarType::Float16:
      case ScalarType::Int32:
      case ScalarType::Int64:
      case ScalarType::Bool:
        for (int64_t i = 0; i < elements; ++i) {
          const double factor = next_uniform_open() < p ? 0.0 : scale;
          if (mask) {
            mask->set_at_linear(i, factor);
          }
          result->set_at_linear(i, input->value_at_linear(i) * factor);
        }
        break;
    }
  } else if (p != 1.0) {
    for (int64_t i = 0; i < elements; ++i) {
      const double factor = next_uniform_open() < p ? 0.0 : scale;
      if (mask) {
        mask->set_at_linear(i, factor);
      }
      result->set_at_linear(i, input->value_at_linear(i) * factor);
    }
  }

  if (result->requires_grad) {
    result->parents = {input};
    result->backward_fn = [input, mask, elements](const Tensor& upstream) {
      auto grad = mtorch::zeros(input->sizes, ScalarType::Float32, input->device);
      if (upstream.dtype == ScalarType::Float32 && upstream.is_contiguous() && grad->is_contiguous() &&
          mask->is_contiguous()) {
        const float* upstream_data = reinterpret_cast<const float*>(upstream.storage->bytes.data()) + upstream.offset;
        const float* mask_data = reinterpret_cast<const float*>(mask->storage->bytes.data()) + mask->offset;
        float* grad_data = reinterpret_cast<float*>(grad->storage->bytes.data()) + grad->offset;
        for (int64_t i = 0; i < elements; ++i) {
          grad_data[i] = upstream_data[i] * mask_data[i];
        }
      } else {
        for (int64_t i = 0; i < elements; ++i) {
          grad->set_at_linear(i, upstream.value_at_linear(i) * mask->value_at_linear(i));
        }
      }
      input->backward_with(*grad);
    };
  }
  return result;
}

void validate_multinomial_weight(double weight) {
  if (!std::isfinite(weight) || weight < 0.0) {
    throw std::runtime_error("probability tensor contains either `inf`, `nan` or element < 0");
  }
}

std::vector<int64_t> multinomial_row_samples(
    const Tensor& input,
    int64_t row,
    int64_t categories,
    int64_t num_samples,
    bool replacement,
    RandomState& rng) {
  std::vector<double> weights(static_cast<size_t>(categories));
  double total = 0.0;
  int64_t positive_count = 0;
  int64_t only_positive_index = -1;
  for (int64_t col = 0; col < categories; ++col) {
    const double weight = input.dim() == 1 ? input.value_at_linear(col) : input.value_at_index({row, col});
    validate_multinomial_weight(weight);
    weights[static_cast<size_t>(col)] = weight;
    total += weight;
    if (weight > 0.0) {
      ++positive_count;
      only_positive_index = col;
    }
  }
  if (total <= 0.0) {
    throw std::runtime_error("invalid multinomial distribution (sum of probabilities <= 0)");
  }

  std::vector<int64_t> samples;
  samples.reserve(static_cast<size_t>(num_samples));
  if (replacement) {
    if (positive_count == 1) {
      samples.assign(static_cast<size_t>(num_samples), only_positive_index);
      return samples;
    }
    std::vector<double> cumulative(static_cast<size_t>(categories));
    double running_total = 0.0;
    for (int64_t col = 0; col < categories; ++col) {
      running_total += weights[static_cast<size_t>(col)];
      cumulative[static_cast<size_t>(col)] = running_total;
    }
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      const double target = next_uniform_open(rng) * total;
      auto iter = std::upper_bound(cumulative.begin(), cumulative.end(), target);
      int64_t chosen = iter == cumulative.end()
          ? categories - 1
          : static_cast<int64_t>(std::distance(cumulative.begin(), iter));
      while (chosen < categories - 1 && weights[static_cast<size_t>(chosen)] == 0.0) {
        ++chosen;
      }
      samples.push_back(chosen);
    }
    return samples;
  }

  std::vector<uint8_t> selected(static_cast<size_t>(categories), uint8_t{0});
  for (int64_t sample = 0; sample < num_samples; ++sample) {
    if (total <= 0.0) {
      for (int64_t col = 0; col < categories && static_cast<int64_t>(samples.size()) < num_samples; ++col) {
        if (selected[static_cast<size_t>(col)] == 0) {
          selected[static_cast<size_t>(col)] = uint8_t{1};
          samples.push_back(col);
        }
      }
      break;
    }

    const double target = next_uniform_open(rng) * total;
    double cumulative = 0.0;
    int64_t chosen = -1;
    for (int64_t col = 0; col < categories; ++col) {
      const double weight = weights[static_cast<size_t>(col)];
      if (selected[static_cast<size_t>(col)] != 0) {
        continue;
      }
      cumulative += weight;
      if (target < cumulative) {
        chosen = col;
        break;
      }
    }
    if (chosen < 0) {
      for (int64_t col = categories - 1; col >= 0; --col) {
        if (selected[static_cast<size_t>(col)] == 0 && weights[static_cast<size_t>(col)] > 0.0) {
          chosen = col;
          break;
        }
      }
    }
    if (chosen < 0) {
      for (int64_t col = 0; col < categories; ++col) {
        if (selected[static_cast<size_t>(col)] == 0) {
          chosen = col;
          break;
        }
      }
    }
    samples.push_back(chosen);
    selected[static_cast<size_t>(chosen)] = uint8_t{1};
    total -= weights[static_cast<size_t>(chosen)];
    weights[static_cast<size_t>(chosen)] = 0.0;
  }
  return samples;
}

void fill_multinomial_replacement_float32_row(
    const float* row_data,
    int64_t categories,
    int64_t num_samples,
    int64_t* output_data,
    std::vector<double>& cumulative,
    RandomState& rng) {
  double total = 0.0;
  int64_t positive_count = 0;
  int64_t only_positive_index = -1;
  for (int64_t col = 0; col < categories; ++col) {
    const double weight = static_cast<double>(row_data[col]);
    validate_multinomial_weight(weight);
    total += weight;
    if (weight > 0.0) {
      ++positive_count;
      only_positive_index = col;
    }
  }
  if (total <= 0.0) {
    throw std::runtime_error("invalid multinomial distribution (sum of probabilities <= 0)");
  }
  if (positive_count == 1) {
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      output_data[sample] = only_positive_index;
    }
    return;
  }

  cumulative.resize(static_cast<size_t>(categories));
  double running_total = 0.0;
  for (int64_t col = 0; col < categories; ++col) {
    running_total += static_cast<double>(row_data[col]);
    cumulative[static_cast<size_t>(col)] = running_total;
  }
  for (int64_t sample = 0; sample < num_samples; ++sample) {
    const double target = next_uniform_open(rng) * total;
    auto iter = std::upper_bound(cumulative.begin(), cumulative.end(), target);
    int64_t chosen = iter == cumulative.end()
        ? categories - 1
        : static_cast<int64_t>(std::distance(cumulative.begin(), iter));
    while (chosen < categories - 1 && row_data[chosen] == 0.0f) {
      ++chosen;
    }
    output_data[sample] = chosen;
  }
}

TensorPtr multinomial_tensor(
    const TensorPtr& input,
    int64_t num_samples,
    bool replacement,
    RandomState& rng = global_rng()) {
  if (!is_floating_scalar_type(input->dtype)) {
    throw std::runtime_error("multinomial only supports floating-point dtypes for input");
  }
  if (input->dim() != 1 && input->dim() != 2) {
    throw std::runtime_error("prob_dist must be 1 or 2 dim");
  }
  if (num_samples <= 0) {
    throw std::runtime_error("cannot sample n_sample <= 0 samples");
  }
  const int64_t categories = input->sizes[static_cast<size_t>(input->dim() - 1)];
  if (!replacement && num_samples > categories) {
    throw std::runtime_error("cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  }

  const int64_t rows = input->dim() == 1 ? 1 : input->sizes[0];
  std::vector<int64_t> out_shape = input->dim() == 1 ? std::vector<int64_t>{num_samples}
                                                     : std::vector<int64_t>{rows, num_samples};
  auto result = mtorch::zeros(out_shape, ScalarType::Int64, input->device);
  if (replacement && input->dtype == ScalarType::Float32 && input->is_contiguous() && result->is_contiguous()) {
    const float* input_data = reinterpret_cast<const float*>(input->storage->bytes.data()) + input->offset;
    int64_t* output_data = reinterpret_cast<int64_t*>(result->storage->bytes.data()) + result->offset;
    std::vector<double> cumulative;
    cumulative.reserve(static_cast<size_t>(categories));
    for (int64_t row = 0; row < rows; ++row) {
      const float* row_data = input_data + row * categories;
      int64_t* output_row = output_data + row * num_samples;
      fill_multinomial_replacement_float32_row(row_data, categories, num_samples, output_row, cumulative, rng);
    }
    return result;
  }
  for (int64_t row = 0; row < rows; ++row) {
    const auto samples = multinomial_row_samples(*input, row, categories, num_samples, replacement, rng);
    for (int64_t sample = 0; sample < num_samples; ++sample) {
      result->set_at_linear(row * num_samples + sample, static_cast<double>(samples[static_cast<size_t>(sample)]));
    }
  }
  return result;
}

struct TypeErrorException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct NotImplementedException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct PyTensor {
  PyObject_HEAD
  TensorPtr* value;
  bool is_parameter;
};

bool is_tensor(PyObject* object) {
  return TensorType != nullptr && PyObject_TypeCheck(object, reinterpret_cast<PyTypeObject*>(TensorType));
}

TensorPtr& tensor_ref(PyObject* object) {
  return *reinterpret_cast<PyTensor*>(object)->value;
}

bool object_is_sequence(PyObject* object);

std::vector<TensorPtr> tensor_sequence_from_py(PyObject* object, const char* name, bool allow_single_tensor = true) {
  if (allow_single_tensor && is_tensor(object)) {
    return {tensor_ref(object)};
  }
  if (!object_is_sequence(object)) {
    std::ostringstream message;
    message << name << " must be a Tensor or a sequence of tensors";
    throw TypeErrorException(message.str());
  }
  const Py_ssize_t length = PySequence_Length(object);
  if (length < 0) {
    std::ostringstream message;
    message << "could not read " << name;
    throw std::invalid_argument(message.str());
  }
  std::vector<TensorPtr> tensors;
  tensors.reserve(static_cast<size_t>(length));
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      std::ostringstream message;
      message << "could not read " << name;
      throw std::invalid_argument(message.str());
    }
    if (!is_tensor(item)) {
      Py_DECREF(item);
      std::ostringstream message;
      message << name << " entries must be tensors";
      throw TypeErrorException(message.str());
    }
    tensors.push_back(tensor_ref(item));
    Py_DECREF(item);
  }
  return tensors;
}

void translate_exception() {
  try {
    throw;
  } catch (const TypeErrorException& exc) {
    PyErr_SetString(PyExc_TypeError, exc.what());
  } catch (const NotImplementedException& exc) {
    PyErr_SetString(PyExc_NotImplementedError, exc.what());
  } catch (const std::out_of_range& exc) {
    PyErr_SetString(PyExc_IndexError, exc.what());
  } catch (const std::invalid_argument& exc) {
    PyErr_SetString(PyExc_ValueError, exc.what());
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception");
  }
}

PyObject* wrap_tensor(const TensorPtr& tensor) {
  auto* object = PyObject_New(PyTensor, reinterpret_cast<PyTypeObject*>(TensorType));
  if (object == nullptr) {
    return nullptr;
  }
  object->value = new TensorPtr(tensor);
  object->is_parameter = false;
  return reinterpret_cast<PyObject*>(object);
}

ScalarType dtype_from_py(PyObject* object, ScalarType fallback = ScalarType::Float32) {
  if (object == nullptr || object == Py_None) {
    return fallback;
  }
  PyObject* text_object = nullptr;
  if (PyUnicode_Check(object)) {
    text_object = object;
    Py_INCREF(text_object);
  } else if (PyObject_HasAttrString(object, "name")) {
    text_object = PyObject_GetAttrString(object, "name");
  } else {
    text_object = PyObject_Str(object);
  }
  if (text_object == nullptr) {
    throw std::invalid_argument("could not parse dtype");
  }
  PyObject* normalized = text_object;
  if (!PyUnicode_Check(normalized)) {
    normalized = PyObject_Str(text_object);
    Py_DECREF(text_object);
    if (normalized == nullptr) {
      throw std::invalid_argument("could not parse dtype");
    }
  }
  const char* text = PyUnicode_AsUTF8(normalized);
  if (text == nullptr) {
    Py_DECREF(normalized);
    throw std::invalid_argument("could not parse dtype");
  }
  const std::string dtype_text(text);
  Py_DECREF(normalized);

  if (dtype_text.find("float16") != std::string::npos) {
    return ScalarType::Float16;
  }
  if (dtype_text.find("float64") != std::string::npos || dtype_text.find("double") != std::string::npos) {
    return ScalarType::Float64;
  }
  if (dtype_text.find("int32") != std::string::npos) {
    return ScalarType::Int32;
  }
  if (dtype_text.find("int64") != std::string::npos || dtype_text.find("long") != std::string::npos) {
    return ScalarType::Int64;
  }
  if (dtype_text.find("bool") != std::string::npos) {
    return ScalarType::Bool;
  }
  return ScalarType::Float32;
}

bool dtype_allows_requires_grad(ScalarType dtype) {
  return dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64;
}

ScalarType default_arange_dtype(PyObject* dtype, PyObject* start, PyObject* end, PyObject* step) {
  if (dtype != nullptr && dtype != Py_None) {
    return dtype_from_py(dtype);
  }
  if (PyFloat_Check(start) || (end != nullptr && end != Py_None && PyFloat_Check(end)) ||
      (step != nullptr && step != Py_None && PyFloat_Check(step))) {
    return ScalarType::Float32;
  }
  return ScalarType::Int64;
}

std::string lowercase_ascii(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return text;
}

std::string object_text_lower(PyObject* object) {
  PyObject* text_object = nullptr;
  if (PyUnicode_Check(object)) {
    text_object = object;
    Py_INCREF(text_object);
  } else {
    text_object = PyObject_Str(object);
  }
  if (text_object == nullptr) {
    throw std::invalid_argument("could not parse Python object");
  }
  const char* text = PyUnicode_AsUTF8(text_object);
  if (text == nullptr) {
    Py_DECREF(text_object);
    throw std::invalid_argument("could not parse Python object");
  }
  std::string lowered = lowercase_ascii(text);
  Py_DECREF(text_object);
  return lowered;
}

bool text_looks_like_device(const std::string& text) {
  return text.find("cpu") != std::string::npos || text.find("mps") != std::string::npos ||
      text.find("metal") != std::string::npos || text.find("cuda") != std::string::npos;
}

enum class MemoryFormat {
  Preserve,
  Contiguous,
  ChannelsLast,
  ChannelsLast3d,
};

std::string py_type_name_lower(PyObject* object) {
  PyObject* type_object = PyObject_Type(object);
  if (type_object == nullptr) {
    throw TypeErrorException("could not inspect Python object type");
  }
  PyObject* name_object = PyObject_GetAttrString(type_object, "__name__");
  Py_DECREF(type_object);
  if (name_object == nullptr) {
    throw TypeErrorException("could not inspect Python object type");
  }
  PyObject* text_object = name_object;
  if (!PyUnicode_Check(text_object)) {
    text_object = PyObject_Str(name_object);
    Py_DECREF(name_object);
    if (text_object == nullptr) {
      throw TypeErrorException("could not inspect Python object type");
    }
  }
  const char* text = PyUnicode_AsUTF8(text_object);
  if (text == nullptr) {
    Py_DECREF(text_object);
    throw TypeErrorException("could not inspect Python object type");
  }
  std::string result = lowercase_ascii(text);
  Py_DECREF(text_object);
  return result;
}

MemoryFormat memory_format_from_py(PyObject* object) {
  if (object == nullptr || object == Py_None) {
    throw TypeErrorException("argument 'memory_format' must be mtorch.memory_format, not NoneType");
  }
  const std::string type_name = py_type_name_lower(object);
  if (type_name != "memory_format") {
    throw TypeErrorException("argument 'memory_format' must be mtorch.memory_format, not " + type_name);
  }
  const std::string text = object_text_lower(object);
  if (text.find("preserve_format") != std::string::npos) {
    return MemoryFormat::Preserve;
  }
  if (text.find("contiguous_format") != std::string::npos) {
    return MemoryFormat::Contiguous;
  }
  if (text.find("channels_last_3d") != std::string::npos) {
    return MemoryFormat::ChannelsLast3d;
  }
  if (text.find("channels_last") != std::string::npos) {
    return MemoryFormat::ChannelsLast;
  }
  throw TypeErrorException("unsupported memory_format value");
}

std::optional<MemoryFormat> optional_memory_format_from_py(PyObject* object) {
  if (object == nullptr || object == Py_None) {
    return std::nullopt;
  }
  return memory_format_from_py(object);
}

std::vector<int64_t> channels_last_strides_2d(const std::vector<int64_t>& sizes) {
  return {sizes[1] * sizes[2] * sizes[3], 1, sizes[3] * sizes[1], sizes[1]};
}

std::vector<int64_t> channels_last_strides_3d(const std::vector<int64_t>& sizes) {
  return {sizes[1] * sizes[2] * sizes[3] * sizes[4], 1, sizes[3] * sizes[4] * sizes[1], sizes[4] * sizes[1], sizes[1]};
}

bool tensor_is_contiguous_memory_format(const Tensor& tensor, MemoryFormat memory_format) {
  switch (memory_format) {
    case MemoryFormat::Preserve:
    case MemoryFormat::Contiguous:
      return tensor.is_contiguous();
    case MemoryFormat::ChannelsLast:
      return tensor.dim() == 4 && tensor.strides == channels_last_strides_2d(tensor.sizes);
    case MemoryFormat::ChannelsLast3d:
      return tensor.dim() == 5 && tensor.strides == channels_last_strides_3d(tensor.sizes);
  }
  return false;
}

void copy_storage_element(const Tensor& source, int64_t source_offset, Tensor& target, int64_t target_offset) {
  const int64_t bytes = mtorch::element_size(source.dtype);
  std::memcpy(
      target.storage->bytes.data() + static_cast<size_t>(target_offset * bytes),
      source.storage->bytes.data() + static_cast<size_t>(source_offset * bytes),
      static_cast<size_t>(bytes));
}

bool try_copy_contiguous_to_channels_last_4d(const Tensor& source, Tensor& target) {
  if (source.dim() != 4 || source.dtype != target.dtype || source.sizes != target.sizes ||
      !source.is_contiguous() || target.strides != channels_last_strides_2d(target.sizes)) {
    return false;
  }
  const int64_t batch = source.sizes[0];
  const int64_t channels = source.sizes[1];
  const int64_t height = source.sizes[2];
  const int64_t width = source.sizes[3];
  if (source.numel() == 0) {
    return true;
  }
  if (channels == 1) {
    const int64_t bytes = source.numel() * mtorch::element_size(source.dtype);
    std::memcpy(
        target.storage->bytes.data() + static_cast<size_t>(target.offset * mtorch::element_size(target.dtype)),
        source.storage->bytes.data() + static_cast<size_t>(source.offset * mtorch::element_size(source.dtype)),
        static_cast<size_t>(bytes));
    return true;
  }
  for (int64_t n = 0; n < batch; ++n) {
    const int64_t source_n_base = source.offset + n * channels * height * width;
    const int64_t target_n_base = target.offset + n * height * width * channels;
    for (int64_t h = 0; h < height; ++h) {
      const int64_t target_h_base = target_n_base + h * width * channels;
      for (int64_t w = 0; w < width; ++w) {
        const int64_t target_pixel_base = target_h_base + w * channels;
        for (int64_t c = 0; c < channels; ++c) {
          const int64_t source_offset = source_n_base + c * height * width + h * width + w;
          copy_storage_element(source, source_offset, target, target_pixel_base + c);
        }
      }
    }
  }
  return true;
}

TensorPtr tensor_to_memory_format(const TensorPtr& input, MemoryFormat memory_format, bool copy = false) {
  switch (memory_format) {
    case MemoryFormat::Preserve:
      return input;
    case MemoryFormat::Contiguous:
      if (!copy && input->is_contiguous()) {
        return input;
      }
      return input->contiguous();
    case MemoryFormat::ChannelsLast: {
      if (input->dim() != 4) {
        throw std::runtime_error("required rank 4 tensor to use channels_last format");
      }
      if (!copy && tensor_is_contiguous_memory_format(*input, memory_format)) {
        return input;
      }
      auto result = mtorch::empty_strided(
          input->sizes, channels_last_strides_2d(input->sizes), input->dtype, input->requires_grad, input->device);
      if (!try_copy_contiguous_to_channels_last_4d(*input, *result)) {
        result->copy_from(*input);
      }
      return result;
    }
    case MemoryFormat::ChannelsLast3d: {
      if (input->dim() != 5) {
        throw std::runtime_error("required rank 5 tensor to use channels_last_3d format");
      }
      if (!copy && tensor_is_contiguous_memory_format(*input, memory_format)) {
        return input;
      }
      auto result = mtorch::empty_strided(
          input->sizes, channels_last_strides_3d(input->sizes), input->dtype, input->requires_grad, input->device);
      result->copy_from(*input);
      return result;
    }
  }
  return input;
}

std::vector<int64_t> strides_for_like_memory_format(const Tensor& source, std::optional<MemoryFormat> memory_format) {
  const MemoryFormat format_value = memory_format.value_or(MemoryFormat::Preserve);
  switch (format_value) {
    case MemoryFormat::Preserve:
      return source.strides;
    case MemoryFormat::Contiguous:
      return mtorch::contiguous_strides(source.sizes);
    case MemoryFormat::ChannelsLast:
      if (source.dim() != 4) {
        throw std::runtime_error("required rank 4 tensor to use channels_last format");
      }
      return channels_last_strides_2d(source.sizes);
    case MemoryFormat::ChannelsLast3d:
      if (source.dim() != 5) {
        throw std::runtime_error("required rank 5 tensor to use channels_last_3d format");
      }
      return channels_last_strides_3d(source.sizes);
  }
  return source.strides;
}

TensorPtr make_like_with_memory_format(
    const TensorPtr& source,
    ScalarType dtype,
    Device device,
    bool requires_grad,
    std::optional<MemoryFormat> memory_format) {
  return mtorch::empty_strided(source->sizes, strides_for_like_memory_format(*source, memory_format), dtype, requires_grad, device);
}

Device device_from_py(PyObject* object, Device fallback = mtorch::cpu_device()) {
  if (object == nullptr || object == Py_None) {
    return fallback;
  }
  PyObject* text_object = PyObject_Str(object);
  if (text_object == nullptr) {
    throw std::invalid_argument("could not parse device");
  }
  const char* text = PyUnicode_AsUTF8(text_object);
  if (text == nullptr) {
    Py_DECREF(text_object);
    throw std::invalid_argument("could not parse device");
  }
  const std::string device_text = lowercase_ascii(text);
  Py_DECREF(text_object);

  if (device_text.find("cpu") != std::string::npos) {
    return mtorch::cpu_device();
  }
  if (device_text.find("mps") != std::string::npos || device_text.find("metal") != std::string::npos) {
    throw NotImplementedException("Metal/MPS device execution is not implemented yet");
  }
  if (device_text.find("cuda") != std::string::npos) {
    throw NotImplementedException("CUDA device execution is not implemented");
  }
  throw std::invalid_argument("unsupported device");
}

ScalarType default_sum_dtype(ScalarType input_dtype) {
  if (input_dtype == ScalarType::Bool || input_dtype == ScalarType::Int32) {
    return ScalarType::Int64;
  }
  return input_dtype;
}

bool object_is_sequence(PyObject* object) {
  return (PyList_Check(object) || PyTuple_Check(object)) && !is_tensor(object);
}

double scalar_from_py(PyObject* object) {
  if (PyBool_Check(object)) {
    return object == Py_True ? 1.0 : 0.0;
  }
  if (PyFloat_Check(object)) {
    return PyFloat_AS_DOUBLE(object);
  }
  if (PyLong_Check(object)) {
    const double value = PyLong_AsDouble(object);
    if (PyErr_Occurred()) {
      throw std::invalid_argument("expected a numeric scalar");
    }
    return value;
  }
  const double value = PyFloat_AsDouble(object);
  if (PyErr_Occurred()) {
    throw std::invalid_argument("expected a numeric scalar");
  }
  return value;
}

std::optional<double> optional_scalar_from_py(PyObject* object) {
  if (object == Py_None) {
    return std::nullopt;
  }
  return scalar_from_py(object);
}

std::optional<int64_t> optional_int64_from_py(PyObject* object, const char* name) {
  if (object == nullptr || object == Py_None) {
    return std::nullopt;
  }
  const int64_t value = PyLong_AsLongLong(object);
  if (PyErr_Occurred()) {
    throw std::invalid_argument(std::string(name) + " must be an integer");
  }
  return value;
}

double correction_from_py(PyObject* correction, PyObject* unbiased) {
  if (correction != Py_None) {
    return scalar_from_py(correction);
  }
  if (unbiased != Py_None) {
    const int value = PyObject_IsTrue(unbiased);
    if (value < 0) {
      throw std::invalid_argument("could not parse unbiased");
    }
    return value != 0 ? 1.0 : 0.0;
  }
  return 1.0;
}

ScalarType infer_scalar_dtype(PyObject* object) {
  if (PyBool_Check(object)) {
    return ScalarType::Bool;
  }
  if (PyLong_Check(object)) {
    return ScalarType::Int64;
  }
  if (PyFloat_Check(object)) {
    return ScalarType::Float32;
  }
  return ScalarType::Float32;
}

ScalarType merge_inferred_dtype(ScalarType left, ScalarType right) {
  if (left == ScalarType::Float32 || left == ScalarType::Float64 || right == ScalarType::Float32 ||
      right == ScalarType::Float64) {
    return ScalarType::Float32;
  }
  if (left == ScalarType::Int64 || left == ScalarType::Int32 || right == ScalarType::Int64 ||
      right == ScalarType::Int32) {
    return ScalarType::Int64;
  }
  return ScalarType::Bool;
}

void parse_nested_data(
    PyObject* object,
    std::vector<double>& values,
    std::vector<int64_t>& shape,
    ScalarType& inferred_dtype,
    int64_t depth) {
  if (!object_is_sequence(object)) {
    inferred_dtype = values.empty() ? infer_scalar_dtype(object) : merge_inferred_dtype(inferred_dtype, infer_scalar_dtype(object));
    values.push_back(scalar_from_py(object));
    return;
  }

  const Py_ssize_t length = PySequence_Size(object);
  if (length < 0) {
    throw std::invalid_argument("could not read sequence length");
  }
  if (static_cast<int64_t>(shape.size()) <= depth) {
    shape.push_back(static_cast<int64_t>(length));
  } else if (shape[static_cast<size_t>(depth)] != static_cast<int64_t>(length)) {
    throw std::invalid_argument("tensor data must be rectangular");
  }

  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::invalid_argument("could not read sequence item");
    }
    try {
      parse_nested_data(item, values, shape, inferred_dtype, depth + 1);
      Py_DECREF(item);
    } catch (...) {
      Py_DECREF(item);
      throw;
    }
  }
}

std::vector<int64_t> shape_from_object(PyObject* object) {
  if (PyLong_Check(object)) {
    return {PyLong_AsLongLong(object)};
  }
  if (PyTuple_Check(object)) {
    const Py_ssize_t length = PyTuple_GET_SIZE(object);
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(length));
    for (Py_ssize_t i = 0; i < length; ++i) {
      const int64_t value = PyLong_AsLongLong(PyTuple_GET_ITEM(object, i));
      if (PyErr_Occurred()) {
        throw std::invalid_argument("shape entries must be integers");
      }
      shape.push_back(value);
    }
    return shape;
  }
  if (PyList_Check(object)) {
    const Py_ssize_t length = PyList_GET_SIZE(object);
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(length));
    for (Py_ssize_t i = 0; i < length; ++i) {
      const int64_t value = PyLong_AsLongLong(PyList_GET_ITEM(object, i));
      if (PyErr_Occurred()) {
        throw std::invalid_argument("shape entries must be integers");
      }
      shape.push_back(value);
    }
    return shape;
  }
  if (!object_is_sequence(object)) {
    throw std::invalid_argument("shape must be an int or a sequence of ints");
  }
  const Py_ssize_t length = PySequence_Size(object);
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(length));
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::invalid_argument("could not read shape item");
    }
    const int64_t value = PyLong_AsLongLong(item);
    Py_DECREF(item);
    if (PyErr_Occurred()) {
      throw std::invalid_argument("shape entries must be integers");
    }
    shape.push_back(value);
  }
  return shape;
}

std::vector<double> double_vector_from_object(PyObject* object) {
  if (PyFloat_Check(object) || PyLong_Check(object)) {
    return {scalar_from_py(object)};
  }
  if (!object_is_sequence(object)) {
    throw std::invalid_argument("value must be a numeric scalar or a sequence of numeric scalars");
  }
  const Py_ssize_t length = PySequence_Size(object);
  std::vector<double> values;
  values.reserve(static_cast<size_t>(length));
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::invalid_argument("could not read sequence item");
    }
    try {
      values.push_back(scalar_from_py(item));
      Py_DECREF(item);
    } catch (...) {
      Py_DECREF(item);
      throw;
    }
  }
  return values;
}

std::vector<int64_t> shape_from_args(PyObject* args, Py_ssize_t start = 0) {
  const Py_ssize_t count = PyTuple_GET_SIZE(args) - start;
  if (count == 1) {
    return shape_from_object(PyTuple_GET_ITEM(args, start));
  }
  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(count));
  for (Py_ssize_t i = start; i < PyTuple_GET_SIZE(args); ++i) {
    const int64_t value = PyLong_AsLongLong(PyTuple_GET_ITEM(args, i));
    if (PyErr_Occurred()) {
      throw std::invalid_argument("shape entries must be integers");
    }
    shape.push_back(value);
  }
  return shape;
}

struct FactoryOptions {
  std::vector<int64_t> shape;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  PyObject* generator = Py_None;
  bool requires_grad = false;
};

RandomState& random_state_from_generator(PyObject* generator) {
  if (generator == nullptr || generator == Py_None) {
    return global_rng();
  }
  if (GeneratorType != nullptr && PyObject_TypeCheck(generator, reinterpret_cast<PyTypeObject*>(GeneratorType))) {
    auto* typed_generator = reinterpret_cast<PyGenerator*>(generator);
    return typed_generator->uses_global ? global_rng() : typed_generator->rng;
  }
  throw std::invalid_argument("generator must be a mtorch.Generator");
}

FactoryOptions factory_options_from_args(
    PyObject* args,
    PyObject* kwargs,
    const char* name,
    bool allow_random_options = false) {
  FactoryOptions options;
  PyObject* size_kw = nullptr;
  if (kwargs != nullptr) {
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t position = 0;
    while (PyDict_Next(kwargs, &position, &key, &value)) {
      const char* key_text = PyUnicode_AsUTF8(key);
      if (key_text == nullptr) {
        throw std::invalid_argument("factory keyword names must be strings");
      }
      if (std::strcmp(key_text, "size") == 0) {
        size_kw = value;
      } else if (std::strcmp(key_text, "dtype") == 0) {
        options.dtype = value;
      } else if (std::strcmp(key_text, "device") == 0) {
        options.device = value;
      } else if (std::strcmp(key_text, "requires_grad") == 0) {
        const int truth = PyObject_IsTrue(value);
        if (truth < 0) {
          throw std::invalid_argument("requires_grad must be truthy or falsy");
        }
        options.requires_grad = truth != 0;
      } else if (allow_random_options && std::strcmp(key_text, "generator") == 0) {
        if (value != Py_None) {
          (void)random_state_from_generator(value);
        }
        options.generator = value;
      } else if (allow_random_options &&
          (std::strcmp(key_text, "out") == 0 || std::strcmp(key_text, "layout") == 0)) {
        if (value != Py_None) {
          throw NotImplementedException(std::string(name) + " " + key_text + " is not implemented");
        }
      } else if (allow_random_options && std::strcmp(key_text, "pin_memory") == 0) {
        const int truth = PyObject_IsTrue(value);
        if (truth < 0) {
          throw std::invalid_argument("pin_memory must be truthy or falsy");
        }
        if (truth != 0) {
          throw NotImplementedException(std::string(name) + " pin_memory=True is not implemented");
        }
      } else {
        throw std::invalid_argument(std::string(name) + " got an unexpected keyword argument '" + key_text + "'");
      }
    }
  }

  const Py_ssize_t positional_count = PyTuple_GET_SIZE(args);
  if (size_kw != nullptr) {
    if (positional_count != 0) {
      throw std::invalid_argument(std::string(name) + " got both positional size and size keyword");
    }
    options.shape = shape_from_object(size_kw);
    return options;
  }
  if (positional_count == 0) {
    throw std::invalid_argument(std::string(name) + " missing required size");
  }
  options.shape = shape_from_args(args);
  return options;
}

std::vector<int64_t> dims_from_args(PyObject* args, Py_ssize_t start = 0) {
  return shape_from_args(args, start);
}

mtorch::TensorIndex make_select_index(int64_t value) {
  mtorch::TensorIndex index;
  index.kind = mtorch::TensorIndexKind::Select;
  index.index = value;
  return index;
}

mtorch::TensorIndex make_slice_index(int64_t start, int64_t length, int64_t step) {
  mtorch::TensorIndex index;
  index.kind = mtorch::TensorIndexKind::Slice;
  index.start = start;
  index.length = length;
  index.step = step;
  return index;
}

mtorch::TensorIndex make_new_axis_index() {
  mtorch::TensorIndex index;
  index.kind = mtorch::TensorIndexKind::NewAxis;
  return index;
}

PyObject* single_bool_mask_key(PyObject* key) {
  PyObject* candidate = key;
  if (PyTuple_Check(key)) {
    if (PyTuple_GET_SIZE(key) != 1) {
      return nullptr;
    }
    candidate = PyTuple_GET_ITEM(key, 0);
  }

  if (PyBool_Check(candidate)) {
    return candidate;
  }
  if (is_tensor(candidate) && tensor_ref(candidate)->dtype == ScalarType::Bool) {
    return candidate;
  }
  return nullptr;
}

TensorPtr bool_mask_from_key(PyObject* key, Device device) {
  PyObject* mask_key = single_bool_mask_key(key);
  if (mask_key == nullptr) {
    return nullptr;
  }
  if (PyBool_Check(mask_key)) {
    return mtorch::make_tensor({mask_key == Py_True ? 1.0 : 0.0}, {}, ScalarType::Bool, false, device);
  }
  return tensor_ref(mask_key);
}

TensorPtr bool_mask_from_candidate(PyObject* candidate) {
  if (is_tensor(candidate) && tensor_ref(candidate)->dtype == ScalarType::Bool) {
    return tensor_ref(candidate);
  }
  return nullptr;
}

void parse_int_index_data(
    PyObject* object,
    std::vector<double>& values,
    std::vector<int64_t>& shape,
    int64_t depth) {
  if (PyList_Check(object) || PyTuple_Check(object)) {
    const Py_ssize_t length = PySequence_Size(object);
    if (static_cast<int64_t>(shape.size()) <= depth) {
      shape.push_back(static_cast<int64_t>(length));
    } else if (shape[static_cast<size_t>(depth)] != static_cast<int64_t>(length)) {
      throw std::invalid_argument("index data must be rectangular");
    }
    for (Py_ssize_t i = 0; i < length; ++i) {
      PyObject* item = PySequence_GetItem(object, i);
      if (item == nullptr) {
        throw std::invalid_argument("could not read index sequence item");
      }
      try {
        parse_int_index_data(item, values, shape, depth + 1);
        Py_DECREF(item);
      } catch (...) {
        Py_DECREF(item);
        throw;
      }
    }
    return;
  }

  if (PyBool_Check(object) || !PyIndex_Check(object)) {
    throw TypeErrorException("integer index sequence entries must be integers");
  }
  PyObject* integer = PyNumber_Index(object);
  if (integer == nullptr) {
    throw TypeErrorException("integer index sequence entries must be integers");
  }
  const int64_t value = PyLong_AsLongLong(integer);
  Py_DECREF(integer);
  if (PyErr_Occurred()) {
    throw std::out_of_range("tensor index is out of range");
  }
  values.push_back(static_cast<double>(value));
}

TensorPtr int_tensor_from_candidate(PyObject* candidate, Device device) {
  if (is_tensor(candidate)) {
    auto tensor = tensor_ref(candidate);
    if (tensor->dtype != ScalarType::Int32 && tensor->dtype != ScalarType::Int64) {
      return nullptr;
    }
    if (tensor->is_scalar()) {
      return nullptr;
    }
    return tensor;
  }

  if (PyList_Check(candidate)) {
    std::vector<double> values;
    std::vector<int64_t> shape;
    parse_int_index_data(candidate, values, shape, 0);
    return mtorch::make_tensor(values, shape, ScalarType::Int64, false, device);
  }

  return nullptr;
}

TensorPtr int_tensor_from_key(PyObject* key, Device device) {
  PyObject* candidate = key;
  if (PyTuple_Check(key)) {
    if (PyTuple_GET_SIZE(key) != 1) {
      return nullptr;
    }
    candidate = PyTuple_GET_ITEM(key, 0);
  }
  return int_tensor_from_candidate(candidate, device);
}

std::vector<TensorPtr> int_tensor_tuple_from_key(PyObject* key, Device device) {
  std::vector<TensorPtr> indices;
  if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) < 2) {
    return indices;
  }
  const Py_ssize_t key_count = PyTuple_GET_SIZE(key);
  indices.reserve(static_cast<size_t>(key_count));
  for (Py_ssize_t i = 0; i < key_count; ++i) {
    auto index = int_tensor_from_candidate(PyTuple_GET_ITEM(key, i), device);
    if (!index) {
      indices.clear();
      return indices;
    }
    indices.push_back(index);
  }
  return indices;
}

bool is_basic_integer_select_candidate(PyObject* candidate) {
  if (PyBool_Check(candidate)) {
    return false;
  }
  if (PyIndex_Check(candidate)) {
    return true;
  }
  if (is_tensor(candidate)) {
    auto tensor = tensor_ref(candidate);
    return (tensor->dtype == ScalarType::Int32 || tensor->dtype == ScalarType::Int64) && tensor->is_scalar();
  }
  return false;
}

std::vector<mtorch::TensorIndex> parse_tensor_indices(PyObject* key, const Tensor& tensor) {
  const bool key_is_tuple = PyTuple_Check(key);
  const Py_ssize_t key_count = key_is_tuple ? PyTuple_GET_SIZE(key) : 1;
  Py_ssize_t ellipsis_count = 0;
  int64_t consumed_dims = 0;

  for (Py_ssize_t i = 0; i < key_count; ++i) {
    PyObject* item = key_is_tuple ? PyTuple_GET_ITEM(key, i) : key;
    if (item == Py_Ellipsis) {
      ++ellipsis_count;
      continue;
    }
    if (item == Py_None) {
      continue;
    }
    ++consumed_dims;
  }

  if (ellipsis_count > 1) {
    throw std::out_of_range("an index can only have a single ellipsis");
  }
  if (consumed_dims > tensor.dim()) {
    throw std::out_of_range("too many indices for tensor");
  }

  const int64_t ellipsis_dims = ellipsis_count == 0 ? 0 : tensor.dim() - consumed_dims;
  std::vector<mtorch::TensorIndex> indices;
  int64_t current_dim = 0;
  bool expanded_ellipsis = false;

  for (Py_ssize_t i = 0; i < key_count; ++i) {
    PyObject* item = key_is_tuple ? PyTuple_GET_ITEM(key, i) : key;

    if (item == Py_Ellipsis) {
      if (expanded_ellipsis) {
        throw std::out_of_range("an index can only have a single ellipsis");
      }
      for (int64_t dim = 0; dim < ellipsis_dims; ++dim) {
        indices.push_back(make_slice_index(0, tensor.sizes[static_cast<size_t>(current_dim)], 1));
        ++current_dim;
      }
      expanded_ellipsis = true;
      continue;
    }

    if (item == Py_None) {
      indices.push_back(make_new_axis_index());
      continue;
    }

    if (current_dim >= tensor.dim()) {
      throw std::out_of_range("too many indices for tensor");
    }

    if (PySlice_Check(item)) {
      Py_ssize_t start = 0;
      Py_ssize_t stop = 0;
      Py_ssize_t step = 0;
      if (PySlice_Unpack(item, &start, &stop, &step) < 0) {
        throw std::invalid_argument("invalid slice");
      }
      if (step <= 0) {
        throw std::invalid_argument("step must be greater than zero");
      }
      const Py_ssize_t length =
          PySlice_AdjustIndices(static_cast<Py_ssize_t>(tensor.sizes[static_cast<size_t>(current_dim)]), &start, &stop, step);
      indices.push_back(make_slice_index(start, length, step));
      ++current_dim;
      continue;
    }

    if (PyBool_Check(item)) {
      throw std::invalid_argument("boolean indexing is not implemented yet");
    }

    if (is_tensor(item)) {
      auto tensor_index = tensor_ref(item);
      if ((tensor_index->dtype == ScalarType::Int32 || tensor_index->dtype == ScalarType::Int64) &&
          tensor_index->is_scalar()) {
        indices.push_back(make_select_index(static_cast<int64_t>(tensor_index->value_at_linear(0))));
        ++current_dim;
        continue;
      }
      throw std::invalid_argument("mixed tensor advanced indexing is not implemented yet");
    }

    if (PyIndex_Check(item)) {
      PyObject* integer = PyNumber_Index(item);
      if (integer == nullptr) {
        throw std::invalid_argument("invalid tensor index");
      }
      const int64_t value = PyLong_AsLongLong(integer);
      Py_DECREF(integer);
      if (PyErr_Occurred()) {
        throw std::out_of_range("tensor index is out of range");
      }
      indices.push_back(make_select_index(value));
      ++current_dim;
      continue;
    }

    throw std::invalid_argument("unsupported tensor index type");
  }

  return indices;
}

struct MixedAdvancedKey {
  TensorPtr mask;
  TensorPtr int_indices;
  std::vector<mtorch::TensorIndex> tail_indices;
};

struct DimIntAdvancedKey {
  TensorPtr base;
  TensorPtr int_indices;
  int64_t dim = 0;
};

std::vector<mtorch::TensorIndex> parse_tail_indices(
    PyObject* key,
    Py_ssize_t start,
    const Tensor& tensor,
    int64_t prefix_rank) {
  if (prefix_rank < 0 || prefix_rank > tensor.dim()) {
    throw std::out_of_range("too many indices for tensor");
  }
  const Py_ssize_t key_count = PyTuple_GET_SIZE(key);
  const Py_ssize_t tail_count = key_count - start;
  if (tail_count <= 0) {
    return {};
  }

  Tensor tail_tensor(
      tensor.storage,
      std::vector<int64_t>(tensor.sizes.begin() + prefix_rank, tensor.sizes.end()),
      std::vector<int64_t>(tensor.strides.begin() + prefix_rank, tensor.strides.end()),
      tensor.offset,
      tensor.dtype,
      tensor.requires_grad);

  if (tail_count == 1) {
    return parse_tensor_indices(PyTuple_GET_ITEM(key, start), tail_tensor);
  }

  PyObject* tail_tuple = PyTuple_New(tail_count);
  if (tail_tuple == nullptr) {
    throw std::runtime_error("could not create tail index tuple");
  }
  for (Py_ssize_t i = 0; i < tail_count; ++i) {
    PyObject* item = PyTuple_GET_ITEM(key, start + i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tail_tuple, i, item);
  }

  try {
    auto result = parse_tensor_indices(tail_tuple, tail_tensor);
    Py_DECREF(tail_tuple);
    return result;
  } catch (...) {
    Py_DECREF(tail_tuple);
    throw;
  }
}

bool parse_mixed_advanced_key(PyObject* key, const Tensor& tensor, MixedAdvancedKey& parsed) {
  if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) < 2) {
    return false;
  }

  PyObject* first = PyTuple_GET_ITEM(key, 0);
  if (auto mask = bool_mask_from_candidate(first)) {
    parsed.mask = mask;
    parsed.tail_indices = parse_tail_indices(key, 1, tensor, mask->dim());
    return true;
  }
  if (auto indices = int_tensor_from_candidate(first, tensor.device)) {
    parsed.int_indices = indices;
    parsed.tail_indices = parse_tail_indices(key, 1, tensor, 1);
    return true;
  }
  return false;
}

bool parse_dim_int_advanced_key(PyObject* key, const TensorPtr& tensor, DimIntAdvancedKey& parsed) {
  if (!PyTuple_Check(key) || PyTuple_GET_SIZE(key) < 2) {
    return false;
  }

  const Py_ssize_t key_count = PyTuple_GET_SIZE(key);
  Py_ssize_t advanced_at = -1;
  TensorPtr advanced_indices;
  for (Py_ssize_t i = 0; i < key_count; ++i) {
    if (auto indices = int_tensor_from_candidate(PyTuple_GET_ITEM(key, i), tensor->device)) {
      if (advanced_at != -1) {
        return false;
      }
      advanced_at = i;
      advanced_indices = indices;
    }
  }

  if (advanced_at <= 0 || advanced_at != key_count - 1) {
    return false;
  }

  int64_t dim_in_base = 0;
  for (Py_ssize_t i = 0; i < advanced_at; ++i) {
    PyObject* item = PyTuple_GET_ITEM(key, i);
    if (PySlice_Check(item)) {
      ++dim_in_base;
      continue;
    }
    if (is_basic_integer_select_candidate(item)) {
      continue;
    }
    return false;
  }

  PyObject* prefix_tuple = PyTuple_New(advanced_at);
  if (prefix_tuple == nullptr) {
    throw std::runtime_error("could not create prefix index tuple");
  }
  for (Py_ssize_t i = 0; i < advanced_at; ++i) {
    PyObject* item = PyTuple_GET_ITEM(key, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(prefix_tuple, i, item);
  }

  try {
    auto prefix_indices = parse_tensor_indices(prefix_tuple, *tensor);
    Py_DECREF(prefix_tuple);
    parsed.base = mtorch::index(tensor, prefix_indices);
    parsed.int_indices = advanced_indices;
    parsed.dim = dim_in_base;
    return true;
  } catch (...) {
    Py_DECREF(prefix_tuple);
    throw;
  }
}

TensorPtr full_row_slice_int_columns_key(PyObject* key, const TensorPtr& tensor) {
  if (tensor->dim() != 2 || !PyTuple_Check(key) || PyTuple_GET_SIZE(key) != 2) {
    return nullptr;
  }
  PyObject* rows = PyTuple_GET_ITEM(key, 0);
  if (!PySlice_Check(rows)) {
    return nullptr;
  }

  Py_ssize_t start = 0;
  Py_ssize_t stop = 0;
  Py_ssize_t step = 0;
  if (PySlice_Unpack(rows, &start, &stop, &step) < 0) {
    throw std::invalid_argument("invalid slice");
  }
  const Py_ssize_t length =
      PySlice_AdjustIndices(static_cast<Py_ssize_t>(tensor->sizes[0]), &start, &stop, step);
  if (start != 0 || step != 1 || length != static_cast<Py_ssize_t>(tensor->sizes[0])) {
    return nullptr;
  }
  return int_tensor_from_candidate(PyTuple_GET_ITEM(key, 1), tensor->device);
}

PyObject* scalar_to_py(double value, ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Bool:
      return PyBool_FromLong(value != 0.0);
    case ScalarType::Int32:
    case ScalarType::Int64:
      return PyLong_FromLongLong(static_cast<long long>(value));
    case ScalarType::Float16:
    case ScalarType::Float32:
    case ScalarType::Float64:
      return PyFloat_FromDouble(value);
  }
  return PyFloat_FromDouble(value);
}

PyObject* tensor_to_nested_list(const Tensor& tensor, size_t depth, std::vector<int64_t>& index) {
  if (depth == tensor.sizes.size()) {
    return scalar_to_py(tensor.value_at_index(index), tensor.dtype);
  }

  PyObject* list = PyList_New(tensor.sizes[depth]);
  if (list == nullptr) {
    return nullptr;
  }
  for (int64_t i = 0; i < tensor.sizes[depth]; ++i) {
    index.push_back(i);
    PyObject* item = tensor_to_nested_list(tensor, depth + 1, index);
    index.pop_back();
    if (item == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SET_ITEM(list, i, item);
  }
  return list;
}

bool pyobject_to_scalar(PyObject* object, double& value, ScalarType* dtype = nullptr) {
  if (PyFloat_Check(object) || PyLong_Check(object) || PyBool_Check(object)) {
    value = scalar_from_py(object);
    if (dtype != nullptr) {
      *dtype = infer_scalar_dtype(object);
    }
    return true;
  }
  return false;
}

struct ToRequest {
  ScalarType dtype;
  Device device;
  bool copy = false;
  std::optional<MemoryFormat> memory_format;
};

void apply_to_argument(PyObject* object, ToRequest& request) {
  if (object == nullptr || object == Py_None) {
    return;
  }
  if (is_tensor(object)) {
    const auto& target = tensor_ref(object);
    request.dtype = target->dtype;
    request.device = target->device;
    return;
  }

  const std::string text = object_text_lower(object);
  if (text_looks_like_device(text)) {
    request.device = device_from_py(object, request.device);
    return;
  }
  request.dtype = dtype_from_py(object, request.dtype);
}

ToRequest parse_to_request(const Tensor& source, PyObject* args, PyObject* kwargs) {
  ToRequest request{source.dtype, source.device, false};
  const Py_ssize_t arg_count = PyTuple_GET_SIZE(args);
  if (arg_count > 4) {
    throw std::invalid_argument("to() accepts at most 4 positional arguments");
  }
  for (Py_ssize_t i = 0; i < arg_count; ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (PyBool_Check(item)) {
      if (i >= 2) {
        request.copy = PyObject_IsTrue(item) != 0;
      }
      continue;
    }
    apply_to_argument(item, request);
  }

  if (kwargs == nullptr) {
    return request;
  }
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    const std::string name = object_text_lower(key);
    if (name == "dtype") {
      request.dtype = dtype_from_py(value, request.dtype);
    } else if (name == "device") {
      request.device = device_from_py(value, request.device);
    } else if (name == "copy") {
      request.copy = PyObject_IsTrue(value) != 0;
    } else if (name == "memory_format") {
      request.memory_format = optional_memory_format_from_py(value);
    } else if (name == "non_blocking") {
      continue;
    } else {
      throw TypeErrorException("to() got an unexpected keyword argument");
    }
  }
  return request;
}

PyObject* binary_dispatch(PyObject* left, PyObject* right, const std::string& op) {
  try {
    if (is_tensor(left) && is_tensor(right)) {
      return wrap_tensor(mtorch::binary_tensor_tensor(tensor_ref(left), tensor_ref(right), op));
    }
    double scalar = 0.0;
    ScalarType scalar_dtype = ScalarType::Float32;
    if (is_tensor(left) && pyobject_to_scalar(right, scalar, &scalar_dtype)) {
      return wrap_tensor(mtorch::binary_tensor_scalar(tensor_ref(left), scalar, scalar_dtype, op));
    }
    if (pyobject_to_scalar(left, scalar, &scalar_dtype) && is_tensor(right)) {
      return wrap_tensor(mtorch::binary_scalar_tensor(scalar, scalar_dtype, tensor_ref(right), op));
    }
    Py_RETURN_NOTIMPLEMENTED;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* unary_dispatch(PyObject* object, const std::string& op) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unary(tensor_ref(object), op));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* parse_tensor_call(PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"data", "dtype", "requires_grad", "device", nullptr};
  PyObject* data = nullptr;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|OpO:tensor", const_cast<char**>(keywords), &data, &dtype, &requires_grad, &device)) {
    return nullptr;
  }
  if (is_tensor(data)) {
    try {
      auto source = tensor_ref(data);
      auto target_dtype = dtype_from_py(dtype, source->dtype);
      auto target_device = device_from_py(device, source->device);
      auto copy = mtorch::make_tensor(source->contiguous_values(), source->sizes, target_dtype, false, target_device);
      copy->requires_grad = requires_grad != 0;
      return wrap_tensor(copy);
    } catch (...) {
      translate_exception();
      return nullptr;
    }
  }

  try {
    std::vector<double> values;
    std::vector<int64_t> shape;
    ScalarType inferred_dtype = ScalarType::Float32;
    parse_nested_data(data, values, shape, inferred_dtype, 0);
    return wrap_tensor(
        mtorch::make_tensor(values, shape, dtype_from_py(dtype, inferred_dtype), requires_grad != 0, device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* make_size_tuple(const std::vector<int64_t>& values) {
  PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
  if (tuple == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), PyLong_FromLongLong(values[i]));
  }
  return tuple;
}

PyObject* py_tensor(PyObject*, PyObject* args, PyObject* kwargs) {
  return parse_tensor_call(args, kwargs);
}

PyObject* py_as_tensor(PyObject*, PyObject* args, PyObject* kwargs) {
  return parse_tensor_call(args, kwargs);
}

PyObject* py_zeros(PyObject*, PyObject* args, PyObject* kwargs) {
  try {
    const auto options = factory_options_from_args(args, kwargs, "zeros");
    auto result = mtorch::zeros(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_ones(PyObject*, PyObject* args, PyObject* kwargs) {
  try {
    const auto options = factory_options_from_args(args, kwargs, "ones");
    auto result = mtorch::ones(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_empty(PyObject*, PyObject* args, PyObject* kwargs) {
  try {
    const auto options = factory_options_from_args(args, kwargs, "empty");
    auto result = mtorch::zeros(options.shape, dtype_from_py(options.dtype), device_from_py(options.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_empty_strided(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"size", "stride", "dtype", "layout", "device", "requires_grad", "pin_memory", nullptr};
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  PyObject* pin_memory = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOpO:empty_strided",
          const_cast<char**>(keywords),
          &size,
          &stride,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &pin_memory)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::empty_strided(
        shape_from_object(size),
        shape_from_object(stride),
        dtype_from_py(dtype),
        requires_grad != 0,
        device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_full(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"size", "fill_value", "dtype", "device", nullptr};
  PyObject* size = nullptr;
  PyObject* fill_value = nullptr;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OO|OO:full", const_cast<char**>(keywords), &size, &fill_value, &dtype, &device)) {
    return nullptr;
  }
  try {
    return wrap_tensor(
        mtorch::full(shape_from_object(size), scalar_from_py(fill_value), dtype_from_py(dtype), device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_empty_like(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dtype", "layout", "device", "requires_grad", "memory_format", nullptr};
  PyObject* input = nullptr;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  PyObject* memory_format = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOOpO:empty_like",
          const_cast<char**>(keywords),
          &input,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &memory_format)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "empty_like expected Tensor");
    return nullptr;
  }
  try {
    const auto& source = tensor_ref(input);
    return wrap_tensor(make_like_with_memory_format(
        source,
        dtype_from_py(dtype, source->dtype),
        device_from_py(device, source->device),
        requires_grad != 0,
        optional_memory_format_from_py(memory_format)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_zeros_like(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dtype", "layout", "device", "requires_grad", "memory_format", nullptr};
  PyObject* input = nullptr;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  PyObject* memory_format = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOOpO:zeros_like",
          const_cast<char**>(keywords),
          &input,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &memory_format)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "zeros_like expected Tensor");
    return nullptr;
  }
  try {
    const auto& source = tensor_ref(input);
    auto result = make_like_with_memory_format(
        source,
        dtype_from_py(dtype, source->dtype),
        device_from_py(device, source->device),
        requires_grad != 0,
        optional_memory_format_from_py(memory_format));
    result->fill_inplace(0.0);
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_ones_like(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dtype", "layout", "device", "requires_grad", "memory_format", nullptr};
  PyObject* input = nullptr;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  PyObject* memory_format = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOOpO:ones_like",
          const_cast<char**>(keywords),
          &input,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &memory_format)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "ones_like expected Tensor");
    return nullptr;
  }
  try {
    const auto& source = tensor_ref(input);
    auto result = make_like_with_memory_format(
        source,
        dtype_from_py(dtype, source->dtype),
        device_from_py(device, source->device),
        requires_grad != 0,
        optional_memory_format_from_py(memory_format));
    result->fill_inplace(1.0);
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_full_like(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "fill_value", "dtype", "layout", "device", "requires_grad", "memory_format", nullptr};
  PyObject* input = nullptr;
  PyObject* fill_value = nullptr;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  PyObject* memory_format = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOpO:full_like",
          const_cast<char**>(keywords),
          &input,
          &fill_value,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &memory_format)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "full_like expected Tensor");
    return nullptr;
  }
  try {
    const auto& source = tensor_ref(input);
    auto result = make_like_with_memory_format(
        source,
        dtype_from_py(dtype, source->dtype),
        device_from_py(device, source->device),
        requires_grad != 0,
        optional_memory_format_from_py(memory_format));
    result->fill_inplace(scalar_from_py(fill_value));
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_arange(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"start", "end", "step", "dtype", "device", nullptr};
  PyObject* start = nullptr;
  PyObject* end = nullptr;
  PyObject* step = nullptr;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|OOOO:arange", const_cast<char**>(keywords), &start, &end, &step, &dtype, &device)) {
    return nullptr;
  }
  try {
    double start_value = 0.0;
    double end_value = 0.0;
    if (end == nullptr || end == Py_None) {
      end_value = scalar_from_py(start);
    } else {
      start_value = scalar_from_py(start);
      end_value = scalar_from_py(end);
    }
    const double step_value = (step == nullptr || step == Py_None) ? 1.0 : scalar_from_py(step);
    return wrap_tensor(
        mtorch::arange(start_value, end_value, step_value, default_arange_dtype(dtype, start, end, step), device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_linspace(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"start", "end", "steps", "dtype", "device", nullptr};
  double start = 0.0;
  double end = 0.0;
  long long steps = 100;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "dd|LOO:linspace", const_cast<char**>(keywords), &start, &end, &steps, &dtype, &device)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::linspace(start, end, steps, dtype_from_py(dtype), device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_eye(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"n", "dtype", "device", nullptr};
  long long n = 0;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "L|OO:eye", const_cast<char**>(keywords), &n, &dtype, &device)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::eye(n, dtype_from_py(dtype), device_from_py(device)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_randint(PyObject*, PyObject* args, PyObject* kwargs) {
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  PyObject* generator = Py_None;
  PyObject* size_kw = nullptr;
  bool requires_grad = false;
  if (kwargs != nullptr) {
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t position = 0;
    while (PyDict_Next(kwargs, &position, &key, &value)) {
      const char* key_text = PyUnicode_AsUTF8(key);
      if (key_text == nullptr) {
        PyErr_SetString(PyExc_TypeError, "randint keyword names must be strings");
        return nullptr;
      }
      if (std::strcmp(key_text, "size") == 0) {
        size_kw = value;
      } else if (std::strcmp(key_text, "dtype") == 0) {
        dtype = value;
      } else if (std::strcmp(key_text, "device") == 0) {
        device = value;
      } else if (std::strcmp(key_text, "requires_grad") == 0) {
        const int truth = PyObject_IsTrue(value);
        if (truth < 0) {
          return nullptr;
        }
        requires_grad = truth != 0;
      } else if (std::strcmp(key_text, "generator") == 0) {
        generator = value;
        try {
          (void)random_state_from_generator(generator);
        } catch (...) {
          translate_exception();
          return nullptr;
        }
      } else if (
          std::strcmp(key_text, "out") == 0 ||
          std::strcmp(key_text, "layout") == 0 || std::strcmp(key_text, "pin_memory") == 0) {
        PyErr_SetString(PyExc_NotImplementedError, "randint optional out/layout/pin_memory arguments are not implemented");
        return nullptr;
      } else {
        PyErr_Format(PyExc_TypeError, "randint got an unexpected keyword argument '%s'", key_text);
        return nullptr;
      }
    }
  }

  try {
    const Py_ssize_t count = PyTuple_GET_SIZE(args);
    int64_t low = 0;
    int64_t high = 0;
    PyObject* size = size_kw;
    if (size_kw != nullptr) {
      if (count == 1) {
        high = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 0));
      } else if (count == 2) {
        low = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 0));
        high = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 1));
      } else {
        throw std::invalid_argument("randint expected high or low, high with size keyword");
      }
    } else if (count == 2) {
      high = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 0));
      size = PyTuple_GET_ITEM(args, 1);
    } else if (count == 3) {
      low = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 0));
      high = PyLong_AsLongLong(PyTuple_GET_ITEM(args, 1));
      size = PyTuple_GET_ITEM(args, 2);
    } else {
      throw std::invalid_argument("randint expected (high, size) or (low, high, size)");
    }
    if (PyErr_Occurred()) {
      return nullptr;
    }
    auto result_dtype = dtype_from_py(dtype, ScalarType::Int64);
    if (requires_grad && !is_floating_scalar_type(result_dtype)) {
      throw std::invalid_argument("only floating point randint results can require gradients");
    }
    auto result = mtorch::zeros(shape_from_object(size), result_dtype, device_from_py(device));
    result->requires_grad = requires_grad;
    fill_randint_result(result, low, high, random_state_from_generator(generator));
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_rand(PyObject*, PyObject* args, PyObject* kwargs) {
  try {
    const auto options = factory_options_from_args(args, kwargs, "rand", true);
    const auto dtype = dtype_from_py(options.dtype);
    if (!is_floating_scalar_type(dtype)) {
      PyErr_SetString(PyExc_NotImplementedError, "rand is only implemented for floating point dtypes");
      return nullptr;
    }
    auto result = mtorch::zeros(options.shape, dtype, device_from_py(options.device));
    result->requires_grad = options.requires_grad;
    uniform_inplace(*result, 0.0, 1.0, random_state_from_generator(options.generator));
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_randn(PyObject*, PyObject* args, PyObject* kwargs) {
  try {
    const auto options = factory_options_from_args(args, kwargs, "randn", true);
    const auto dtype = dtype_from_py(options.dtype);
    if (dtype != ScalarType::Float16 && dtype != ScalarType::Float32 && dtype != ScalarType::Float64) {
      PyErr_SetString(PyExc_NotImplementedError, "randn is only implemented for floating point dtypes");
      return nullptr;
    }
    auto result = mtorch::zeros(options.shape, dtype, device_from_py(options.device));
    result->requires_grad = options.requires_grad;
    fill_randn_result(result, random_state_from_generator(options.generator));
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_randperm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "n", "generator", "out", "dtype", "layout", "device", "requires_grad", "pin_memory", nullptr};
  long long n = 0;
  PyObject* generator = Py_None;
  PyObject* out = Py_None;
  PyObject* dtype = Py_None;
  PyObject* layout = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  int pin_memory = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "L|OOOOOpp:randperm",
          const_cast<char**>(keywords),
          &n,
          &generator,
          &out,
          &dtype,
          &layout,
          &device,
          &requires_grad,
          &pin_memory)) {
    return nullptr;
  }
  try {
    if (n < 0) {
      throw std::invalid_argument("randperm expects n to be non-negative");
    }
    if (out != Py_None) {
      throw NotImplementedException("randperm out is not implemented");
    }
    if (layout != Py_None) {
      throw NotImplementedException("randperm layout is not implemented");
    }
    if (pin_memory != 0) {
      throw NotImplementedException("randperm pin_memory=True is not implemented");
    }
    const auto result_dtype = dtype_from_py(dtype, ScalarType::Int64);
    if (requires_grad != 0 && !is_floating_scalar_type(result_dtype)) {
      throw std::invalid_argument("only floating point randperm results can require gradients");
    }
    auto result = mtorch::zeros({n}, result_dtype, device_from_py(device));
    result->requires_grad = requires_grad != 0;
    for (int64_t i = 0; i < n; ++i) {
      result->set_at_linear(i, static_cast<double>(i));
    }
    auto& rng = random_state_from_generator(generator);
    for (int64_t i = n - 1; i > 0; --i) {
      const int64_t j = static_cast<int64_t>(next_random_u64(rng) % static_cast<uint64_t>(i + 1));
      const double current = result->value_at_linear(i);
      result->set_at_linear(i, result->value_at_linear(j));
      result->set_at_linear(j, current);
    }
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_trunc_normal_inplace(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"tensor", "mean", "std", "a", "b", "generator", nullptr};
  PyObject* tensor = nullptr;
  double mean = 0.0;
  double std = 1.0;
  double a = -2.0;
  double b = 2.0;
  PyObject* generator = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|ddddO:trunc_normal_", const_cast<char**>(keywords), &tensor, &mean, &std, &a, &b, &generator)) {
    return nullptr;
  }
  if (!is_tensor(tensor)) {
    PyErr_SetString(PyExc_TypeError, "trunc_normal_ expected Tensor");
    return nullptr;
  }
  try {
    trunc_normal_inplace(*tensor_ref(tensor), mean, std, a, b, random_state_from_generator(generator));
    Py_INCREF(tensor);
    return tensor;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_manual_seed(PyObject*, PyObject* args) {
  long long seed = 0;
  if (!PyArg_ParseTuple(args, "L:manual_seed", &seed)) {
    return nullptr;
  }
  const uint64_t normalized_seed = static_cast<uint64_t>(seed);
  seed_random_state(global_rng(), normalized_seed);
  global_initial_seed() = normalized_seed;
  PyObject* generator = PyObject_CallObject(GeneratorType, nullptr);
  if (generator == nullptr) {
    return nullptr;
  }
  auto* typed_generator = reinterpret_cast<PyGenerator*>(generator);
  typed_generator->uses_global = true;
  typed_generator->initial_seed = normalized_seed;
  return generator;
}

PyObject* py_initial_seed(PyObject*, PyObject*) {
  return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(global_initial_seed()));
}

PyObject* Generator_new(PyTypeObject* type, PyObject*, PyObject*) {
  auto* self = reinterpret_cast<PyGenerator*>(type->tp_alloc(type, 0));
  if (self == nullptr) {
    return nullptr;
  }
  seed_random_state(self->rng, 0);
  self->uses_global = false;
  self->initial_seed = 0;
  return reinterpret_cast<PyObject*>(self);
}

int Generator_init(PyGenerator* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"device", nullptr};
  PyObject* device = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:Generator", const_cast<char**>(keywords), &device)) {
    return -1;
  }
  try {
    (void)device_from_py(device);
    if (!self->uses_global) {
      seed_random_state(self->rng, 0);
      self->initial_seed = 0;
    }
    return 0;
  } catch (...) {
    translate_exception();
    return -1;
  }
}

PyObject* Generator_manual_seed(PyGenerator* self, PyObject* args) {
  long long seed = 0;
  if (!PyArg_ParseTuple(args, "L:manual_seed", &seed)) {
    return nullptr;
  }
  const uint64_t normalized_seed = static_cast<uint64_t>(seed);
  seed_random_state(random_state_from_generator(reinterpret_cast<PyObject*>(self)), normalized_seed);
  self->initial_seed = normalized_seed;
  if (self->uses_global) {
    global_initial_seed() = normalized_seed;
  }
  Py_INCREF(reinterpret_cast<PyObject*>(self));
  return reinterpret_cast<PyObject*>(self);
}

PyObject* Generator_initial_seed(PyGenerator* self, PyObject*) {
  return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(self->initial_seed));
}

PyObject* Generator_repr(PyGenerator* self) {
  return PyUnicode_FromString(self->uses_global ? "<mtorch.Generator device='cpu' global=True>"
                                                : "<mtorch.Generator device='cpu'>");
}

PyObject* py_multinomial(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "num_samples", "replacement", "generator", "out", nullptr};
  PyObject* input = nullptr;
  long long num_samples = 0;
  int replacement = 0;
  PyObject* generator = Py_None;
  PyObject* out = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OL|pOO:multinomial",
          const_cast<char**>(keywords),
          &input,
          &num_samples,
          &replacement,
          &generator,
          &out)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "multinomial expected input Tensor");
    return nullptr;
  }
  if (out != Py_None) {
    PyErr_SetString(PyExc_NotImplementedError, "multinomial out is not implemented");
    return nullptr;
  }
  try {
    return wrap_tensor(multinomial_tensor(tensor_ref(input), num_samples, replacement != 0, random_state_from_generator(generator)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_bernoulli(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "generator", "out", nullptr};
  PyObject* input = nullptr;
  PyObject* generator = Py_None;
  PyObject* out = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OO:bernoulli",
          const_cast<char**>(keywords),
          &input,
          &generator,
          &out)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "bernoulli expected input Tensor");
    return nullptr;
  }
  if (out != Py_None && !is_tensor(out)) {
    PyErr_SetString(PyExc_TypeError, "bernoulli out must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    if (out != Py_None && input_tensor->requires_grad) {
      throw std::runtime_error(
          "bernoulli(): functions with out=... arguments don't support automatic differentiation");
    }
    auto result = bernoulli_tensor(input_tensor, random_state_from_generator(generator));
    if (out != Py_None) {
      tensor_ref(out)->copy_from(*result);
      Py_INCREF(out);
      return out;
    }
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unary(PyObject*, PyObject* args, const std::string& op) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O", &input)) {
    return nullptr;
  }
  return unary_dispatch(input, op);
}

PyObject* py_neg(PyObject* self, PyObject* args) {
  return py_unary(self, args, "neg");
}

PyObject* py_abs(PyObject* self, PyObject* args) {
  return py_unary(self, args, "abs");
}

PyObject* py_exp(PyObject* self, PyObject* args) {
  return py_unary(self, args, "exp");
}

PyObject* py_expm1(PyObject* self, PyObject* args) {
  return py_unary(self, args, "expm1");
}

PyObject* py_log(PyObject* self, PyObject* args) {
  return py_unary(self, args, "log");
}

PyObject* py_log1p(PyObject* self, PyObject* args) {
  return py_unary(self, args, "log1p");
}

PyObject* py_log2(PyObject* self, PyObject* args) {
  return py_unary(self, args, "log2");
}

PyObject* py_log10(PyObject* self, PyObject* args) {
  return py_unary(self, args, "log10");
}

PyObject* py_sqrt(PyObject* self, PyObject* args) {
  return py_unary(self, args, "sqrt");
}

PyObject* py_rsqrt(PyObject* self, PyObject* args) {
  return py_unary(self, args, "rsqrt");
}

PyObject* py_reciprocal(PyObject* self, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:reciprocal", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::reciprocal(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_sign(PyObject* self, PyObject* args) {
  return py_unary(self, args, "sign");
}

PyObject* py_floor(PyObject* self, PyObject* args) {
  return py_unary(self, args, "floor");
}

PyObject* py_ceil(PyObject* self, PyObject* args) {
  return py_unary(self, args, "ceil");
}

PyObject* py_trunc(PyObject* self, PyObject* args) {
  return py_unary(self, args, "trunc");
}

PyObject* py_round(PyObject* self, PyObject* args) {
  return py_unary(self, args, "round");
}

PyObject* py_sin(PyObject* self, PyObject* args) {
  return py_unary(self, args, "sin");
}

PyObject* py_cos(PyObject* self, PyObject* args) {
  return py_unary(self, args, "cos");
}

PyObject* py_tan(PyObject* self, PyObject* args) {
  return py_unary(self, args, "tan");
}

PyObject* py_sinh(PyObject* self, PyObject* args) {
  return py_unary(self, args, "sinh");
}

PyObject* py_cosh(PyObject* self, PyObject* args) {
  return py_unary(self, args, "cosh");
}

PyObject* py_tanh(PyObject* self, PyObject* args) {
  return py_unary(self, args, "tanh");
}

PyObject* py_asin(PyObject* self, PyObject* args) {
  return py_unary(self, args, "asin");
}

PyObject* py_acos(PyObject* self, PyObject* args) {
  return py_unary(self, args, "acos");
}

PyObject* py_atan(PyObject* self, PyObject* args) {
  return py_unary(self, args, "atan");
}

PyObject* py_sigmoid(PyObject* self, PyObject* args) {
  return py_unary(self, args, "sigmoid");
}

PyObject* py_erf(PyObject* self, PyObject* args) {
  return py_unary(self, args, "erf");
}

PyObject* py_erfc(PyObject* self, PyObject* args) {
  return py_unary(self, args, "erfc");
}

PyObject* py_deg2rad(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:deg2rad", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "deg2rad expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::deg2rad(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_rad2deg(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:rad2deg", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "rad2deg expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::rad2deg(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_frac(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:frac", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "frac expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::frac(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unary_predicate(PyObject*, PyObject* args, const std::string& op) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "predicate expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unary_predicate(tensor_ref(input), op));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_isnan(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "isnan");
}

PyObject* py_isinf(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "isinf");
}

PyObject* py_isfinite(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "isfinite");
}

PyObject* py_signbit(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "signbit");
}

PyObject* py_isposinf(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "isposinf");
}

PyObject* py_isneginf(PyObject* self, PyObject* args) {
  return py_unary_predicate(self, args, "isneginf");
}

PyObject* py_logical_not(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:logical_not", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "logical_not expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::logical_not(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_bitwise_not(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:bitwise_not", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "bitwise_not expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::bitwise_not(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_nan_to_num(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "nan", "posinf", "neginf", nullptr};
  PyObject* input = nullptr;
  PyObject* nan_obj = nullptr;
  PyObject* posinf_obj = Py_None;
  PyObject* neginf_obj = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOO:nan_to_num",
          const_cast<char**>(keywords),
          &input,
          &nan_obj,
          &posinf_obj,
          &neginf_obj)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "nan_to_num expected Tensor");
    return nullptr;
  }
  try {
    const double nan = nan_obj == nullptr ? 0.0 : scalar_from_py(nan_obj);
    return wrap_tensor(
        mtorch::nan_to_num(tensor_ref(input), nan, optional_scalar_from_py(posinf_obj), optional_scalar_from_py(neginf_obj)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_square(PyObject* self, PyObject* args) {
  return py_unary(self, args, "square");
}

PyObject* py_gelu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "approximate", nullptr};
  PyObject* input = nullptr;
  const char* approximate = "none";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s:gelu", const_cast<char**>(keywords), &input, &approximate)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "gelu expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::gelu(tensor_ref(input), approximate));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_clamp(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "min", "max", nullptr};
  PyObject* input = nullptr;
  PyObject* min = Py_None;
  PyObject* max = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO:clamp", const_cast<char**>(keywords), &input, &min, &max)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "clamp expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::clamp(tensor_ref(input), optional_scalar_from_py(min), optional_scalar_from_py(max)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_clip(PyObject* self, PyObject* args, PyObject* kwargs) {
  return py_clamp(self, args, kwargs);
}

PyObject* py_clamp_min(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "min", nullptr};
  PyObject* input = nullptr;
  PyObject* min = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:clamp_min", const_cast<char**>(keywords), &input, &min)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "clamp_min expected Tensor");
    return nullptr;
  }
  try {
    if (is_tensor(min)) {
      return wrap_tensor(mtorch::binary_tensor_tensor(tensor_ref(input), tensor_ref(min), "maximum"));
    }
    return wrap_tensor(mtorch::clamp_min(tensor_ref(input), scalar_from_py(min)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_clamp_max(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "max", nullptr};
  PyObject* input = nullptr;
  PyObject* max = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:clamp_max", const_cast<char**>(keywords), &input, &max)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "clamp_max expected Tensor");
    return nullptr;
  }
  try {
    if (is_tensor(max)) {
      return wrap_tensor(mtorch::binary_tensor_tensor(tensor_ref(input), tensor_ref(max), "minimum"));
    }
    return wrap_tensor(mtorch::clamp_max(tensor_ref(input), scalar_from_py(max)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_softmax(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = nullptr;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:softmax", const_cast<char**>(keywords), &input, &dim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "softmax expected Tensor");
    return nullptr;
  }
  const long long dim_value = PyLong_AsLongLong(dim);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    return wrap_tensor(mtorch::softmax(tensor, dim_value, dtype_from_py(dtype, tensor->dtype)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_log_softmax(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = nullptr;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:log_softmax", const_cast<char**>(keywords), &input, &dim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "log_softmax expected Tensor");
    return nullptr;
  }
  const long long dim_value = PyLong_AsLongLong(dim);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    return wrap_tensor(mtorch::log_softmax(tensor, dim_value, dtype_from_py(dtype, tensor->dtype)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

double norm_order_from_py(PyObject* object) {
  if (object == Py_None) {
    return 2.0;
  }
  if (PyUnicode_Check(object)) {
    PyObject* text_object = PyObject_Str(object);
    if (text_object == nullptr) {
      throw std::invalid_argument("could not parse norm order");
    }
    const char* text = PyUnicode_AsUTF8(text_object);
    if (text == nullptr) {
      Py_DECREF(text_object);
      throw std::invalid_argument("could not parse norm order");
    }
    const std::string value = lowercase_ascii(text);
    Py_DECREF(text_object);
    if (value == "fro") {
      return 2.0;
    }
    throw NotImplementedException("norm p='nuc' is not implemented yet");
  }
  return scalar_from_py(object);
}

std::vector<int64_t> normalized_dims_for_tensor(const TensorPtr& input, const std::vector<int64_t>& dims, const char* name) {
  std::vector<int64_t> normalized;
  normalized.reserve(dims.size());
  for (int64_t dim : dims) {
    if (dim < 0) {
      dim += input->dim();
    }
    if (dim < 0 || dim >= input->dim()) {
      throw std::out_of_range(std::string(name) + " dimension out of range");
    }
    if (std::find(normalized.begin(), normalized.end(), dim) != normalized.end()) {
      throw std::invalid_argument(std::string(name) + " dim contains duplicate values");
    }
    normalized.push_back(dim);
  }
  std::sort(normalized.begin(), normalized.end(), std::greater<int64_t>());
  return normalized;
}

TensorPtr reduce_sum_dims_for_norm(
    const TensorPtr& input,
    const std::optional<std::vector<int64_t>>& dims,
    bool keepdim,
    ScalarType dtype) {
  if (!dims.has_value()) {
    return mtorch::reduce_sum(input, dtype);
  }
  auto result = input;
  for (int64_t dim : normalized_dims_for_tensor(input, *dims, "norm")) {
    result = mtorch::reduce_sum_dim(result, dim, keepdim, dtype);
  }
  return result;
}

TensorPtr norm_zero_tensor(
    const TensorPtr& input,
    const std::optional<std::vector<int64_t>>& dims,
    bool keepdim,
    ScalarType dtype) {
  if (!dims.has_value()) {
    int64_t count = 0;
    if (input->dtype == ScalarType::Float32 && input->is_contiguous()) {
      const auto* source = reinterpret_cast<const float*>(input->storage->bytes.data()) + input->offset;
      for (int64_t i = 0; i < input->numel(); ++i) {
        count += source[i] != 0.0f ? 1 : 0;
      }
    } else {
      for (int64_t i = 0; i < input->numel(); ++i) {
        count += input->value_at_linear(i) != 0.0 ? 1 : 0;
      }
    }
    return mtorch::make_tensor({static_cast<double>(count)}, {}, dtype, false, input->device);
  }

  const auto normalized_dims = normalized_dims_for_tensor(input, *dims, "norm");
  if (normalized_dims.size() == 1 && input->dtype == ScalarType::Float32 && input->is_contiguous() &&
      input->dim() == 2) {
    const int64_t dim = normalized_dims[0];
    const int64_t rows = input->sizes[0];
    const int64_t cols = input->sizes[1];
    std::vector<int64_t> out_shape = keepdim
        ? (dim == 0 ? std::vector<int64_t>{1, cols} : std::vector<int64_t>{rows, 1})
        : (dim == 0 ? std::vector<int64_t>{cols} : std::vector<int64_t>{rows});
    auto result = mtorch::zeros(out_shape, dtype, input->device);
    const auto* source = reinterpret_cast<const float*>(input->storage->bytes.data()) + input->offset;
    if (dim == 1) {
      for (int64_t row = 0; row < rows; ++row) {
        int64_t count = 0;
        const float* source_row = source + row * cols;
        for (int64_t col = 0; col < cols; ++col) {
          count += source_row[col] != 0.0f ? 1 : 0;
        }
        result->set_at_linear(row, static_cast<double>(count));
      }
      return result;
    }
    for (int64_t col = 0; col < cols; ++col) {
      int64_t count = 0;
      for (int64_t row = 0; row < rows; ++row) {
        count += source[row * cols + col] != 0.0f ? 1 : 0;
      }
      result->set_at_linear(col, static_cast<double>(count));
    }
    return result;
  }

  std::vector<bool> reduce_axis(static_cast<size_t>(input->dim()), false);
  for (int64_t dim : normalized_dims) {
    reduce_axis[static_cast<size_t>(dim)] = true;
  }
  std::vector<int64_t> out_shape;
  out_shape.reserve(input->sizes.size());
  for (int64_t dim = 0; dim < input->dim(); ++dim) {
    if (reduce_axis[static_cast<size_t>(dim)]) {
      if (keepdim) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input->sizes[static_cast<size_t>(dim)]);
    }
  }
  auto result = mtorch::zeros(out_shape, dtype, input->device);
  for (int64_t linear = 0; linear < input->numel(); ++linear) {
    if (input->value_at_linear(linear) == 0.0) {
      continue;
    }
    std::vector<int64_t> input_index(static_cast<size_t>(input->dim()));
    int64_t remaining = linear;
    for (int64_t dim = input->dim() - 1; dim >= 0; --dim) {
      const int64_t size = input->sizes[static_cast<size_t>(dim)];
      input_index[static_cast<size_t>(dim)] = remaining % size;
      remaining /= size;
    }
    std::vector<int64_t> out_index;
    out_index.reserve(out_shape.size());
    for (int64_t dim = 0; dim < input->dim(); ++dim) {
      if (reduce_axis[static_cast<size_t>(dim)]) {
        if (keepdim) {
          out_index.push_back(0);
        }
      } else {
        out_index.push_back(input_index[static_cast<size_t>(dim)]);
      }
    }
    result->set_at_index(out_index, result->value_at_index(out_index) + 1.0);
  }
  return result;
}

TensorPtr norm_tensor(
    const TensorPtr& input,
    double p,
    const std::optional<std::vector<int64_t>>& dims,
    bool keepdim,
    ScalarType dtype) {
  if (!is_floating_scalar_type(input->dtype)) {
    throw std::runtime_error("norm expects a floating point input");
  }
  if (!is_floating_scalar_type(dtype)) {
    throw std::runtime_error("norm dtype must be floating point");
  }
  TensorPtr source = input->dtype == dtype ? input : mtorch::to(input, dtype, input->device);
  TensorPtr result;
  if (p == std::numeric_limits<double>::infinity()) {
    auto values = mtorch::unary(source, "abs");
    result = dims.has_value() ? mtorch::amax(values, *dims, keepdim) : mtorch::reduce_max(values);
  } else if (p == -std::numeric_limits<double>::infinity()) {
    auto values = mtorch::unary(source, "abs");
    result = dims.has_value() ? mtorch::amin(values, *dims, keepdim) : mtorch::reduce_min(values);
  } else if (p == 1.0) {
    result = reduce_sum_dims_for_norm(mtorch::unary(source, "abs"), dims, keepdim, dtype);
  } else if (p == 2.0) {
    auto squared = mtorch::binary_tensor_tensor(source, source, "mul");
    result = mtorch::unary(reduce_sum_dims_for_norm(squared, dims, keepdim, dtype), "sqrt");
  } else if (p == 0.0) {
    result = norm_zero_tensor(source, dims, keepdim, dtype);
  } else {
    auto powered = mtorch::binary_tensor_scalar(mtorch::unary(source, "abs"), p, ScalarType::Float32, "pow");
    auto total = reduce_sum_dims_for_norm(powered, dims, keepdim, dtype);
    result = mtorch::binary_tensor_scalar(total, 1.0 / p, ScalarType::Float32, "pow");
  }
  return result;
}

std::optional<std::vector<int64_t>> optional_dims_from_py(PyObject* dim) {
  if (dim == Py_None) {
    return std::nullopt;
  }
  return shape_from_object(dim);
}

PyObject* py_norm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "p", "dim", "keepdim", "out", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* p = Py_None;
  PyObject* dim = Py_None;
  int keepdim = 0;
  PyObject* out = Py_None;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|OOpOO:norm", const_cast<char**>(keywords), &input, &p, &dim, &keepdim, &out, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "norm expected Tensor input");
    return nullptr;
  }
  if (out != Py_None && !is_tensor(out)) {
    PyErr_SetString(PyExc_TypeError, "norm out must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    auto result = norm_tensor(
        input_tensor,
        norm_order_from_py(p),
        optional_dims_from_py(dim),
        keepdim != 0,
        dtype_from_py(dtype, input_tensor->dtype));
    if (out != Py_None) {
      tensor_ref(out)->copy_from(*result);
      Py_INCREF(out);
      return out;
    }
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_normalize_l2(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "eps", nullptr};
  PyObject* input = nullptr;
  long long dim = 1;
  double eps = 1e-12;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Ld:_normalize_l2", const_cast<char**>(keywords), &input, &dim, &eps)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_normalize_l2 expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::normalize_l2(tensor_ref(input), dim, eps));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_binary(PyObject*, PyObject* args, const std::string& op) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &left, &right)) {
    return nullptr;
  }
  return binary_dispatch(left, right, op);
}

PyObject* py_add(PyObject* self, PyObject* args) {
  return py_binary(self, args, "add");
}

PyObject* py_sub(PyObject* self, PyObject* args) {
  return py_binary(self, args, "sub");
}

PyObject* py_mul(PyObject* self, PyObject* args) {
  return py_binary(self, args, "mul");
}

PyObject* py_div(PyObject* self, PyObject* args) {
  return py_binary(self, args, "div");
}

PyObject* py_pow(PyObject* self, PyObject* args) {
  return py_binary(self, args, "pow");
}

PyObject* py_floor_divide(PyObject* self, PyObject* args) {
  return py_binary(self, args, "floor_divide");
}

PyObject* py_float_power(PyObject* self, PyObject* args) {
  return py_binary(self, args, "float_power");
}

PyObject* py_remainder(PyObject* self, PyObject* args) {
  return py_binary(self, args, "remainder");
}

PyObject* py_fmod(PyObject* self, PyObject* args) {
  return py_binary(self, args, "fmod");
}

PyObject* py_atan2(PyObject* self, PyObject* args) {
  return py_binary(self, args, "atan2");
}

PyObject* py_hypot(PyObject* self, PyObject* args) {
  return py_binary(self, args, "hypot");
}

PyObject* py_ldexp(PyObject* self, PyObject* args) {
  return py_binary(self, args, "ldexp");
}

PyObject* py_nextafter(PyObject* self, PyObject* args) {
  return py_binary(self, args, "nextafter");
}

PyObject* py_copysign(PyObject* self, PyObject* args) {
  return py_binary(self, args, "copysign");
}

PyObject* py_heaviside(PyObject* self, PyObject* args) {
  return py_binary(self, args, "heaviside");
}

PyObject* py_logaddexp(PyObject* self, PyObject* args) {
  return py_binary(self, args, "logaddexp");
}

PyObject* py_logaddexp2(PyObject* self, PyObject* args) {
  return py_binary(self, args, "logaddexp2");
}

PyObject* py_xlogy(PyObject* self, PyObject* args) {
  return py_binary(self, args, "xlogy");
}

PyObject* py_fmax(PyObject* self, PyObject* args) {
  return py_binary(self, args, "fmax");
}

PyObject* py_fmin(PyObject* self, PyObject* args) {
  return py_binary(self, args, "fmin");
}

PyObject* py_addcmul(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "tensor1", "tensor2", "value", nullptr};
  PyObject* input = nullptr;
  PyObject* tensor1 = nullptr;
  PyObject* tensor2 = nullptr;
  double value = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|d:addcmul", const_cast<char**>(keywords), &input, &tensor1, &tensor2, &value)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(tensor1) || !is_tensor(tensor2)) {
    PyErr_SetString(PyExc_TypeError, "addcmul expected Tensor inputs");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::addcmul(tensor_ref(input), tensor_ref(tensor1), tensor_ref(tensor2), value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_addcdiv(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "tensor1", "tensor2", "value", nullptr};
  PyObject* input = nullptr;
  PyObject* tensor1 = nullptr;
  PyObject* tensor2 = nullptr;
  double value = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|d:addcdiv", const_cast<char**>(keywords), &input, &tensor1, &tensor2, &value)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(tensor1) || !is_tensor(tensor2)) {
    PyErr_SetString(PyExc_TypeError, "addcdiv expected Tensor inputs");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::addcdiv(tensor_ref(input), tensor_ref(tensor1), tensor_ref(tensor2), value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_maximum(PyObject* self, PyObject* args) {
  return py_binary(self, args, "maximum");
}

PyObject* py_minimum(PyObject* self, PyObject* args) {
  return py_binary(self, args, "minimum");
}

PyObject* py_eq(PyObject* self, PyObject* args) {
  return py_binary(self, args, "eq");
}

PyObject* py_ne(PyObject* self, PyObject* args) {
  return py_binary(self, args, "ne");
}

PyObject* py_lt(PyObject* self, PyObject* args) {
  return py_binary(self, args, "lt");
}

PyObject* py_le(PyObject* self, PyObject* args) {
  return py_binary(self, args, "le");
}

PyObject* py_gt(PyObject* self, PyObject* args) {
  return py_binary(self, args, "gt");
}

PyObject* py_ge(PyObject* self, PyObject* args) {
  return py_binary(self, args, "ge");
}

PyObject* py_logical_and(PyObject* self, PyObject* args) {
  return py_binary(self, args, "logical_and");
}

PyObject* py_logical_or(PyObject* self, PyObject* args) {
  return py_binary(self, args, "logical_or");
}

PyObject* py_logical_xor(PyObject* self, PyObject* args) {
  return py_binary(self, args, "logical_xor");
}

PyObject* py_bitwise_and(PyObject* self, PyObject* args) {
  return py_binary(self, args, "bitwise_and");
}

PyObject* py_bitwise_or(PyObject* self, PyObject* args) {
  return py_binary(self, args, "bitwise_or");
}

PyObject* py_bitwise_xor(PyObject* self, PyObject* args) {
  return py_binary(self, args, "bitwise_xor");
}

PyObject* py_isclose(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "other", "rtol", "atol", "equal_nan", nullptr};
  PyObject* input = nullptr;
  PyObject* other = nullptr;
  double rtol = 1e-5;
  double atol = 1e-8;
  int equal_nan = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|ddp:isclose",
          const_cast<char**>(keywords),
          &input,
          &other,
          &rtol,
          &atol,
          &equal_nan)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "isclose expected Tensor inputs");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::isclose(tensor_ref(input), tensor_ref(other), rtol, atol, equal_nan != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_allclose(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "other", "rtol", "atol", "equal_nan", nullptr};
  PyObject* input = nullptr;
  PyObject* other = nullptr;
  double rtol = 1e-5;
  double atol = 1e-8;
  int equal_nan = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|ddp:allclose",
          const_cast<char**>(keywords),
          &input,
          &other,
          &rtol,
          &atol,
          &equal_nan)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "allclose expected Tensor inputs");
    return nullptr;
  }
  try {
    if (mtorch::allclose(tensor_ref(input), tensor_ref(other), rtol, atol, equal_nan != 0)) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_equal(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:equal", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "equal expected Tensor inputs");
    return nullptr;
  }
  try {
    const TensorPtr& left_tensor = tensor_ref(left);
    const TensorPtr& right_tensor = tensor_ref(right);
    if (!mtorch::devices_equal(left_tensor->device, right_tensor->device) || left_tensor->sizes != right_tensor->sizes) {
      Py_RETURN_FALSE;
    }
    if (left_tensor->dtype != right_tensor->dtype) {
      Py_RETURN_FALSE;
    }
    for (int64_t i = 0; i < left_tensor->numel(); ++i) {
      const double left_value = left_tensor->value_at_linear(i);
      const double right_value = right_tensor->value_at_linear(i);
      if (left_value != right_value) {
        Py_RETURN_FALSE;
      }
    }
    Py_RETURN_TRUE;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

int tensor_truth_value(const TensorPtr& tensor) {
  const int64_t elements = tensor->numel();
  if (elements == 0) {
    PyErr_SetString(PyExc_RuntimeError, "Boolean value of Tensor with no values is ambiguous");
    return -1;
  }
  if (elements > 1) {
    PyErr_SetString(PyExc_RuntimeError, "Boolean value of Tensor with more than one value is ambiguous");
    return -1;
  }
  return tensor->value_at_linear(0) != 0.0 ? 1 : 0;
}

PyObject* py_is_nonzero(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_nonzero", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "is_nonzero expected Tensor");
    return nullptr;
  }
  try {
    const int value = tensor_truth_value(tensor_ref(input));
    if (value < 0) {
      return nullptr;
    }
    if (value != 0) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_lerp(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* end = nullptr;
  PyObject* weight = nullptr;
  if (!PyArg_ParseTuple(args, "OOO:lerp", &input, &end, &weight)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(end)) {
    PyErr_SetString(PyExc_TypeError, "lerp expected Tensor input and end");
    return nullptr;
  }
  try {
    if (is_tensor(weight)) {
      return wrap_tensor(mtorch::lerp(tensor_ref(input), tensor_ref(end), tensor_ref(weight)));
    }
    double scalar = 0.0;
    ScalarType scalar_dtype = ScalarType::Float32;
    if (pyobject_to_scalar(weight, scalar, &scalar_dtype)) {
      return wrap_tensor(mtorch::lerp(tensor_ref(input), tensor_ref(end), scalar, scalar_dtype));
    }
    PyErr_SetString(PyExc_TypeError, "lerp weight must be a Tensor or scalar");
    return nullptr;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_sum(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* dtype = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpO:sum", const_cast<char**>(keywords), &input, &dim, &keepdim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "sum expected Tensor");
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    const ScalarType result_dtype = dtype_from_py(dtype, default_sum_dtype(tensor->dtype));
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_sum(tensor, result_dtype));
    }
    return wrap_tensor(mtorch::reduce_sum_dim(tensor, PyLong_AsLongLong(dim), keepdim != 0, result_dtype));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_diff_float32(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", nullptr};
  PyObject* input = nullptr;
  long long dim = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OL:_diff_float32", const_cast<char**>(keywords), &input, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_diff_float32 expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diff_float32(tensor_ref(input), static_cast<int64_t>(dim)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cumsum(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = nullptr;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:cumsum", const_cast<char**>(keywords), &input, &dim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "cumsum expected Tensor");
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    return wrap_tensor(mtorch::cumsum(tensor, PyLong_AsLongLong(dim), dtype_from_py(dtype, default_sum_dtype(tensor->dtype))));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cumprod(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = nullptr;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:cumprod", const_cast<char**>(keywords), &input, &dim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "cumprod expected Tensor");
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    return wrap_tensor(mtorch::cumprod(tensor, PyLong_AsLongLong(dim), dtype_from_py(dtype, default_sum_dtype(tensor->dtype))));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cumulative_extreme(PyObject* args, PyObject* kwargs, bool max_mode) {
  static const char* keywords[] = {"input", "dim", "out", nullptr};
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* out = Py_None;
  const char* name = max_mode ? "cummax" : "cummin";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OL|O", const_cast<char**>(keywords), &input, &dim, &out)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, max_mode ? "cummax expected Tensor" : "cummin expected Tensor");
    return nullptr;
  }
  try {
    auto result = max_mode ? mtorch::cummax(tensor_ref(input), dim) : mtorch::cummin(tensor_ref(input), dim);
    if (out != Py_None) {
      if (!PyTuple_Check(out) || PyTuple_GET_SIZE(out) != 2 || !is_tensor(PyTuple_GET_ITEM(out, 0)) ||
          !is_tensor(PyTuple_GET_ITEM(out, 1))) {
        PyErr_Format(PyExc_TypeError, "%s out must be a tuple of two Tensors", name);
        return nullptr;
      }
      tensor_ref(PyTuple_GET_ITEM(out, 0))->copy_from(*result.first);
      tensor_ref(PyTuple_GET_ITEM(out, 1))->copy_from(*result.second);
      result = {tensor_ref(PyTuple_GET_ITEM(out, 0)), tensor_ref(PyTuple_GET_ITEM(out, 1))};
    }
    PyObject* tuple = PyTuple_New(2);
    if (tuple == nullptr) {
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, wrap_tensor(result.first));
    PyTuple_SET_ITEM(tuple, 1, wrap_tensor(result.second));
    return tuple;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cummax(PyObject*, PyObject* args, PyObject* kwargs) {
  return py_cumulative_extreme(args, kwargs, true);
}

PyObject* py_cummin(PyObject*, PyObject* args, PyObject* kwargs) {
  return py_cumulative_extreme(args, kwargs, false);
}

PyObject* py_trapezoid_dx(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dx", "dim", nullptr};
  PyObject* input = nullptr;
  double dx = 1.0;
  long long dim = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dL:_trapezoid_dx", const_cast<char**>(keywords), &input, &dx, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_trapezoid_dx expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::trapezoid_dx(tensor_ref(input), dx, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cumulative_trapezoid_dx(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dx", "dim", nullptr};
  PyObject* input = nullptr;
  double dx = 1.0;
  long long dim = -1;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|dL:_cumulative_trapezoid_dx",
          const_cast<char**>(keywords),
          &input,
          &dx,
          &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_cumulative_trapezoid_dx expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::cumulative_trapezoid_dx(tensor_ref(input), dx, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_gradient_uniform(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "spacing", "dim", "edge_order", nullptr};
  PyObject* input = nullptr;
  double spacing = 1.0;
  long long dim = -1;
  long long edge_order = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|dLL:_gradient_uniform",
          const_cast<char**>(keywords),
          &input,
          &spacing,
          &dim,
          &edge_order)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_gradient_uniform expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::gradient_uniform(tensor_ref(input), spacing, dim, edge_order));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_mean(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* dtype = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpO:mean", const_cast<char**>(keywords), &input, &dim, &keepdim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "mean expected Tensor");
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    const ScalarType result_dtype = dtype_from_py(dtype, tensor->dtype);
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_mean(tensor, result_dtype));
    }
    return wrap_tensor(mtorch::reduce_mean_dim(tensor, PyLong_AsLongLong(dim), keepdim != 0, result_dtype));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_prod(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", "dtype", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* dtype = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OpO:prod", const_cast<char**>(keywords), &input, &dim, &keepdim, &dtype)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "prod expected Tensor");
    return nullptr;
  }
  try {
    const TensorPtr& tensor = tensor_ref(input);
    const ScalarType result_dtype = dtype_from_py(dtype, default_sum_dtype(tensor->dtype));
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_prod(tensor, result_dtype));
    }
    return wrap_tensor(mtorch::reduce_prod_dim(tensor, PyLong_AsLongLong(dim), keepdim != 0, result_dtype));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_var(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "unbiased", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_None;
  int keepdim = 0;
  PyObject* correction = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOpO:var",
          const_cast<char**>(keywords),
          &input,
          &dim,
          &unbiased,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "var expected Tensor");
    return nullptr;
  }
  try {
    const double correction_value = correction_from_py(correction, unbiased);
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_var(tensor_ref(input), correction_value));
    }
    return wrap_tensor(mtorch::reduce_var_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0, correction_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_std(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "unbiased", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_None;
  int keepdim = 0;
  PyObject* correction = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOpO:std",
          const_cast<char**>(keywords),
          &input,
          &dim,
          &unbiased,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "std expected Tensor");
    return nullptr;
  }
  try {
    const double correction_value = correction_from_py(correction, unbiased);
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_std(tensor_ref(input), correction_value));
    }
    return wrap_tensor(mtorch::reduce_std_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0, correction_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_var_tail(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "start_dim", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  long long start_dim = 0;
  int keepdim = 0;
  double correction = 1.0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OL|pd:_var_tail",
          const_cast<char**>(keywords),
          &input,
          &start_dim,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "var tail expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::reduce_var_tail(tensor_ref(input), static_cast<int64_t>(start_dim), keepdim != 0, correction));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_std_tail(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "start_dim", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  long long start_dim = 0;
  int keepdim = 0;
  double correction = 1.0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OL|pd:_std_tail",
          const_cast<char**>(keywords),
          &input,
          &start_dim,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "std tail expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::reduce_std_tail(tensor_ref(input), static_cast<int64_t>(start_dim), keepdim != 0, correction));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* wrap_tensor_pair(const std::pair<TensorPtr, TensorPtr>& result) {
  PyObject* tuple = PyTuple_New(2);
  if (tuple == nullptr) {
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple, 0, wrap_tensor(result.first));
  PyTuple_SET_ITEM(tuple, 1, wrap_tensor(result.second));
  return tuple;
}

PyObject* py_var_mean(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "unbiased", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_None;
  int keepdim = 0;
  PyObject* correction = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOpO:_var_mean",
          const_cast<char**>(keywords),
          &input,
          &dim,
          &unbiased,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "var_mean expected Tensor");
    return nullptr;
  }
  try {
    const double correction_value = correction_from_py(correction, unbiased);
    if (dim == Py_None) {
      return wrap_tensor_pair(mtorch::reduce_var_mean(tensor_ref(input), correction_value));
    }
    return wrap_tensor_pair(
        mtorch::reduce_var_mean_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0, correction_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_std_mean(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "unbiased", "keepdim", "correction", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  PyObject* unbiased = Py_None;
  int keepdim = 0;
  PyObject* correction = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOpO:_std_mean",
          const_cast<char**>(keywords),
          &input,
          &dim,
          &unbiased,
          &keepdim,
          &correction)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "std_mean expected Tensor");
    return nullptr;
  }
  try {
    const double correction_value = correction_from_py(correction, unbiased);
    if (dim == Py_None) {
      return wrap_tensor_pair(mtorch::reduce_std_mean(tensor_ref(input), correction_value));
    }
    return wrap_tensor_pair(
        mtorch::reduce_std_mean_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0, correction_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_all(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:all", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "all expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_all(tensor_ref(input)));
    }
    return wrap_tensor(mtorch::reduce_all_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_any(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:any", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "any expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_any(tensor_ref(input)));
    }
    return wrap_tensor(mtorch::reduce_any_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_amax(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:amax", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "amax expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_max(tensor_ref(input)));
    }
    return wrap_tensor(mtorch::amax(tensor_ref(input), shape_from_object(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_amin(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:amin", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "amin expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_min(tensor_ref(input)));
    }
    return wrap_tensor(mtorch::amin(tensor_ref(input), shape_from_object(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_max(PyObject*, PyObject* args, PyObject* kwargs) {
  if ((kwargs == nullptr || PyDict_Size(kwargs) == 0) && PyTuple_GET_SIZE(args) == 2) {
    PyObject* left = PyTuple_GET_ITEM(args, 0);
    PyObject* right = PyTuple_GET_ITEM(args, 1);
    if (is_tensor(left) && is_tensor(right)) {
      return binary_dispatch(left, right, "maximum");
    }
  }
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:max", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "max expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_max(tensor_ref(input)));
    }
    auto result = mtorch::reduce_max_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0);
    PyObject* tuple = PyTuple_New(2);
    if (tuple == nullptr) {
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, wrap_tensor(result.first));
    PyTuple_SET_ITEM(tuple, 1, wrap_tensor(result.second));
    return tuple;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_argmax(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:argmax", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "argmax expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::argmax(tensor_ref(input), keepdim != 0));
    }
    return wrap_tensor(mtorch::argmax_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_min(PyObject*, PyObject* args, PyObject* kwargs) {
  if ((kwargs == nullptr || PyDict_Size(kwargs) == 0) && PyTuple_GET_SIZE(args) == 2) {
    PyObject* left = PyTuple_GET_ITEM(args, 0);
    PyObject* right = PyTuple_GET_ITEM(args, 1);
    if (is_tensor(left) && is_tensor(right)) {
      return binary_dispatch(left, right, "minimum");
    }
  }
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:min", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "min expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::reduce_min(tensor_ref(input)));
    }
    auto result = mtorch::reduce_min_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0);
    PyObject* tuple = PyTuple_New(2);
    if (tuple == nullptr) {
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, 0, wrap_tensor(result.first));
    PyTuple_SET_ITEM(tuple, 1, wrap_tensor(result.second));
    return tuple;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_argmin(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "keepdim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  int keepdim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op:argmin", const_cast<char**>(keywords), &input, &dim, &keepdim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "argmin expected Tensor");
    return nullptr;
  }
  try {
    if (dim == Py_None) {
      return wrap_tensor(mtorch::argmin(tensor_ref(input), keepdim != 0));
    }
    return wrap_tensor(mtorch::argmin_dim(tensor_ref(input), PyLong_AsLongLong(dim), keepdim != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_reshape(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* shape = nullptr;
  if (!PyArg_ParseTuple(args, "OO:reshape", &input, &shape)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "reshape expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::reshape(tensor_ref(input), shape_from_object(shape)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unflatten(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* sizes = nullptr;
  if (!PyArg_ParseTuple(args, "OLO:unflatten", &input, &dim, &sizes)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unflatten expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unflatten(tensor_ref(input), dim, shape_from_object(sizes)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_transpose(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim0 = 0;
  long long dim1 = 0;
  if (!PyArg_ParseTuple(args, "OLL:transpose", &input, &dim0, &dim1)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "transpose expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(tensor_ref(input), dim0, dim1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_permute(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* dims = nullptr;
  if (!PyArg_ParseTuple(args, "OO:permute", &input, &dims)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "permute expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::permute(tensor_ref(input), shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_movedim(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "source", "destination", nullptr};
  PyObject* input = nullptr;
  PyObject* source = nullptr;
  PyObject* destination = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOO:movedim", const_cast<char**>(keywords), &input, &source, &destination)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "movedim expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::movedim(tensor_ref(input), shape_from_object(source), shape_from_object(destination)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_swapaxes(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "axis0", "axis1", nullptr};
  PyObject* input = nullptr;
  long long axis0 = 0;
  long long axis1 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OLL:swapaxes", const_cast<char**>(keywords), &input, &axis0, &axis1)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "swapaxes expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(tensor_ref(input), axis0, axis1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_swapdims(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim0", "dim1", nullptr};
  PyObject* input = nullptr;
  long long dim0 = 0;
  long long dim1 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OLL:swapdims", const_cast<char**>(keywords), &input, &dim0, &dim1)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "swapdims expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(tensor_ref(input), dim0, dim1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_flatten(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "start_dim", "end_dim", nullptr};
  PyObject* input = nullptr;
  long long start_dim = 0;
  long long end_dim = -1;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|LL:flatten", const_cast<char**>(keywords), &input, &start_dim, &end_dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "flatten expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::flatten(tensor_ref(input), start_dim, end_dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_ravel(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:ravel", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "ravel expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::ravel(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_t(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:t", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "t expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::t(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_broadcast_to(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* shape = nullptr;
  if (!PyArg_ParseTuple(args, "OO:broadcast_to", &input, &shape)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "broadcast_to expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::broadcast_to(tensor_ref(input), shape_from_object(shape)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_tile(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* dims = nullptr;
  if (!PyArg_ParseTuple(args, "OO:tile", &input, &dims)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "tile expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::tile(tensor_ref(input), shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_repeat_interleave(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "repeats", "dim", "output_size", nullptr};
  PyObject* input = nullptr;
  PyObject* repeats = nullptr;
  PyObject* dim = Py_None;
  PyObject* output_size = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OO|OO:repeat_interleave", const_cast<char**>(keywords), &input, &repeats, &dim, &output_size)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "repeat_interleave expected Tensor input");
    return nullptr;
  }
  try {
    const auto dim_value = optional_int64_from_py(dim, "dim");
    const auto output_size_value = optional_int64_from_py(output_size, "output_size");
    if (is_tensor(repeats)) {
      return wrap_tensor(mtorch::repeat_interleave(tensor_ref(input), tensor_ref(repeats), dim_value, output_size_value));
    }
    const int64_t repeat_count = PyLong_AsLongLong(repeats);
    if (PyErr_Occurred()) {
      throw std::invalid_argument("repeat_interleave repeats must be an integer or Tensor");
    }
    return wrap_tensor(mtorch::repeat_interleave(tensor_ref(input), repeat_count, dim_value, output_size_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_flip(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* dims = nullptr;
  if (!PyArg_ParseTuple(args, "OO:flip", &input, &dims)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "flip expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::flip(tensor_ref(input), shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_pad(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "pad", "mode", "value", nullptr};
  PyObject* input = nullptr;
  PyObject* padding = nullptr;
  PyObject* mode = Py_None;
  PyObject* value = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO:pad", const_cast<char**>(keywords), &input, &padding, &mode, &value)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "pad expected Tensor input");
    return nullptr;
  }
  try {
    std::string mode_text = "constant";
    if (mode != Py_None) {
      const char* parsed_mode = PyUnicode_AsUTF8(mode);
      if (parsed_mode == nullptr) {
        throw std::invalid_argument("pad mode must be a string");
      }
      mode_text = parsed_mode;
    }
    if (mode_text != "constant" && value != Py_None) {
      throw std::invalid_argument("pad value is only supported for constant mode");
    }
    const double fill_value = value == Py_None ? 0.0 : scalar_from_py(value);
    return wrap_tensor(mtorch::pad(tensor_ref(input), shape_from_object(padding), mode_text, fill_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_adaptive_avg_pool1d(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* output_size = nullptr;
  if (!PyArg_ParseTuple(args, "OO:adaptive_avg_pool1d", &input, &output_size)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "adaptive_avg_pool1d expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::adaptive_avg_pool1d(tensor_ref(input), shape_from_object(output_size)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_adaptive_avg_pool2d(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* output_size = nullptr;
  if (!PyArg_ParseTuple(args, "OO:adaptive_avg_pool2d", &input, &output_size)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "adaptive_avg_pool2d expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::adaptive_avg_pool2d(tensor_ref(input), shape_from_object(output_size)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_pixel_shuffle(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  int64_t factor = 0;
  if (!PyArg_ParseTuple(args, "OL:pixel_shuffle", &input, &factor)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "pixel_shuffle expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::pixel_shuffle(tensor_ref(input), factor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_pixel_unshuffle(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  int64_t factor = 0;
  if (!PyArg_ParseTuple(args, "OL:pixel_unshuffle", &input, &factor)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "pixel_unshuffle expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::pixel_unshuffle(tensor_ref(input), factor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_channel_shuffle(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  int64_t groups = 0;
  if (!PyArg_ParseTuple(args, "OL:channel_shuffle", &input, &groups)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "channel_shuffle expected Tensor input");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::channel_shuffle(tensor_ref(input), groups));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_interpolate(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "size", "scale_factor", "mode", "align_corners", "recompute_scale_factor", "antialias", nullptr};
  PyObject* input = nullptr;
  PyObject* size = Py_None;
  PyObject* scale_factor = Py_None;
  PyObject* mode = Py_None;
  PyObject* align_corners = Py_None;
  PyObject* recompute_scale_factor = Py_None;
  PyObject* antialias = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOOOOO:interpolate",
          const_cast<char**>(keywords),
          &input,
          &size,
          &scale_factor,
          &mode,
          &align_corners,
          &recompute_scale_factor,
          &antialias)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "interpolate expected Tensor input");
    return nullptr;
  }
  try {
    auto tensor = tensor_ref(input);
    if (tensor->dim() < 3) {
      throw std::invalid_argument("interpolate expects at least a 3-D tensor");
    }
    const int64_t spatial_dims = tensor->dim() - 2;
    std::string mode_text = "nearest";
    if (mode != Py_None) {
      const char* raw_mode = PyUnicode_AsUTF8(mode);
      if (raw_mode == nullptr) {
        throw std::invalid_argument("interpolate mode must be a string");
      }
      mode_text = raw_mode;
    }
    const int align_corners_value = align_corners == Py_None ? 0 : PyObject_IsTrue(align_corners);
    if (align_corners_value < 0) {
      return nullptr;
    }
    if ((mode_text == "nearest" || mode_text == "nearest-exact" || mode_text == "area") && align_corners != Py_None) {
      throw std::invalid_argument("interpolate align_corners is not supported for nearest, nearest-exact, or area mode");
    }
    if (antialias != Py_None && PyObject_IsTrue(antialias)) {
      throw std::invalid_argument("interpolate antialias is not implemented");
    }
    (void)recompute_scale_factor;

    std::vector<int64_t> output_size;
    if (size != Py_None) {
      output_size = shape_from_object(size);
      if (output_size.size() == 1 && spatial_dims > 1) {
        const int64_t repeated_size = output_size[0];
        output_size.assign(static_cast<size_t>(spatial_dims), repeated_size);
      }
    } else if (scale_factor != Py_None) {
      auto factors = double_vector_from_object(scale_factor);
      if (factors.size() == 1 && spatial_dims > 1) {
        const double repeated_factor = factors[0];
        factors.assign(static_cast<size_t>(spatial_dims), repeated_factor);
      }
      if (static_cast<int64_t>(factors.size()) != spatial_dims) {
        throw std::invalid_argument("interpolate scale_factor must match spatial dimensions");
      }
      output_size.reserve(static_cast<size_t>(spatial_dims));
      for (int64_t dim = 0; dim < spatial_dims; ++dim) {
        const double scaled = static_cast<double>(tensor->sizes[static_cast<size_t>(dim + 2)]) *
            factors[static_cast<size_t>(dim)];
        output_size.push_back(static_cast<int64_t>(std::floor(scaled)));
      }
    } else {
      throw std::invalid_argument("interpolate requires size or scale_factor");
    }
    return wrap_tensor(mtorch::interpolate(tensor, output_size, mode_text, align_corners_value != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_grid_sample(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "grid", "mode", "padding_mode", "align_corners", nullptr};
  PyObject* input = nullptr;
  PyObject* grid = nullptr;
  PyObject* mode = Py_None;
  PyObject* padding_mode = Py_None;
  PyObject* align_corners = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOO:grid_sample",
          const_cast<char**>(keywords),
          &input,
          &grid,
          &mode,
          &padding_mode,
          &align_corners)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(grid)) {
    PyErr_SetString(PyExc_TypeError, "grid_sample expected input and grid tensors");
    return nullptr;
  }
  try {
    std::string mode_text = "bilinear";
    if (mode != Py_None) {
      const char* raw_mode = PyUnicode_AsUTF8(mode);
      if (raw_mode == nullptr) {
        throw std::invalid_argument("grid_sample mode must be a string");
      }
      mode_text = raw_mode;
    }
    std::string padding_mode_text = "zeros";
    if (padding_mode != Py_None) {
      const char* raw_padding_mode = PyUnicode_AsUTF8(padding_mode);
      if (raw_padding_mode == nullptr) {
        throw std::invalid_argument("grid_sample padding_mode must be a string");
      }
      padding_mode_text = raw_padding_mode;
    }
    const int align_corners_value = align_corners == Py_None ? 0 : PyObject_IsTrue(align_corners);
    if (align_corners_value < 0) {
      return nullptr;
    }
    return wrap_tensor(
        mtorch::grid_sample(tensor_ref(input), tensor_ref(grid), mode_text, padding_mode_text, align_corners_value != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_affine_grid(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"theta", "size", "align_corners", nullptr};
  PyObject* theta = nullptr;
  PyObject* size = nullptr;
  PyObject* align_corners = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OO|O:affine_grid", const_cast<char**>(keywords), &theta, &size, &align_corners)) {
    return nullptr;
  }
  if (!is_tensor(theta)) {
    PyErr_SetString(PyExc_TypeError, "affine_grid expected theta Tensor");
    return nullptr;
  }
  try {
    const int align_corners_value = align_corners == Py_None ? 0 : PyObject_IsTrue(align_corners);
    if (align_corners_value < 0) {
      return nullptr;
    }
    return wrap_tensor(mtorch::affine_grid(tensor_ref(theta), shape_from_object(size), align_corners_value != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_fliplr(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:fliplr", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "fliplr expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::fliplr(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_flipud(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:flipud", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "flipud expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::flipud(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_rot90(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "k", "dims", nullptr};
  PyObject* input = nullptr;
  long long k = 1;
  PyObject* dims = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|LO:rot90", const_cast<char**>(keywords), &input, &k, &dims)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "rot90 expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::rot90(tensor_ref(input), k, dims == Py_None ? std::vector<int64_t>{0, 1} : shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_roll(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "shifts", "dims", nullptr};
  PyObject* input = nullptr;
  PyObject* shifts = nullptr;
  PyObject* dims = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:roll", const_cast<char**>(keywords), &input, &shifts, &dims)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "roll expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::roll(tensor_ref(input), shape_from_object(shifts), dims == Py_None ? std::vector<int64_t>{} : shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_squeeze(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:squeeze", const_cast<char**>(keywords), &input, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "squeeze expected Tensor");
    return nullptr;
  }
  try {
    if (dim != Py_None) {
      return wrap_tensor(mtorch::squeeze(tensor_ref(input), PyLong_AsLongLong(dim)));
    }
    return wrap_tensor(mtorch::squeeze(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unsqueeze(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTuple(args, "OL:unsqueeze", &input, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unsqueeze expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unsqueeze(tensor_ref(input), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_narrow(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  long long start = 0;
  long long length = 0;
  if (!PyArg_ParseTuple(args, "OLLL:narrow", &input, &dim, &start, &length)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "narrow expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::narrow(tensor_ref(input), dim, start, length));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_select(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  long long index = 0;
  if (!PyArg_ParseTuple(args, "OLL:select", &input, &dim, &index)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "select expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::select(tensor_ref(input), dim, index));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_as_strided(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "size", "stride", "storage_offset", nullptr};
  PyObject* input = nullptr;
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  PyObject* storage_offset = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOO|O:as_strided", const_cast<char**>(keywords), &input, &size, &stride, &storage_offset)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "as_strided expected Tensor");
    return nullptr;
  }
  try {
    std::optional<int64_t> offset;
    if (storage_offset != Py_None) {
      offset = PyLong_AsLongLong(storage_offset);
      if (PyErr_Occurred()) {
        throw std::invalid_argument("storage_offset must be an integer or None");
      }
    }
    return wrap_tensor(mtorch::as_strided(tensor_ref(input), shape_from_object(size), shape_from_object(stride), offset));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_diagonal(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "offset", "dim1", "dim2", nullptr};
  PyObject* input = nullptr;
  long long offset = 0;
  long long dim1 = 0;
  long long dim2 = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|LLL:diagonal", const_cast<char**>(keywords), &input, &offset, &dim1, &dim2)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "diagonal expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diagonal(tensor_ref(input), offset, dim1, dim2));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_diag(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "diagonal", nullptr};
  PyObject* input = nullptr;
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:diag", const_cast<char**>(keywords), &input, &diagonal)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "diag expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diag(tensor_ref(input), diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_diagflat(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "offset", nullptr};
  PyObject* input = nullptr;
  long long offset = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:diagflat", const_cast<char**>(keywords), &input, &offset)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "diagflat expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diagflat(tensor_ref(input), offset));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_diag_embed(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "offset", "dim1", "dim2", nullptr};
  PyObject* input = nullptr;
  long long offset = 0;
  long long dim1 = -2;
  long long dim2 = -1;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|LLL:diag_embed", const_cast<char**>(keywords), &input, &offset, &dim1, &dim2)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "diag_embed expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diag_embed(tensor_ref(input), offset, dim1, dim2));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_block_diag(PyObject*, PyObject* args) {
  std::vector<TensorPtr> tensors;
  tensors.reserve(static_cast<size_t>(PyTuple_GET_SIZE(args)));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!is_tensor(item)) {
      PyErr_SetString(PyExc_TypeError, "block_diag expected Tensor arguments");
      return nullptr;
    }
    tensors.push_back(tensor_ref(item));
  }
  try {
    return wrap_tensor(mtorch::block_diag(tensors));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_tril(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "diagonal", nullptr};
  PyObject* input = nullptr;
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:tril", const_cast<char**>(keywords), &input, &diagonal)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "tril expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::tril(tensor_ref(input), diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_triu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "diagonal", nullptr};
  PyObject* input = nullptr;
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:triu", const_cast<char**>(keywords), &input, &diagonal)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "triu expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::triu(tensor_ref(input), diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_trace(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:trace", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "trace expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::trace(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* tuple_from_tensors(const std::vector<TensorPtr>& tensors) {
  PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(tensors.size()));
  if (tuple == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    PyObject* item = wrap_tensor(tensors[i]);
    if (item == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), item);
  }
  return tuple;
}

PyObject* py_sort(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "descending", "stable", nullptr};
  PyObject* input = nullptr;
  long long dim = -1;
  int descending = 0;
  int stable = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Lpp:sort", const_cast<char**>(keywords), &input, &dim, &descending, &stable)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "sort expected Tensor");
    return nullptr;
  }
  try {
    auto result = mtorch::sort(tensor_ref(input), dim, descending != 0, stable != 0);
    return tuple_from_tensors({result.first, result.second});
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_argsort(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", "descending", "stable", nullptr};
  PyObject* input = nullptr;
  long long dim = -1;
  int descending = 0;
  int stable = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Lpp:argsort", const_cast<char**>(keywords), &input, &dim, &descending, &stable)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "argsort expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::argsort(tensor_ref(input), dim, descending != 0, stable != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_quantile_dim_2d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "q", "dim", "interpolation", nullptr};
  PyObject* input = nullptr;
  double q = 0.0;
  long long dim = -1;
  const char* interpolation = "linear";
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OdL|s:_quantile_dim_2d", const_cast<char**>(keywords), &input, &q, &dim, &interpolation)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "quantile_dim_2d expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::quantile_dim_2d(tensor_ref(input), q, dim, interpolation));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_quantile_flat(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "q", "interpolation", nullptr};
  PyObject* input = nullptr;
  double q = 0.0;
  const char* interpolation = "linear";
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "Od|s:_quantile_flat", const_cast<char**>(keywords), &input, &q, &interpolation)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "quantile_flat expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::quantile_flat(tensor_ref(input), q, interpolation));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_topk(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "k", "dim", "largest", "sorted", nullptr};
  PyObject* input = nullptr;
  long long k = 0;
  PyObject* dim = Py_None;
  int largest = 1;
  int sorted_flag = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OL|Opp:topk", const_cast<char**>(keywords), &input, &k, &dim, &largest, &sorted_flag)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "topk expected Tensor");
    return nullptr;
  }
  try {
    const int64_t dim_value = dim == Py_None ? -1 : PyLong_AsLongLong(dim);
    auto result = mtorch::topk(tensor_ref(input), k, dim_value, largest != 0, sorted_flag != 0);
    return tuple_from_tensors({result.first, result.second});
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_searchsorted(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"sorted_sequence", "input", "out_int32", "right", "side", "out", "sorter", nullptr};
  PyObject* sorted_sequence = nullptr;
  PyObject* input = nullptr;
  int out_int32 = 0;
  int right = 0;
  PyObject* side = Py_None;
  PyObject* out = Py_None;
  PyObject* sorter = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|ppOOO:searchsorted",
          const_cast<char**>(keywords),
          &sorted_sequence,
          &input,
          &out_int32,
          &right,
          &side,
          &out,
          &sorter)) {
    return nullptr;
  }
  if (!is_tensor(sorted_sequence)) {
    PyErr_SetString(PyExc_TypeError, "searchsorted expected Tensor sorted_sequence");
    return nullptr;
  }
  if (sorter != Py_None) {
    PyErr_SetString(PyExc_NotImplementedError, "searchsorted sorter is not implemented");
    return nullptr;
  }
  if (out != Py_None && !is_tensor(out)) {
    PyErr_SetString(PyExc_TypeError, "searchsorted out must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto sorted_tensor = tensor_ref(sorted_sequence);
    TensorPtr input_tensor;
    if (is_tensor(input)) {
      input_tensor = tensor_ref(input);
    } else {
      double scalar = 0.0;
      ScalarType scalar_dtype = ScalarType::Float32;
      if (!pyobject_to_scalar(input, scalar, &scalar_dtype)) {
        throw std::invalid_argument("searchsorted input must be a Tensor or scalar");
      }
      input_tensor = mtorch::make_tensor({scalar}, {}, scalar_dtype, false, sorted_tensor->device);
    }
    bool right_value = right != 0;
    if (side != Py_None) {
      const char* raw_side = PyUnicode_AsUTF8(side);
      if (raw_side == nullptr) {
        return nullptr;
      }
      const std::string side_text(raw_side);
      if (side_text == "left") {
        if (right_value) {
          throw std::invalid_argument("searchsorted side='left' is inconsistent with right=True");
        }
        right_value = false;
      } else if (side_text == "right") {
        right_value = true;
      } else {
        throw std::invalid_argument("searchsorted side must be 'left' or 'right'");
      }
    }
    auto result = mtorch::searchsorted(sorted_tensor, input_tensor, out_int32 != 0, right_value);
    if (out != Py_None) {
      tensor_ref(out)->copy_from(*result);
      Py_INCREF(out);
      return out;
    }
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

TensorPtr tensor_or_scalar_for_isin(PyObject* object, Device scalar_device, const char* name) {
  if (is_tensor(object)) {
    return tensor_ref(object);
  }
  double scalar = 0.0;
  ScalarType scalar_dtype = ScalarType::Float32;
  if (!pyobject_to_scalar(object, scalar, &scalar_dtype)) {
    throw TypeErrorException(std::string("isin ") + name + " must be a Tensor or scalar");
  }
  return mtorch::make_tensor({scalar}, {}, scalar_dtype, false, scalar_device);
}

PyObject* py_isin(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"elements", "test_elements", "assume_unique", "invert", nullptr};
  PyObject* elements = nullptr;
  PyObject* test_elements = nullptr;
  int assume_unique = 0;
  int invert = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|pp:isin",
          const_cast<char**>(keywords),
          &elements,
          &test_elements,
          &assume_unique,
          &invert)) {
    return nullptr;
  }
  if (!is_tensor(elements) && !is_tensor(test_elements)) {
    PyErr_SetString(PyExc_TypeError, "isin expected at least one Tensor argument");
    return nullptr;
  }
  try {
    const Device scalar_device =
        is_tensor(elements) ? tensor_ref(elements)->device : tensor_ref(test_elements)->device;
    auto element_tensor = tensor_or_scalar_for_isin(elements, scalar_device, "elements");
    auto test_tensor = tensor_or_scalar_for_isin(test_elements, scalar_device, "test_elements");
    return wrap_tensor(mtorch::isin(element_tensor, test_tensor, assume_unique != 0, invert != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* unique_result_to_py(const UniqueResult& result, bool return_inverse, bool return_counts) {
  if (!return_inverse && !return_counts) {
    return wrap_tensor(result.output);
  }
  const Py_ssize_t size = 1 + (return_inverse ? 1 : 0) + (return_counts ? 1 : 0);
  PyObject* tuple = PyTuple_New(size);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_ssize_t index = 0;
  PyObject* output = wrap_tensor(result.output);
  if (output == nullptr) {
    Py_DECREF(tuple);
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple, index++, output);
  if (return_inverse) {
    PyObject* inverse = wrap_tensor(result.inverse_indices);
    if (inverse == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, index++, inverse);
  }
  if (return_counts) {
    PyObject* counts = wrap_tensor(result.counts);
    if (counts == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, index++, counts);
  }
  return tuple;
}

PyObject* py_unique(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "sorted", "return_inverse", "return_counts", "dim", nullptr};
  PyObject* input = nullptr;
  int sorted = 1;
  int return_inverse = 0;
  int return_counts = 0;
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|pppO:unique",
          const_cast<char**>(keywords),
          &input,
          &sorted,
          &return_inverse,
          &return_counts,
          &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unique expected input Tensor");
    return nullptr;
  }
  try {
    const auto dim_value = optional_int64_from_py(dim, "dim");
    auto result = mtorch::unique(tensor_ref(input), sorted != 0, return_inverse != 0, return_counts != 0, dim_value);
    return unique_result_to_py(result, return_inverse != 0, return_counts != 0);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unique_consecutive(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "return_inverse", "return_counts", "dim", nullptr};
  PyObject* input = nullptr;
  int return_inverse = 0;
  int return_counts = 0;
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|ppO:unique_consecutive",
          const_cast<char**>(keywords),
          &input,
          &return_inverse,
          &return_counts,
          &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unique_consecutive expected input Tensor");
    return nullptr;
  }
  try {
    const auto dim_value = optional_int64_from_py(dim, "dim");
    auto result = mtorch::unique_consecutive(tensor_ref(input), return_inverse != 0, return_counts != 0, dim_value);
    return unique_result_to_py(result, return_inverse != 0, return_counts != 0);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* tuple_from_int64s(const std::vector<int64_t>& values) {
  PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
  if (tuple == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    PyObject* item = PyLong_FromLongLong(values[i]);
    if (item == nullptr) {
      Py_DECREF(tuple);
      return nullptr;
    }
    PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), item);
  }
  return tuple;
}

PyObject* py_broadcast_shapes(PyObject*, PyObject* args) {
  try {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(static_cast<size_t>(PyTuple_GET_SIZE(args)));
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
      shapes.push_back(shape_from_object(PyTuple_GET_ITEM(args, i)));
    }
    return tuple_from_int64s(mtorch::broadcast_shapes(shapes));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_broadcast_tensors(PyObject*, PyObject* args) {
  try {
    std::vector<TensorPtr> tensors;
    tensors.reserve(static_cast<size_t>(PyTuple_GET_SIZE(args)));
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
      PyObject* item = PyTuple_GET_ITEM(args, i);
      if (!is_tensor(item)) {
        PyErr_SetString(PyExc_TypeError, "broadcast_tensors expected Tensor arguments");
        return nullptr;
      }
      tensors.push_back(tensor_ref(item));
    }
    return tuple_from_tensors(mtorch::broadcast_tensors(tensors));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_split(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"tensor", "split_size_or_sections", "dim", nullptr};
  PyObject* input = nullptr;
  PyObject* split_size_or_sections = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OO|L:split", const_cast<char**>(keywords), &input, &split_size_or_sections, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "split expected Tensor");
    return nullptr;
  }
  try {
    if (PyLong_Check(split_size_or_sections)) {
      return tuple_from_tensors(
          mtorch::split(tensor_ref(input), PyLong_AsLongLong(split_size_or_sections), dim));
    }
    return tuple_from_tensors(
        mtorch::split(tensor_ref(input), shape_from_object(split_size_or_sections), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_chunk(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "chunks", "dim", nullptr};
  PyObject* input = nullptr;
  long long chunks = 0;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OL|L:chunk", const_cast<char**>(keywords), &input, &chunks, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "chunk expected Tensor");
    return nullptr;
  }
  try {
    return tuple_from_tensors(mtorch::chunk(tensor_ref(input), chunks, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unbind(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", nullptr};
  PyObject* input = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:unbind", const_cast<char**>(keywords), &input, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unbind expected Tensor");
    return nullptr;
  }
  try {
    return tuple_from_tensors(mtorch::unbind(tensor_ref(input), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

std::vector<TensorPtr> tensor_list_from_py(PyObject* object) {
  if (PyList_Check(object) || PyTuple_Check(object)) {
    const bool is_tuple = PyTuple_Check(object);
    const Py_ssize_t length = is_tuple ? PyTuple_GET_SIZE(object) : PyList_GET_SIZE(object);
    std::vector<TensorPtr> tensors;
    tensors.reserve(static_cast<size_t>(length));
    for (Py_ssize_t i = 0; i < length; ++i) {
      PyObject* item = is_tuple ? PyTuple_GET_ITEM(object, i) : PyList_GET_ITEM(object, i);
      if (!is_tensor(item)) {
        throw std::invalid_argument("expected a sequence of tensors");
      }
      tensors.push_back(tensor_ref(item));
    }
    return tensors;
  }
  if (!object_is_sequence(object)) {
    throw std::invalid_argument("expected a sequence of tensors");
  }
  const Py_ssize_t length = PySequence_Size(object);
  std::vector<TensorPtr> tensors;
  tensors.reserve(static_cast<size_t>(length));
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::invalid_argument("could not read tensor sequence");
    }
    if (!is_tensor(item)) {
      Py_DECREF(item);
      throw std::invalid_argument("expected a sequence of tensors");
    }
    tensors.push_back(tensor_ref(item));
    Py_DECREF(item);
  }
  return tensors;
}

PyObject* py_cat(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"tensors", "dim", nullptr};
  PyObject* tensors = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:cat", const_cast<char**>(keywords), &tensors, &dim)) {
    return nullptr;
  }
  try {
    if (PyList_Check(tensors) || PyTuple_Check(tensors)) {
      const bool is_tuple = PyTuple_Check(tensors);
      const Py_ssize_t length = is_tuple ? PyTuple_GET_SIZE(tensors) : PyList_GET_SIZE(tensors);
      if (length == 2) {
        PyObject* left = is_tuple ? PyTuple_GET_ITEM(tensors, 0) : PyList_GET_ITEM(tensors, 0);
        PyObject* right = is_tuple ? PyTuple_GET_ITEM(tensors, 1) : PyList_GET_ITEM(tensors, 1);
        if (!is_tensor(left) || !is_tensor(right)) {
          throw std::invalid_argument("expected a sequence of tensors");
        }
        return wrap_tensor(mtorch::cat_pair(tensor_ref(left), tensor_ref(right), dim));
      }
    }
    return wrap_tensor(mtorch::cat(tensor_list_from_py(tensors), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_stack(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"tensors", "dim", nullptr};
  PyObject* tensors = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:stack", const_cast<char**>(keywords), &tensors, &dim)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::stack(tensor_list_from_py(tensors), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_hstack(PyObject*, PyObject* tensors) {
  try {
    return wrap_tensor(mtorch::hstack(tensor_list_from_py(tensors)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_vstack(PyObject*, PyObject* tensors) {
  try {
    return wrap_tensor(mtorch::vstack(tensor_list_from_py(tensors)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_dstack(PyObject*, PyObject* tensors) {
  try {
    return wrap_tensor(mtorch::dstack(tensor_list_from_py(tensors)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_column_stack(PyObject*, PyObject* tensors) {
  try {
    return wrap_tensor(mtorch::column_stack(tensor_list_from_py(tensors)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cartesian_prod(PyObject*, PyObject* args) {
  std::vector<TensorPtr> tensors;
  tensors.reserve(static_cast<size_t>(PyTuple_GET_SIZE(args)));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!is_tensor(item)) {
      PyErr_SetString(PyExc_TypeError, "cartesian_prod expected Tensor arguments");
      return nullptr;
    }
    tensors.push_back(tensor_ref(item));
  }
  try {
    return wrap_tensor(mtorch::cartesian_prod(tensors));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

void ensure_not_bool_matrix_contraction(const char* op, const TensorPtr& tensor) {
  if (tensor->dtype == ScalarType::Bool) {
    throw std::runtime_error(std::string(op) + " is not implemented for bool tensors");
  }
}

void ensure_same_dtype_matrix_contraction(const char* op, const TensorPtr& left, const TensorPtr& right) {
  if (left->dtype != right->dtype) {
    throw std::runtime_error(std::string("expected ") + op + " operands to have the same dtype");
  }
  ensure_not_bool_matrix_contraction(op, left);
}

void ensure_all_same_dtype_matrix_contraction(const char* op, const std::vector<TensorPtr>& tensors) {
  if (tensors.empty()) {
    return;
  }
  const ScalarType dtype = tensors.front()->dtype;
  for (const auto& tensor : tensors) {
    if (tensor->dtype != dtype) {
      throw std::runtime_error(std::string("expected ") + op + " operands to have the same dtype");
    }
  }
  ensure_not_bool_matrix_contraction(op, tensors.front());
}

void ensure_all_same_dtype_non_bool(const char* op, const std::vector<TensorPtr>& tensors) {
  if (tensors.empty()) {
    return;
  }
  const ScalarType dtype = tensors.front()->dtype;
  for (const auto& tensor : tensors) {
    if (tensor->dtype != dtype) {
      throw std::runtime_error(std::string("expected ") + op + " tensors to have the same dtype");
    }
  }
  if (dtype == ScalarType::Bool) {
    throw std::runtime_error(std::string(op) + " is not implemented for bool tensors");
  }
}

double default_rms_norm_eps(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Float64:
      return 2.22044604925031308085e-16;
    case ScalarType::Float16:
      return 9.765625e-04;
    case ScalarType::Float32:
    default:
      return 1.1920928955078125e-07;
  }
}

PyObject* py_matmul(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:matmul", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "matmul expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("matmul", left_tensor, right_tensor);
    return wrap_tensor(mtorch::matmul(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_mm(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:mm", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "mm expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    if (left_tensor->dim() != 2 || right_tensor->dim() != 2) {
      throw std::runtime_error("mm expects two 2-D tensors");
    }
    ensure_same_dtype_matrix_contraction("mm", left_tensor, right_tensor);
    return wrap_tensor(mtorch::matmul(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

std::string normalized_equation(PyObject* equation) {
  const char* raw = PyUnicode_AsUTF8(equation);
  if (raw == nullptr) {
    throw std::invalid_argument("einsum equation must be a string");
  }
  std::string result;
  for (const char* cursor = raw; *cursor != '\0'; ++cursor) {
    if (!std::isspace(static_cast<unsigned char>(*cursor))) {
      result.push_back(*cursor);
    }
  }
  return result;
}

TensorPtr einsum_attention_scores(const TensorPtr& left, const TensorPtr& right) {
  if (left->dim() != right->dim() || left->dim() < 2) {
    throw std::invalid_argument("einsum attention score pattern expects tensors with matching rank >= 2");
  }
  if (left->dtype != right->dtype) {
    throw std::runtime_error("expected einsum operands to have the same dtype for contraction");
  }
  if (left->dtype == ScalarType::Bool) {
    throw std::runtime_error("einsum contraction is not implemented for bool tensors");
  }
  const int64_t rank = left->dim();
  const int64_t depth = left->sizes[static_cast<size_t>(rank - 1)];
  if (right->sizes[static_cast<size_t>(rank - 1)] != depth) {
    throw std::invalid_argument("einsum attention score pattern has mismatched contraction dimensions");
  }
  bool matching_prefix_shape = true;
  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    if (left->sizes[static_cast<size_t>(dim)] != right->sizes[static_cast<size_t>(dim)]) {
      matching_prefix_shape = false;
      break;
    }
  }
#if defined(MTORCH_USE_ACCELERATE)
  if (!(mtorch::is_grad_enabled() && (left->requires_grad || right->requires_grad)) &&
      left->dtype == ScalarType::Float32 && right->dtype == ScalarType::Float32 &&
      left->is_contiguous() && right->is_contiguous() && matching_prefix_shape) {
    const int64_t query_length = left->sizes[static_cast<size_t>(rank - 2)];
    const int64_t key_length = right->sizes[static_cast<size_t>(rank - 2)];
    if (query_length <= 0 || key_length <= 0 || depth <= 0 || !accelerate_int_ok(query_length) ||
        !accelerate_int_ok(key_length) || !accelerate_int_ok(depth)) {
      return mtorch::matmul(left, mtorch::transpose(right, rank - 2, rank - 1));
    }
    const int64_t batch = left->numel() / (query_length * depth);
    std::vector<int64_t> out_shape(left->sizes.begin(), left->sizes.end() - 2);
    out_shape.push_back(query_length);
    out_shape.push_back(key_length);
    auto result = mtorch::zeros(out_shape, ScalarType::Float32, left->device);
    const float* left_data = reinterpret_cast<const float*>(left->storage->bytes.data()) + left->offset;
    const float* right_data = reinterpret_cast<const float*>(right->storage->bytes.data()) + right->offset;
    float* output_data = reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset;
    for (int64_t prefix = 0; prefix < batch; ++prefix) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasTrans,
          static_cast<int>(query_length),
          static_cast<int>(key_length),
          static_cast<int>(depth),
          1.0f,
          left_data + prefix * query_length * depth,
          static_cast<int>(depth),
          right_data + prefix * key_length * depth,
          static_cast<int>(depth),
          0.0f,
          output_data + prefix * query_length * key_length,
          static_cast<int>(key_length));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    }
    return result;
  }
#endif
  auto right_t = mtorch::transpose(right, rank - 2, rank - 1);
  return mtorch::matmul(left, right_t);
}

TensorPtr einsum_attention_values(const TensorPtr& left, const TensorPtr& right) {
  if (left->dim() != right->dim() || left->dim() < 2) {
    throw std::invalid_argument("einsum attention value pattern expects tensors with matching rank >= 2");
  }
  if (left->dtype != right->dtype) {
    throw std::runtime_error("expected einsum operands to have the same dtype for contraction");
  }
  if (left->dtype == ScalarType::Bool) {
    throw std::runtime_error("einsum contraction is not implemented for bool tensors");
  }
  const int64_t rank = left->dim();
  const int64_t key_length = left->sizes[static_cast<size_t>(rank - 1)];
  if (right->sizes[static_cast<size_t>(rank - 2)] != key_length) {
    throw std::invalid_argument("einsum attention value pattern has mismatched contraction dimensions");
  }
  bool matching_prefix_shape = true;
  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    if (left->sizes[static_cast<size_t>(dim)] != right->sizes[static_cast<size_t>(dim)]) {
      matching_prefix_shape = false;
      break;
    }
  }
#if defined(MTORCH_USE_ACCELERATE)
  if (!(mtorch::is_grad_enabled() && (left->requires_grad || right->requires_grad)) &&
      left->dtype == ScalarType::Float32 && right->dtype == ScalarType::Float32 &&
      left->is_contiguous() && right->is_contiguous() && matching_prefix_shape) {
    const int64_t query_length = left->sizes[static_cast<size_t>(rank - 2)];
    const int64_t depth = right->sizes[static_cast<size_t>(rank - 1)];
    if (query_length <= 0 || key_length <= 0 || depth <= 0 || !accelerate_int_ok(query_length) ||
        !accelerate_int_ok(key_length) || !accelerate_int_ok(depth)) {
      return mtorch::matmul(left, right);
    }
    const int64_t batch = left->numel() / (query_length * key_length);
    std::vector<int64_t> out_shape(left->sizes.begin(), left->sizes.end() - 2);
    out_shape.push_back(query_length);
    out_shape.push_back(depth);
    auto result = mtorch::zeros(out_shape, ScalarType::Float32, left->device);
    const float* left_data = reinterpret_cast<const float*>(left->storage->bytes.data()) + left->offset;
    const float* right_data = reinterpret_cast<const float*>(right->storage->bytes.data()) + right->offset;
    float* output_data = reinterpret_cast<float*>(result->storage->bytes.data()) + result->offset;
    for (int64_t prefix = 0; prefix < batch; ++prefix) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans,
          static_cast<int>(query_length),
          static_cast<int>(depth),
          static_cast<int>(key_length),
          1.0f,
          left_data + prefix * query_length * key_length,
          static_cast<int>(key_length),
          right_data + prefix * key_length * depth,
          static_cast<int>(depth),
          0.0f,
          output_data + prefix * query_length * depth,
          static_cast<int>(depth));
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    }
    return result;
  }
#endif
  return mtorch::matmul(left, right);
}

void ensure_einsum_contraction_supported(const TensorPtr& left, const TensorPtr& right) {
  if (left->dtype != right->dtype) {
    throw std::runtime_error("expected einsum operands to have the same dtype for contraction");
  }
  if (left->dtype == ScalarType::Bool) {
    throw std::runtime_error("einsum contraction is not implemented for bool tensors");
  }
}

struct SimpleBinaryEinsum {
  std::string left;
  std::string right;
  std::string output;
};

bool split_simple_binary_einsum(const std::string& text, SimpleBinaryEinsum& parsed) {
  if (text.find("...") != std::string::npos) {
    return false;
  }
  const auto arrow = text.find("->");
  if (arrow == std::string::npos || text.find("->", arrow + 2) != std::string::npos) {
    return false;
  }
  const std::string inputs = text.substr(0, arrow);
  const auto comma = inputs.find(',');
  if (comma == std::string::npos || inputs.find(',', comma + 1) != std::string::npos) {
    return false;
  }
  parsed.left = inputs.substr(0, comma);
  parsed.right = inputs.substr(comma + 1);
  parsed.output = text.substr(arrow + 2);
  return !parsed.left.empty() && !parsed.right.empty();
}

bool einsum_labels_are_unique(const std::string& labels) {
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels.find(labels[i], i + 1) != std::string::npos) {
      return false;
    }
  }
  return true;
}

bool einsum_labels_match_rank(const TensorPtr& tensor, const std::string& labels) {
  return tensor->dim() == static_cast<int64_t>(labels.size());
}

TensorPtr try_simple_binary_einsum_fast_path(const std::string& text, const TensorPtr& left, const TensorPtr& right) {
  SimpleBinaryEinsum equation;
  if (!split_simple_binary_einsum(text, equation) ||
      !einsum_labels_match_rank(left, equation.left) ||
      !einsum_labels_match_rank(right, equation.right) ||
      !einsum_labels_are_unique(equation.left) ||
      !einsum_labels_are_unique(equation.right) ||
      !einsum_labels_are_unique(equation.output)) {
    return nullptr;
  }
  if (equation.left.size() < 2 || equation.right.size() < 2 || equation.output.size() < 2) {
    return nullptr;
  }

  const std::string left_prefix = equation.left.substr(0, equation.left.size() - 2);
  const std::string right_prefix = equation.right.substr(0, equation.right.size() - 2);
  if (left_prefix != right_prefix) {
    return nullptr;
  }

  const char left_row = equation.left[equation.left.size() - 2];
  const char left_col = equation.left[equation.left.size() - 1];
  const char right_row = equation.right[equation.right.size() - 2];
  const char right_col = equation.right[equation.right.size() - 1];

  const std::string matmul_output = left_prefix + left_row + right_col;
  if (left_col == right_row && equation.output == matmul_output) {
    ensure_einsum_contraction_supported(left, right);
    return mtorch::matmul(left, right);
  }

  const std::string attention_scores_output = left_prefix + left_row + right_row;
  if (left_col == right_col && equation.output == attention_scores_output) {
    return einsum_attention_scores(left, right);
  }

  return nullptr;
}

PyObject* py_einsum(PyObject*, PyObject* args) {
  if (PyTuple_GET_SIZE(args) != 3) {
    PyErr_SetString(PyExc_NotImplementedError, "einsum currently supports equation plus exactly two Tensor operands");
    return nullptr;
  }
  PyObject* equation = PyTuple_GET_ITEM(args, 0);
  PyObject* left = PyTuple_GET_ITEM(args, 1);
  PyObject* right = PyTuple_GET_ITEM(args, 2);
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "einsum operands must be Tensors");
    return nullptr;
  }
  try {
    const std::string text = normalized_equation(equation);
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    if (auto result = try_simple_binary_einsum_fast_path(text, left_tensor, right_tensor)) {
      return wrap_tensor(result);
    }
    if (text == "ij,jk" || text == "ij,jk->ik" || text == "...ij,...jk->...ik") {
      ensure_einsum_contraction_supported(left_tensor, right_tensor);
      return wrap_tensor(mtorch::matmul(left_tensor, right_tensor));
    }
    if (text == "bij,bjk->bik" && left_tensor->dim() == 3 && right_tensor->dim() == 3 &&
        left_tensor->sizes[0] == right_tensor->sizes[0]) {
      ensure_einsum_contraction_supported(left_tensor, right_tensor);
      return wrap_tensor(mtorch::bmm(left_tensor, right_tensor));
    }
    if (text == "bid,bjd->bij" || text == "bhid,bhjd->bhij") {
      return wrap_tensor(einsum_attention_scores(left_tensor, right_tensor));
    }
    if (text == "bij,bjd->bid" || text == "bhij,bhjd->bhid") {
      return wrap_tensor(einsum_attention_values(left_tensor, right_tensor));
    }
    PyErr_SetString(PyExc_NotImplementedError, "einsum equation is not implemented");
    return nullptr;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_bmm(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:bmm", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "bmm expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("bmm", left_tensor, right_tensor);
    return wrap_tensor(mtorch::bmm(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_addmm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "mat1", "mat2", "beta", "alpha", nullptr};
  PyObject* input = nullptr;
  PyObject* mat1 = nullptr;
  PyObject* mat2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|dd:addmm", const_cast<char**>(keywords), &input, &mat1, &mat2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(mat1) || !is_tensor(mat2)) {
    PyErr_SetString(PyExc_TypeError, "addmm expected tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto mat1_tensor = tensor_ref(mat1);
    const auto mat2_tensor = tensor_ref(mat2);
    ensure_all_same_dtype_matrix_contraction("addmm", {input_tensor, mat1_tensor, mat2_tensor});
    return wrap_tensor(mtorch::addmm(input_tensor, mat1_tensor, mat2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_addmv(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "mat", "vec", "beta", "alpha", nullptr};
  PyObject* input = nullptr;
  PyObject* mat = nullptr;
  PyObject* vec = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|dd:addmv", const_cast<char**>(keywords), &input, &mat, &vec, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(mat) || !is_tensor(vec)) {
    PyErr_SetString(PyExc_TypeError, "addmv expected tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto mat_tensor = tensor_ref(mat);
    const auto vec_tensor = tensor_ref(vec);
    ensure_same_dtype_matrix_contraction("addmv", mat_tensor, vec_tensor);
    return wrap_tensor(mtorch::addmv(input_tensor, mat_tensor, vec_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_addr(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "vec1", "vec2", "beta", "alpha", nullptr};
  PyObject* input = nullptr;
  PyObject* vec1 = nullptr;
  PyObject* vec2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|dd:addr", const_cast<char**>(keywords), &input, &vec1, &vec2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(vec1) || !is_tensor(vec2)) {
    PyErr_SetString(PyExc_TypeError, "addr expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::addr(tensor_ref(input), tensor_ref(vec1), tensor_ref(vec2), beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_baddbmm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "batch1", "batch2", "beta", "alpha", nullptr};
  PyObject* input = nullptr;
  PyObject* batch1 = nullptr;
  PyObject* batch2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|dd:baddbmm", const_cast<char**>(keywords), &input, &batch1, &batch2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(batch1) || !is_tensor(batch2)) {
    PyErr_SetString(PyExc_TypeError, "baddbmm expected tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto batch1_tensor = tensor_ref(batch1);
    const auto batch2_tensor = tensor_ref(batch2);
    ensure_all_same_dtype_matrix_contraction("baddbmm", {input_tensor, batch1_tensor, batch2_tensor});
    return wrap_tensor(mtorch::baddbmm(input_tensor, batch1_tensor, batch2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_addbmm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "batch1", "batch2", "beta", "alpha", nullptr};
  PyObject* input = nullptr;
  PyObject* batch1 = nullptr;
  PyObject* batch2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|dd:addbmm", const_cast<char**>(keywords), &input, &batch1, &batch2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(batch1) || !is_tensor(batch2)) {
    PyErr_SetString(PyExc_TypeError, "addbmm expected tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto batch1_tensor = tensor_ref(batch1);
    const auto batch2_tensor = tensor_ref(batch2);
    ensure_all_same_dtype_matrix_contraction("addbmm", {input_tensor, batch1_tensor, batch2_tensor});
    return wrap_tensor(mtorch::addbmm(input_tensor, batch1_tensor, batch2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_vdot(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:vdot", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "vdot expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("vdot", left_tensor, right_tensor);
    return wrap_tensor(mtorch::vdot(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_inner(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:inner", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "inner expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    if (!left_tensor->is_scalar() && !right_tensor->is_scalar()) {
      ensure_same_dtype_matrix_contraction("inner", left_tensor, right_tensor);
    }
    return wrap_tensor(mtorch::inner(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_chain_matmul(PyObject*, PyObject* args) {
  const Py_ssize_t count = PyTuple_Size(args);
  std::vector<TensorPtr> matrices;
  matrices.reserve(static_cast<size_t>(std::max<Py_ssize_t>(count, 0)));
  for (Py_ssize_t i = 0; i < count; ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    if (!is_tensor(item)) {
      PyErr_SetString(PyExc_TypeError, "chain_matmul expected Tensor arguments");
      return nullptr;
    }
    matrices.push_back(tensor_ref(item));
  }
  try {
    ensure_all_same_dtype_matrix_contraction("chain_matmul", matrices);
    return wrap_tensor(mtorch::chain_matmul(matrices));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_matrix_power(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long n = 0;
  if (!PyArg_ParseTuple(args, "OL:matrix_power", &input, &n)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "matrix_power expected a tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::matrix_power(tensor_ref(input), static_cast<int64_t>(n)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

bool parse_dim_sequence(PyObject* object, const char* name, std::vector<int64_t>& dims) {
  PyObject* sequence = PySequence_Fast(object, name);
  if (!sequence) {
    return false;
  }
  const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
  dims.reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(sequence, i);
    if (!PyLong_Check(item)) {
      Py_DECREF(sequence);
      PyErr_SetString(PyExc_TypeError, "tensordot dims entries must be integers");
      return false;
    }
    const long long value = PyLong_AsLongLong(item);
    if (PyErr_Occurred()) {
      Py_DECREF(sequence);
      return false;
    }
    dims.push_back(static_cast<int64_t>(value));
  }
  Py_DECREF(sequence);
  return true;
}

bool parse_tensordot_dims(
    PyObject* dims_object,
    int64_t left_rank,
    int64_t right_rank,
    std::vector<int64_t>& left_dims,
    std::vector<int64_t>& right_dims) {
  if (!dims_object || dims_object == Py_None || PyLong_Check(dims_object)) {
    long long count = 2;
    if (dims_object && dims_object != Py_None) {
      count = PyLong_AsLongLong(dims_object);
      if (PyErr_Occurred()) {
        return false;
      }
    }
    if (count < 0 || count > left_rank || count > right_rank) {
      PyErr_SetString(PyExc_ValueError, "tensordot dims is out of range");
      return false;
    }
    for (long long i = 0; i < count; ++i) {
      left_dims.push_back(left_rank - count + i);
      right_dims.push_back(i);
    }
    return true;
  }

  PyObject* pair = PySequence_Fast(dims_object, "tensordot dims must be an int or a pair of dimension lists");
  if (!pair) {
    return false;
  }
  if (PySequence_Fast_GET_SIZE(pair) != 2) {
    Py_DECREF(pair);
    PyErr_SetString(PyExc_ValueError, "tensordot dims pair must have length 2");
    return false;
  }
  PyObject* first = PySequence_Fast_GET_ITEM(pair, 0);
  PyObject* second = PySequence_Fast_GET_ITEM(pair, 1);
  const bool ok = parse_dim_sequence(first, "tensordot dims_self must be a sequence of integers", left_dims) &&
      parse_dim_sequence(second, "tensordot dims_other must be a sequence of integers", right_dims);
  Py_DECREF(pair);
  return ok;
}

PyObject* py_tensordot(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"a", "b", "dims", nullptr};
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  PyObject* dims = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:tensordot", const_cast<char**>(keywords), &left, &right, &dims)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "tensordot expected tensors");
    return nullptr;
  }
  std::vector<int64_t> left_dims;
  std::vector<int64_t> right_dims;
  if (!parse_tensordot_dims(dims, tensor_ref(left)->dim(), tensor_ref(right)->dim(), left_dims, right_dims)) {
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("tensordot", left_tensor, right_tensor);
    return wrap_tensor(mtorch::tensordot(left_tensor, right_tensor, left_dims, right_dims));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_kron(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:kron", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "kron expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::kron(tensor_ref(left), tensor_ref(right)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_dot(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:dot", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "dot expected tensors");
    return nullptr;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("dot", left_tensor, right_tensor);
    return wrap_tensor(mtorch::dot(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_mv(PyObject*, PyObject* args) {
  PyObject* matrix = nullptr;
  PyObject* vector = nullptr;
  if (!PyArg_ParseTuple(args, "OO:mv", &matrix, &vector)) {
    return nullptr;
  }
  if (!is_tensor(matrix) || !is_tensor(vector)) {
    PyErr_SetString(PyExc_TypeError, "mv expected tensors");
    return nullptr;
  }
  try {
    const auto matrix_tensor = tensor_ref(matrix);
    const auto vector_tensor = tensor_ref(vector);
    ensure_same_dtype_matrix_contraction("mv", matrix_tensor, vector_tensor);
    return wrap_tensor(mtorch::mv(matrix_tensor, vector_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_outer(PyObject*, PyObject* args) {
  PyObject* left = nullptr;
  PyObject* right = nullptr;
  if (!PyArg_ParseTuple(args, "OO:outer", &left, &right)) {
    return nullptr;
  }
  if (!is_tensor(left) || !is_tensor(right)) {
    PyErr_SetString(PyExc_TypeError, "outer expected Tensor arguments");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::outer(tensor_ref(left), tensor_ref(right)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_linear(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "weight", "bias", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O:linear", const_cast<char**>(keywords), &input, &weight, &bias)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "linear expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    if (bias == Py_None) {
      ensure_same_dtype_matrix_contraction("linear", input_tensor, weight_tensor);
      return wrap_tensor(mtorch::linear(input_tensor, weight_tensor, nullptr));
    }
    const auto bias_tensor = tensor_ref(bias);
    ensure_all_same_dtype_matrix_contraction("linear", {input_tensor, weight_tensor, bias_tensor});
    return wrap_tensor(mtorch::linear(input_tensor, weight_tensor, bias_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv1d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "weight", "bias", "stride", "padding", "dilation", "groups", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* dilation = Py_None;
  long long groups = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOL:conv1d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &dilation,
          &groups)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv1d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv1d", dtype_tensors);
    return wrap_tensor(mtorch::conv1d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0} : shape_from_object(padding),
        dilation == Py_None ? std::vector<int64_t>{1} : shape_from_object(dilation),
        static_cast<int64_t>(groups)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv_transpose1d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* output_padding = Py_None;
  long long groups = 1;
  PyObject* dilation = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOLO:conv_transpose1d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &output_padding,
          &groups,
          &dilation)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv_transpose1d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv_transpose1d", dtype_tensors);
    return wrap_tensor(mtorch::conv_transpose1d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0} : shape_from_object(padding),
        output_padding == Py_None ? std::vector<int64_t>{0} : shape_from_object(output_padding),
        static_cast<int64_t>(groups),
        dilation == Py_None ? std::vector<int64_t>{1} : shape_from_object(dilation)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv2d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "weight", "bias", "stride", "padding", "dilation", "groups", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* dilation = Py_None;
  long long groups = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOL:conv2d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &dilation,
          &groups)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv2d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv2d", dtype_tensors);
    return wrap_tensor(mtorch::conv2d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        dilation == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(dilation),
        static_cast<int64_t>(groups)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv3d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "weight", "bias", "stride", "padding", "dilation", "groups", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* dilation = Py_None;
  long long groups = 1;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOL:conv3d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &dilation,
          &groups)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv3d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv3d", dtype_tensors);
    return wrap_tensor(mtorch::conv3d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1, 1, 1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0, 0} : shape_from_object(padding),
        dilation == Py_None ? std::vector<int64_t>{1, 1, 1} : shape_from_object(dilation),
        static_cast<int64_t>(groups)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv_transpose2d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* output_padding = Py_None;
  long long groups = 1;
  PyObject* dilation = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOLO:conv_transpose2d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &output_padding,
          &groups,
          &dilation)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv_transpose2d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv_transpose2d", dtype_tensors);
    return wrap_tensor(mtorch::conv_transpose2d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        output_padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(output_padding),
        static_cast<int64_t>(groups),
        dilation == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(dilation)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_conv_transpose3d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* bias = Py_None;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* output_padding = Py_None;
  long long groups = 1;
  PyObject* dilation = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOLO:conv_transpose3d",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &bias,
          &stride,
          &padding,
          &output_padding,
          &groups,
          &dilation)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight) || (bias != Py_None && !is_tensor(bias))) {
    PyErr_SetString(PyExc_TypeError, "conv_transpose3d expected input, weight, and optional bias tensors");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const auto weight_tensor = tensor_ref(weight);
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor, weight_tensor};
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("conv_transpose3d", dtype_tensors);
    return wrap_tensor(mtorch::conv_transpose3d(
        input_tensor,
        weight_tensor,
        bias_tensor,
        stride == Py_None ? std::vector<int64_t>{1, 1, 1} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0, 0} : shape_from_object(padding),
        output_padding == Py_None ? std::vector<int64_t>{0, 0, 0} : shape_from_object(output_padding),
        static_cast<int64_t>(groups),
        dilation == Py_None ? std::vector<int64_t>{1, 1, 1} : shape_from_object(dilation)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_max_pool1d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices", nullptr};
  PyObject* input = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* dilation = Py_None;
  PyObject* ceil_mode = Py_False;
  PyObject* return_indices = Py_False;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOO:max_pool1d",
          const_cast<char**>(keywords),
          &input,
          &kernel_size,
          &stride,
          &padding,
          &dilation,
          &ceil_mode,
          &return_indices)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "max_pool1d expected input Tensor");
    return nullptr;
  }
  const int ceil = PyObject_IsTrue(ceil_mode);
  if (ceil < 0) {
    return nullptr;
  }
  const int wants_indices = PyObject_IsTrue(return_indices);
  if (wants_indices < 0) {
    return nullptr;
  }
  try {
    if (wants_indices != 0) {
      throw NotImplementedException("max_pool1d return_indices=True is not implemented yet");
    }
    return wrap_tensor(mtorch::max_pool1d(
        tensor_ref(input),
        shape_from_object(kernel_size),
        stride == Py_None ? std::vector<int64_t>{} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0} : shape_from_object(padding),
        dilation == Py_None ? std::vector<int64_t>{1} : shape_from_object(dilation),
        ceil != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_avg_pool1d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override", nullptr};
  PyObject* input = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* ceil_mode = Py_False;
  PyObject* count_include_pad = Py_True;
  PyObject* divisor_override = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOO:avg_pool1d",
          const_cast<char**>(keywords),
          &input,
          &kernel_size,
          &stride,
          &padding,
          &ceil_mode,
          &count_include_pad,
          &divisor_override)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "avg_pool1d expected input Tensor");
    return nullptr;
  }
  const int ceil = PyObject_IsTrue(ceil_mode);
  if (ceil < 0) {
    return nullptr;
  }
  const int include_pad = PyObject_IsTrue(count_include_pad);
  if (include_pad < 0) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::avg_pool1d(
        tensor_ref(input),
        shape_from_object(kernel_size),
        stride == Py_None ? std::vector<int64_t>{} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0} : shape_from_object(padding),
        ceil != 0,
        include_pad != 0,
        optional_int64_from_py(divisor_override, "divisor_override")));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_max_pool2d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices", nullptr};
  PyObject* input = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* dilation = Py_None;
  PyObject* ceil_mode = Py_False;
  PyObject* return_indices = Py_False;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOO:max_pool2d",
          const_cast<char**>(keywords),
          &input,
          &kernel_size,
          &stride,
          &padding,
          &dilation,
          &ceil_mode,
          &return_indices)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "max_pool2d expected input Tensor");
    return nullptr;
  }
  const int ceil = PyObject_IsTrue(ceil_mode);
  if (ceil < 0) {
    return nullptr;
  }
  const int wants_indices = PyObject_IsTrue(return_indices);
  if (wants_indices < 0) {
    return nullptr;
  }
  try {
    if (wants_indices != 0) {
      throw NotImplementedException("max_pool2d return_indices=True is not implemented yet");
    }
    return wrap_tensor(mtorch::max_pool2d(
        tensor_ref(input),
        shape_from_object(kernel_size),
        stride == Py_None ? std::vector<int64_t>{} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        dilation == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(dilation),
        ceil != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_avg_pool2d(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override", nullptr};
  PyObject* input = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* stride = Py_None;
  PyObject* padding = Py_None;
  PyObject* ceil_mode = Py_False;
  PyObject* count_include_pad = Py_True;
  PyObject* divisor_override = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOOO:avg_pool2d",
          const_cast<char**>(keywords),
          &input,
          &kernel_size,
          &stride,
          &padding,
          &ceil_mode,
          &count_include_pad,
          &divisor_override)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "avg_pool2d expected input Tensor");
    return nullptr;
  }
  const int ceil = PyObject_IsTrue(ceil_mode);
  if (ceil < 0) {
    return nullptr;
  }
  const int include_pad = PyObject_IsTrue(count_include_pad);
  if (include_pad < 0) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::avg_pool2d(
        tensor_ref(input),
        shape_from_object(kernel_size),
        stride == Py_None ? std::vector<int64_t>{} : shape_from_object(stride),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        ceil != 0,
        include_pad != 0,
        optional_int64_from_py(divisor_override, "divisor_override")));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_unfold(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "kernel_size", "dilation", "padding", "stride", nullptr};
  PyObject* input = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* dilation = Py_None;
  PyObject* padding = Py_None;
  PyObject* stride = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOO:unfold",
          const_cast<char**>(keywords),
          &input,
          &kernel_size,
          &dilation,
          &padding,
          &stride)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "unfold expected input Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unfold2d(
        tensor_ref(input),
        shape_from_object(kernel_size),
        dilation == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(dilation),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        stride == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(stride)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_fold(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "output_size", "kernel_size", "dilation", "padding", "stride", nullptr};
  PyObject* input = nullptr;
  PyObject* output_size = nullptr;
  PyObject* kernel_size = nullptr;
  PyObject* dilation = Py_None;
  PyObject* padding = Py_None;
  PyObject* stride = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OOO|OOO:fold",
          const_cast<char**>(keywords),
          &input,
          &output_size,
          &kernel_size,
          &dilation,
          &padding,
          &stride)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "fold expected input Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::fold2d(
        tensor_ref(input),
        shape_from_object(output_size),
        shape_from_object(kernel_size),
        dilation == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(dilation),
        padding == Py_None ? std::vector<int64_t>{0, 0} : shape_from_object(padding),
        stride == Py_None ? std::vector<int64_t>{1, 1} : shape_from_object(stride)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_scaled_dot_product_attention(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa", nullptr};
  PyObject* query = nullptr;
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  PyObject* attn_mask = Py_None;
  double dropout_p = 0.0;
  PyObject* is_causal = Py_False;
  PyObject* scale = Py_None;
  PyObject* enable_gqa = Py_False;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OOO|OdOOO:scaled_dot_product_attention",
          const_cast<char**>(keywords),
          &query,
          &key,
          &value,
          &attn_mask,
          &dropout_p,
          &is_causal,
          &scale,
          &enable_gqa)) {
    return nullptr;
  }
  if (!is_tensor(query) || !is_tensor(key) || !is_tensor(value) || (attn_mask != Py_None && !is_tensor(attn_mask))) {
    PyErr_SetString(PyExc_TypeError, "scaled_dot_product_attention expected query, key, value, and optional attn_mask tensors");
    return nullptr;
  }
  const int causal = PyObject_IsTrue(is_causal);
  if (causal < 0) {
    return nullptr;
  }
  const int gqa = PyObject_IsTrue(enable_gqa);
  if (gqa < 0) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::scaled_dot_product_attention(
        tensor_ref(query),
        tensor_ref(key),
        tensor_ref(value),
        attn_mask == Py_None ? nullptr : tensor_ref(attn_mask),
        dropout_p,
        causal != 0,
        optional_scalar_from_py(scale),
        gqa != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_layer_norm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "normalized_shape", "weight", "bias", "eps", nullptr};
  PyObject* input = nullptr;
  PyObject* normalized_shape = nullptr;
  PyObject* weight = Py_None;
  PyObject* bias = Py_None;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOd:layer_norm",
          const_cast<char**>(keywords),
          &input,
          &normalized_shape,
          &weight,
          &bias,
          &eps)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "layer_norm expected Tensor input");
    return nullptr;
  }
  if (weight != Py_None && !is_tensor(weight)) {
    PyErr_SetString(PyExc_TypeError, "layer_norm weight must be a Tensor or None");
    return nullptr;
  }
  if (bias != Py_None && !is_tensor(bias)) {
    PyErr_SetString(PyExc_TypeError, "layer_norm bias must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    TensorPtr weight_tensor;
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor};
    if (weight != Py_None) {
      weight_tensor = tensor_ref(weight);
      dtype_tensors.push_back(weight_tensor);
    }
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("layer_norm", dtype_tensors);
    return wrap_tensor(mtorch::layer_norm(
        input_tensor,
        shape_from_object(normalized_shape),
        weight_tensor,
        bias_tensor,
        eps));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_rms_norm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "normalized_shape", "weight", "eps", nullptr};
  PyObject* input = nullptr;
  PyObject* normalized_shape = nullptr;
  PyObject* weight = Py_None;
  PyObject* eps_object = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OO:rms_norm",
          const_cast<char**>(keywords),
          &input,
          &normalized_shape,
          &weight,
          &eps_object)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "rms_norm expected Tensor input");
    return nullptr;
  }
  if (weight != Py_None && !is_tensor(weight)) {
    PyErr_SetString(PyExc_TypeError, "rms_norm weight must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    const double eps = eps_object == Py_None ? default_rms_norm_eps(input_tensor->dtype) : scalar_from_py(eps_object);
    TensorPtr weight_tensor;
    if (weight != Py_None) {
      weight_tensor = tensor_ref(weight);
    }
    return wrap_tensor(mtorch::rms_norm(
        input_tensor,
        shape_from_object(normalized_shape),
        weight_tensor,
        eps));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_batch_norm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "running_mean", "running_var", "weight", "bias", "training", "momentum", "eps", nullptr};
  PyObject* input = nullptr;
  PyObject* running_mean = Py_None;
  PyObject* running_var = Py_None;
  PyObject* weight = Py_None;
  PyObject* bias = Py_None;
  int training = 0;
  double momentum = 0.1;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OOOOpdd:batch_norm",
          const_cast<char**>(keywords),
          &input,
          &running_mean,
          &running_var,
          &weight,
          &bias,
          &training,
          &momentum,
          &eps)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "batch_norm expected Tensor input");
    return nullptr;
  }
  if (running_mean != Py_None && !is_tensor(running_mean)) {
    PyErr_SetString(PyExc_TypeError, "batch_norm running_mean must be a Tensor or None");
    return nullptr;
  }
  if (running_var != Py_None && !is_tensor(running_var)) {
    PyErr_SetString(PyExc_TypeError, "batch_norm running_var must be a Tensor or None");
    return nullptr;
  }
  if (weight != Py_None && !is_tensor(weight)) {
    PyErr_SetString(PyExc_TypeError, "batch_norm weight must be a Tensor or None");
    return nullptr;
  }
  if (bias != Py_None && !is_tensor(bias)) {
    PyErr_SetString(PyExc_TypeError, "batch_norm bias must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    TensorPtr running_mean_tensor;
    TensorPtr running_var_tensor;
    TensorPtr weight_tensor;
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor};
    if (running_mean != Py_None) {
      running_mean_tensor = tensor_ref(running_mean);
      dtype_tensors.push_back(running_mean_tensor);
    }
    if (running_var != Py_None) {
      running_var_tensor = tensor_ref(running_var);
      dtype_tensors.push_back(running_var_tensor);
    }
    if (weight != Py_None) {
      weight_tensor = tensor_ref(weight);
      dtype_tensors.push_back(weight_tensor);
    }
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("batch_norm", dtype_tensors);
    return wrap_tensor(mtorch::batch_norm(
        input_tensor,
        running_mean_tensor,
        running_var_tensor,
        weight_tensor,
        bias_tensor,
        training != 0,
        momentum,
        eps));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_group_norm(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "num_groups", "weight", "bias", "eps", nullptr};
  PyObject* input = nullptr;
  long long num_groups = 0;
  PyObject* weight = Py_None;
  PyObject* bias = Py_None;
  double eps = 1e-5;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OL|OOd:group_norm", const_cast<char**>(keywords), &input, &num_groups, &weight, &bias, &eps)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "group_norm expected Tensor input");
    return nullptr;
  }
  if (weight != Py_None && !is_tensor(weight)) {
    PyErr_SetString(PyExc_TypeError, "group_norm weight must be a Tensor or None");
    return nullptr;
  }
  if (bias != Py_None && !is_tensor(bias)) {
    PyErr_SetString(PyExc_TypeError, "group_norm bias must be a Tensor or None");
    return nullptr;
  }
  try {
    const auto input_tensor = tensor_ref(input);
    TensorPtr weight_tensor;
    TensorPtr bias_tensor;
    std::vector<TensorPtr> dtype_tensors = {input_tensor};
    if (weight != Py_None) {
      weight_tensor = tensor_ref(weight);
      dtype_tensors.push_back(weight_tensor);
    }
    if (bias != Py_None) {
      bias_tensor = tensor_ref(bias);
      dtype_tensors.push_back(bias_tensor);
    }
    ensure_all_same_dtype_non_bool("group_norm", dtype_tensors);
    return wrap_tensor(mtorch::group_norm(
        input_tensor,
        num_groups,
        weight_tensor,
        bias_tensor,
        eps));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_embedding(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "weight", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse", nullptr};
  PyObject* input = nullptr;
  PyObject* weight = nullptr;
  PyObject* padding_idx = Py_None;
  PyObject* max_norm = Py_None;
  double norm_type = 2.0;
  int scale_grad_by_freq = 0;
  int sparse = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOdpp:embedding",
          const_cast<char**>(keywords),
          &input,
          &weight,
          &padding_idx,
          &max_norm,
          &norm_type,
          &scale_grad_by_freq,
          &sparse)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(weight)) {
    PyErr_SetString(PyExc_TypeError, "embedding expected input and weight Tensors");
    return nullptr;
  }
  if (sparse) {
    PyErr_SetString(PyExc_NotImplementedError, "embedding sparse gradients are not implemented yet");
    return nullptr;
  }
  try {
    std::optional<int64_t> padding;
    if (padding_idx != Py_None) {
      padding = PyLong_AsLongLong(padding_idx);
      if (PyErr_Occurred()) {
        return nullptr;
      }
    }
    std::optional<double> max_norm_value;
    if (max_norm != Py_None) {
      max_norm_value = scalar_from_py(max_norm);
    }
    return wrap_tensor(mtorch::embedding(
        tensor_ref(weight),
        tensor_ref(input),
        padding,
        max_norm_value,
        norm_type,
        scale_grad_by_freq != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_dropout(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "p", "training", "inplace", nullptr};
  PyObject* input = nullptr;
  double p = 0.5;
  int training = 1;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|dpp:dropout", const_cast<char**>(keywords), &input, &p, &training, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "dropout expected input Tensor");
    return nullptr;
  }
  if (inplace && training && p != 0.0) {
    PyErr_SetString(PyExc_NotImplementedError, "dropout inplace=True is not implemented for training=True");
    return nullptr;
  }
  try {
    return wrap_tensor(dropout_tensor(tensor_ref(input), p, training != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_mse_loss(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "target", "reduction", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  const char* reduction = "mean";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|s:mse_loss", const_cast<char**>(keywords), &input, &target, &reduction)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "mse_loss expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::mse_loss(tensor_ref(input), tensor_ref(target), reduction));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_l1_loss(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "target", "reduction", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  const char* reduction = "mean";
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|s:l1_loss", const_cast<char**>(keywords), &input, &target, &reduction)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "l1_loss expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::l1_loss(tensor_ref(input), tensor_ref(target), reduction));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_nll_loss(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "target", "weight", "size_average", "ignore_index", "reduce", "reduction", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  PyObject* weight = Py_None;
  PyObject* size_average = Py_None;
  long long ignore_index = -100;
  PyObject* reduce = Py_None;
  const char* reduction = "mean";
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOLOs:nll_loss",
          const_cast<char**>(keywords),
          &input,
          &target,
          &weight,
          &size_average,
          &ignore_index,
          &reduce,
          &reduction)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "nll_loss expected input and target tensors");
    return nullptr;
  }
  TensorPtr weight_tensor = nullptr;
  if (weight != Py_None) {
    if (!is_tensor(weight)) {
      PyErr_SetString(PyExc_TypeError, "nll_loss weight must be a Tensor");
      return nullptr;
    }
    weight_tensor = tensor_ref(weight);
  }
  std::string reduction_text = reduction;
  if (reduce != Py_None) {
    const int reduce_truth = PyObject_IsTrue(reduce);
    if (reduce_truth < 0) {
      return nullptr;
    }
    if (!reduce_truth) {
      reduction_text = "none";
    } else if (size_average != Py_None) {
      const int average_truth = PyObject_IsTrue(size_average);
      if (average_truth < 0) {
        return nullptr;
      }
      reduction_text = average_truth ? "mean" : "sum";
    }
  }
  try {
    return wrap_tensor(mtorch::nll_loss(
        tensor_ref(input),
        tensor_ref(target),
        reduction_text,
        static_cast<int64_t>(ignore_index),
        weight_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_cross_entropy(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "target", "weight", "size_average", "ignore_index", "reduce", "reduction", "label_smoothing", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  PyObject* weight = Py_None;
  PyObject* size_average = Py_None;
  int ignore_index = -100;
  PyObject* reduce = Py_None;
  const char* reduction = "mean";
  double label_smoothing = 0.0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOiOsd:cross_entropy",
          const_cast<char**>(keywords),
          &input,
          &target,
          &weight,
          &size_average,
          &ignore_index,
          &reduce,
          &reduction,
          &label_smoothing)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "cross_entropy expected input and target tensors");
    return nullptr;
  }
  TensorPtr weight_tensor = nullptr;
  if (weight != Py_None) {
    if (!is_tensor(weight)) {
      PyErr_SetString(PyExc_TypeError, "cross_entropy weight must be a Tensor");
      return nullptr;
    }
    weight_tensor = tensor_ref(weight);
  }
  std::string reduction_text = reduction;
  if (reduce != Py_None) {
    const int reduce_truth = PyObject_IsTrue(reduce);
    if (reduce_truth < 0) {
      return nullptr;
    }
    if (!reduce_truth) {
      reduction_text = "none";
    } else if (size_average != Py_None) {
      const int average_truth = PyObject_IsTrue(size_average);
      if (average_truth < 0) {
        return nullptr;
      }
      reduction_text = average_truth ? "mean" : "sum";
    }
  }
  try {
    return wrap_tensor(
        mtorch::cross_entropy_loss(
            tensor_ref(input),
            tensor_ref(target),
            reduction_text,
            ignore_index,
            label_smoothing,
            weight_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_binary_cross_entropy(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "target", "weight", "size_average", "reduce", "reduction", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  PyObject* weight = Py_None;
  PyObject* size_average = Py_None;
  PyObject* reduce = Py_None;
  const char* reduction = "mean";
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOs:binary_cross_entropy",
          const_cast<char**>(keywords),
          &input,
          &target,
          &weight,
          &size_average,
          &reduce,
          &reduction)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "binary_cross_entropy expected input and target tensors");
    return nullptr;
  }
  TensorPtr weight_tensor = nullptr;
  if (weight != Py_None) {
    if (!is_tensor(weight)) {
      PyErr_SetString(PyExc_TypeError, "binary_cross_entropy weight must be a Tensor");
      return nullptr;
    }
    weight_tensor = tensor_ref(weight);
  }
  std::string reduction_text = reduction;
  if (reduce != Py_None) {
    const int reduce_truth = PyObject_IsTrue(reduce);
    if (reduce_truth < 0) {
      return nullptr;
    }
    if (!reduce_truth) {
      reduction_text = "none";
    } else if (size_average != Py_None) {
      const int average_truth = PyObject_IsTrue(size_average);
      if (average_truth < 0) {
        return nullptr;
      }
      reduction_text = average_truth ? "mean" : "sum";
    }
  }
  try {
    return wrap_tensor(mtorch::binary_cross_entropy_loss(tensor_ref(input), tensor_ref(target), reduction_text, weight_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_binary_cross_entropy_with_logits(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {
      "input", "target", "weight", "size_average", "reduce", "reduction", "pos_weight", nullptr};
  PyObject* input = nullptr;
  PyObject* target = nullptr;
  PyObject* weight = Py_None;
  PyObject* size_average = Py_None;
  PyObject* reduce = Py_None;
  const char* reduction = "mean";
  PyObject* pos_weight = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|OOOsO:binary_cross_entropy_with_logits",
          const_cast<char**>(keywords),
          &input,
          &target,
          &weight,
          &size_average,
          &reduce,
          &reduction,
          &pos_weight)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(target)) {
    PyErr_SetString(PyExc_TypeError, "binary_cross_entropy_with_logits expected input and target tensors");
    return nullptr;
  }
  TensorPtr weight_tensor = nullptr;
  if (weight != Py_None) {
    if (!is_tensor(weight)) {
      PyErr_SetString(PyExc_TypeError, "binary_cross_entropy_with_logits weight must be a Tensor");
      return nullptr;
    }
    weight_tensor = tensor_ref(weight);
  }
  TensorPtr pos_weight_tensor = nullptr;
  if (pos_weight != Py_None) {
    if (!is_tensor(pos_weight)) {
      PyErr_SetString(PyExc_TypeError, "binary_cross_entropy_with_logits pos_weight must be a Tensor");
      return nullptr;
    }
    pos_weight_tensor = tensor_ref(pos_weight);
  }
  std::string reduction_text = reduction;
  if (reduce != Py_None) {
    const int reduce_truth = PyObject_IsTrue(reduce);
    if (reduce_truth < 0) {
      return nullptr;
    }
    if (!reduce_truth) {
      reduction_text = "none";
    } else if (size_average != Py_None) {
      const int average_truth = PyObject_IsTrue(size_average);
      if (average_truth < 0) {
        return nullptr;
      }
      reduction_text = average_truth ? "mean" : "sum";
    }
  }
  try {
    return wrap_tensor(mtorch::binary_cross_entropy_with_logits_loss(
        tensor_ref(input),
        tensor_ref(target),
        reduction_text,
        weight_tensor,
        pos_weight_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_where(PyObject*, PyObject* args) {
  try {
    const Py_ssize_t count = PyTuple_GET_SIZE(args);
    if (count == 1) {
      PyObject* condition = PyTuple_GET_ITEM(args, 0);
      if (!is_tensor(condition)) {
        throw TypeErrorException("where expected condition tensor");
      }
      return tuple_from_tensors(mtorch::nonzero_tuple(tensor_ref(condition)));
    }
    if (count != 3) {
      throw TypeErrorException("where expected either condition or condition, input, other");
    }

    PyObject* condition = PyTuple_GET_ITEM(args, 0);
    PyObject* left = PyTuple_GET_ITEM(args, 1);
    PyObject* right = PyTuple_GET_ITEM(args, 2);
    if (!is_tensor(condition)) {
      throw TypeErrorException("where expected condition tensor");
    }
    const auto& condition_tensor = tensor_ref(condition);

    auto tensor_or_scalar = [&condition_tensor](PyObject* object) -> TensorPtr {
      if (is_tensor(object)) {
        return tensor_ref(object);
      }
      double scalar = 0.0;
      ScalarType scalar_dtype = ScalarType::Float32;
      if (pyobject_to_scalar(object, scalar, &scalar_dtype)) {
        return mtorch::full({}, scalar, scalar_dtype, condition_tensor->device);
      }
      throw TypeErrorException("where expected tensor or numeric scalar operands");
    };

    return wrap_tensor(mtorch::where(condition_tensor, tensor_or_scalar(left), tensor_or_scalar(right)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_take(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "OO:take", &input, &indices)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "take expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::take(tensor_ref(input), tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_index_select(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "OLO:index_select", &input, &dim, &indices)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "index_select expected Tensor arguments");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::index_select(tensor_ref(input), dim, tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_gather(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "OLO:gather", &input, &dim, &indices)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "gather expected Tensor arguments");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::gather(tensor_ref(input), dim, tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

std::vector<TensorPtr> index_put_indices_from_py(PyObject* object) {
  if (!PyTuple_Check(object) && !PyList_Check(object)) {
    throw TypeErrorException("index_put(): argument 'indices' must be tuple of Tensors");
  }
  const Py_ssize_t length = PySequence_Size(object);
  if (length <= 0) {
    throw std::invalid_argument("index_put expected a non-empty indices tuple");
  }
  std::vector<TensorPtr> indices;
  indices.reserve(static_cast<size_t>(length));
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::runtime_error("could not read index_put index");
    }
    try {
      if (!is_tensor(item)) {
        std::ostringstream message;
        message << "expected Tensor as element " << i << " in argument 1";
        throw TypeErrorException(message.str());
      }
      indices.push_back(tensor_ref(item));
      Py_DECREF(item);
    } catch (...) {
      Py_DECREF(item);
      throw;
    }
  }
  return indices;
}

PyObject* py_index_put(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "indices", "values", "accumulate", nullptr};
  PyObject* input = nullptr;
  PyObject* indices = nullptr;
  PyObject* values = nullptr;
  int accumulate = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOO|p:index_put", const_cast<char**>(keywords), &input, &indices, &values, &accumulate)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "index_put expected Tensor input");
    return nullptr;
  }
  if (!is_tensor(values)) {
    PyErr_SetString(PyExc_TypeError, "index_put(): argument 'values' must be Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::index_put(tensor_ref(input), index_put_indices_from_py(indices), *tensor_ref(values), accumulate != 0));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

TensorPtr scatter_source_from_py(const TensorPtr& input, PyObject* source, const char* name) {
  if (is_tensor(source)) {
    return tensor_ref(source);
  }
  double scalar = 0.0;
  if (pyobject_to_scalar(source, scalar)) {
    return mtorch::make_tensor({scalar}, {}, input->dtype, false, input->device);
  }
  std::ostringstream message;
  message << name << " expected Tensor or scalar source";
  throw TypeErrorException(message.str());
}

PyObject* py_scatter(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "OLOO:scatter", &input, &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter expected Tensor input and index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(tensor_ref(input), source, "scatter");
    return wrap_tensor(mtorch::scatter(tensor_ref(input), dim, tensor_ref(indices), *source_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_scatter_add(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "OLOO:scatter_add", &input, &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter_add expected Tensor input and index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(tensor_ref(input), source, "scatter_add");
    return wrap_tensor(mtorch::scatter_add(tensor_ref(input), dim, tensor_ref(indices), *source_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_masked_select(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* mask = nullptr;
  if (!PyArg_ParseTuple(args, "OO:masked_select", &input, &mask)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(mask)) {
    PyErr_SetString(PyExc_TypeError, "masked_select expected Tensor arguments");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::masked_select(tensor_ref(input), tensor_ref(mask)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_masked_fill(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  PyObject* mask = nullptr;
  PyObject* value = nullptr;
  if (!PyArg_ParseTuple(args, "OOO:masked_fill", &input, &mask, &value)) {
    return nullptr;
  }
  if (!is_tensor(input) || !is_tensor(mask)) {
    PyErr_SetString(PyExc_TypeError, "masked_fill expected Tensor input and mask");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::masked_fill(tensor_ref(input), tensor_ref(mask), scalar_from_py(value)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_nonzero(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "as_tuple", nullptr};
  PyObject* input = nullptr;
  int as_tuple = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:nonzero", const_cast<char**>(keywords), &input, &as_tuple)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "nonzero expected Tensor");
    return nullptr;
  }
  try {
    if (as_tuple != 0) {
      return tuple_from_tensors(mtorch::nonzero_tuple(tensor_ref(input)));
    }
    return wrap_tensor(mtorch::nonzero(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_argwhere(PyObject* self, PyObject* args) {
  (void) self;
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:argwhere", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "argwhere expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::nonzero(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_count_nonzero(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "dim", nullptr};
  PyObject* input = nullptr;
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:count_nonzero", const_cast<char**>(keywords), &input, &dim)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "count_nonzero expected Tensor");
    return nullptr;
  }
  try {
    if (dim != Py_None) {
      return wrap_tensor(mtorch::count_nonzero_dim(tensor_ref(input), PyLong_AsLongLong(dim)));
    }
    return wrap_tensor(mtorch::count_nonzero(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_bincount(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "weights", "minlength", nullptr};
  PyObject* input = nullptr;
  PyObject* weights = Py_None;
  PyObject* minlength = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO:bincount", const_cast<char**>(keywords), &input, &weights, &minlength)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "bincount expected Tensor input");
    return nullptr;
  }
  TensorPtr weight_tensor;
  if (weights != Py_None) {
    if (!is_tensor(weights)) {
      PyErr_SetString(PyExc_TypeError, "bincount weights must be a Tensor or None");
      return nullptr;
    }
    weight_tensor = tensor_ref(weights);
  }
  int64_t minlength_value = 0;
  if (minlength != Py_None) {
    if (PyBool_Check(minlength) || !PyLong_Check(minlength)) {
      PyErr_SetString(PyExc_TypeError, "bincount minlength must be int");
      return nullptr;
    }
    minlength_value = PyLong_AsLongLong(minlength);
    if (PyErr_Occurred()) {
      return nullptr;
    }
  }
  try {
    return wrap_tensor(mtorch::bincount(tensor_ref(input), weight_tensor, minlength_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_one_hot(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  int64_t num_classes = -1;
  if (!PyArg_ParseTuple(args, "O|L:one_hot", &input, &num_classes)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "one_hot expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::one_hot(tensor_ref(input), num_classes));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_relu(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:relu", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "relu expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::relu(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_leaky_relu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "negative_slope", "inplace", nullptr};
  PyObject* input = nullptr;
  double negative_slope = 0.01;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|dp:leaky_relu", const_cast<char**>(keywords), &input, &negative_slope, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "leaky_relu expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "leaky_relu inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::leaky_relu(tensor_ref(input), negative_slope));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_silu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:silu", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "silu expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "silu inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::silu(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_elu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "alpha", "inplace", nullptr};
  PyObject* input = nullptr;
  double alpha = 1.0;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|dp:elu", const_cast<char**>(keywords), &input, &alpha, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "elu expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "elu inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::elu(tensor_ref(input), alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_selu(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:selu", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "selu expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "selu inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::selu(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_softplus(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "beta", "threshold", nullptr};
  PyObject* input = nullptr;
  double beta = 1.0;
  double threshold = 20.0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|dd:softplus", const_cast<char**>(keywords), &input, &beta, &threshold)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "softplus expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::softplus(tensor_ref(input), beta, threshold));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_hardtanh(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "min_val", "max_val", "inplace", nullptr};
  PyObject* input = nullptr;
  double min_value = -1.0;
  double max_value = 1.0;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|ddp:hardtanh", const_cast<char**>(keywords), &input, &min_value, &max_value, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "hardtanh expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "hardtanh inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::hardtanh(tensor_ref(input), min_value, max_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_relu6(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:relu6", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "relu6 expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "relu6 inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::relu6(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_hardsigmoid(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|p:hardsigmoid", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "hardsigmoid expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "hardsigmoid inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::hardsigmoid(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_hardswish(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:hardswish", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "hardswish expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "hardswish inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::hardswish(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_softsign(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:softsign", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "softsign expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::softsign(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_mish(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"input", "inplace", nullptr};
  PyObject* input = nullptr;
  int inplace = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:mish", const_cast<char**>(keywords), &input, &inplace)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "mish expected Tensor");
    return nullptr;
  }
  if (inplace) {
    PyErr_SetString(PyExc_NotImplementedError, "mish inplace is not implemented yet");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::mish(tensor_ref(input)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_clone(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:clone", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "clone expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(tensor_ref(input)->clone());
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_numel(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:numel", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "numel expected Tensor");
    return nullptr;
  }
  return PyLong_FromLongLong(tensor_ref(input)->numel());
}

PyObject* py_is_tensor(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_tensor", &input)) {
    return nullptr;
  }
  return PyBool_FromLong(is_tensor(input) ? 1 : 0);
}

PyObject* py_mark_parameter(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:_mark_parameter", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "_mark_parameter expected Tensor");
    return nullptr;
  }
  reinterpret_cast<PyTensor*>(input)->is_parameter = true;
  Py_RETURN_NONE;
}

PyObject* py_is_parameter(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:_is_parameter", &input)) {
    return nullptr;
  }
  return PyBool_FromLong(is_tensor(input) && reinterpret_cast<PyTensor*>(input)->is_parameter ? 1 : 0);
}

PyObject* py_is_floating_point(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_floating_point", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "is_floating_point expected Tensor");
    return nullptr;
  }
  const auto dtype = tensor_ref(input)->dtype;
  return PyBool_FromLong(
      dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64 ? 1 : 0);
}

bool dtype_is_signed(ScalarType dtype) {
  return dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64 ||
      dtype == ScalarType::Int32 || dtype == ScalarType::Int64;
}

PyObject* py_is_complex(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_complex", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "is_complex expected Tensor");
    return nullptr;
  }
  Py_RETURN_FALSE;
}

PyObject* py_is_conj(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_conj", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "is_conj expected Tensor");
    return nullptr;
  }
  Py_RETURN_FALSE;
}

PyObject* py_is_signed(PyObject*, PyObject* args) {
  PyObject* input = nullptr;
  if (!PyArg_ParseTuple(args, "O:is_signed", &input)) {
    return nullptr;
  }
  if (!is_tensor(input)) {
    PyErr_SetString(PyExc_TypeError, "is_signed expected Tensor");
    return nullptr;
  }
  return PyBool_FromLong(dtype_is_signed(tensor_ref(input)->dtype) ? 1 : 0);
}

PyObject* py_is_grad_enabled(PyObject*, PyObject*) {
  return PyBool_FromLong(mtorch::is_grad_enabled() ? 1 : 0);
}

PyObject* py_set_grad_enabled(PyObject*, PyObject* args) {
  PyObject* enabled = nullptr;
  if (!PyArg_ParseTuple(args, "O:_set_grad_enabled", &enabled)) {
    return nullptr;
  }
  const int truth = PyObject_IsTrue(enabled);
  if (truth < 0) {
    return nullptr;
  }
  const bool previous = mtorch::set_grad_enabled(truth != 0);
  return PyBool_FromLong(previous ? 1 : 0);
}

void collect_autograd_nodes(const TensorPtr& tensor, std::vector<TensorPtr>& nodes) {
  for (const auto& node : nodes) {
    if (node.get() == tensor.get()) {
      return;
    }
  }
  nodes.push_back(tensor);
  for (const auto& parent : tensor->parents) {
    if (parent) {
      collect_autograd_nodes(parent, nodes);
    }
  }
}

struct GradSnapshot {
  std::vector<TensorPtr> nodes;
  std::vector<TensorPtr> saved_grads;

  explicit GradSnapshot(std::vector<TensorPtr> graph_nodes) : nodes(std::move(graph_nodes)) {
    saved_grads.reserve(nodes.size());
    for (const auto& node : nodes) {
      saved_grads.push_back(node->grad);
      node->grad = nullptr;
    }
  }

  ~GradSnapshot() {
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i]->grad = saved_grads[i];
    }
  }
};

std::vector<TensorPtr> grad_outputs_from_py(
    PyObject* object,
    const std::vector<TensorPtr>& outputs) {
  std::vector<TensorPtr> grad_outputs;
  grad_outputs.reserve(outputs.size());
  if (object == nullptr || object == Py_None) {
    grad_outputs.resize(outputs.size());
    return grad_outputs;
  }
  if (outputs.size() == 1 && (is_tensor(object) || object == Py_None)) {
    grad_outputs.push_back(object == Py_None ? nullptr : tensor_ref(object));
    return grad_outputs;
  }
  if (!object_is_sequence(object)) {
    throw TypeErrorException("grad_outputs must be a Tensor, None, or a sequence");
  }
  const Py_ssize_t length = PySequence_Length(object);
  if (length < 0) {
    throw std::invalid_argument("could not read grad_outputs");
  }
  if (static_cast<size_t>(length) != outputs.size()) {
    throw std::invalid_argument("grad_outputs must match outputs length");
  }
  for (Py_ssize_t i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(object, i);
    if (item == nullptr) {
      throw std::invalid_argument("could not read grad_outputs");
    }
    if (item == Py_None) {
      grad_outputs.push_back(nullptr);
      Py_DECREF(item);
      continue;
    }
    if (!is_tensor(item)) {
      Py_DECREF(item);
      throw TypeErrorException("grad_outputs entries must be tensors or None");
    }
    grad_outputs.push_back(tensor_ref(item));
    Py_DECREF(item);
  }
  return grad_outputs;
}

PyObject* py_autograd_grad(PyObject*, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"outputs", "inputs", "grad_outputs", "allow_unused", "materialize_grads", nullptr};
  PyObject* outputs_object = nullptr;
  PyObject* inputs_object = nullptr;
  PyObject* grad_outputs_object = Py_None;
  int allow_unused = 0;
  int materialize_grads = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "OO|Opp:_autograd_grad",
          const_cast<char**>(keywords),
          &outputs_object,
          &inputs_object,
          &grad_outputs_object,
          &allow_unused,
          &materialize_grads)) {
    return nullptr;
  }

  try {
    auto outputs = tensor_sequence_from_py(outputs_object, "outputs");
    auto inputs = tensor_sequence_from_py(inputs_object, "inputs");
    auto grad_outputs = grad_outputs_from_py(grad_outputs_object, outputs);
    std::vector<TensorPtr> graph_nodes;
    for (const auto& output : outputs) {
      collect_autograd_nodes(output, graph_nodes);
    }
    GradSnapshot snapshot(std::move(graph_nodes));

    for (const auto& input : inputs) {
      if (!input->requires_grad) {
        throw std::runtime_error("one of the differentiated Tensors does not require grad");
      }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      TensorPtr upstream = grad_outputs[i];
      if (!upstream) {
        if (outputs[i]->numel() != 1) {
          throw std::invalid_argument("grad can be implicitly created only for scalar outputs");
        }
        upstream = mtorch::ones({}, ScalarType::Float32, outputs[i]->device);
      } else if (upstream->sizes != outputs[i]->sizes) {
        throw std::invalid_argument("grad_output has an incompatible shape");
      }
      outputs[i]->backward_with(*upstream);
    }

    for (const auto& input : inputs) {
      if (!input->grad && !allow_unused && !materialize_grads) {
        throw std::runtime_error("one of the differentiated Tensors appears to not have been used in the graph");
      }
    }

    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(inputs.size()));
    if (tuple == nullptr) {
      return nullptr;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
      PyObject* item = nullptr;
      if (inputs[i]->grad) {
        item = wrap_tensor(inputs[i]->grad);
      } else if (materialize_grads) {
        item = wrap_tensor(mtorch::zeros_like(inputs[i], inputs[i]->dtype, inputs[i]->device, false));
      } else {
        Py_INCREF(Py_None);
        item = Py_None;
      }
      if (item == nullptr) {
        Py_DECREF(tuple);
        return nullptr;
      }
      PyTuple_SET_ITEM(tuple, static_cast<Py_ssize_t>(i), item);
    }
    return tuple;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* py_not_implemented(PyObject*, PyObject*) {
  PyErr_SetString(PyExc_NotImplementedError, "not implemented yet");
  return nullptr;
}

void Tensor_dealloc(PyTensor* self) {
  delete self->value;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Tensor_new(PyTypeObject*, PyObject* args, PyObject* kwargs) {
  return parse_tensor_call(args, kwargs);
}

PyObject* Tensor_repr(PyTensor* self) {
  std::ostringstream stream;
  stream << "mtorch.Tensor(shape=(";
  for (size_t i = 0; i < self->value->get()->sizes.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << self->value->get()->sizes[i];
  }
  stream << "), dtype=" << mtorch::dtype_name(self->value->get()->dtype) << ")";
  return PyUnicode_FromString(stream.str().c_str());
}

PyObject* Tensor_get_shape(PyTensor* self, void*) {
  return make_size_tuple(self->value->get()->sizes);
}

PyObject* Tensor_get_ndim(PyTensor* self, void*) {
  return PyLong_FromLongLong(self->value->get()->dim());
}

PyObject* Tensor_get_dtype(PyTensor* self, void*) {
  return PyUnicode_FromString(mtorch::dtype_name(self->value->get()->dtype).c_str());
}

PyObject* Tensor_get_device(PyTensor* self, void*) {
  return PyUnicode_FromString(mtorch::device_name(self->value->get()->device).c_str());
}

PyObject* Tensor_get_requires_grad(PyTensor* self, void*) {
  return PyBool_FromLong(self->value->get()->requires_grad ? 1 : 0);
}

PyObject* Tensor_get_grad(PyTensor* self, void*) {
  if (!self->value->get()->grad) {
    Py_RETURN_NONE;
  }
  return wrap_tensor(self->value->get()->grad);
}

PyObject* Tensor_get_data(PyTensor* self, void*) {
  auto source = self->value->get();
  auto result = std::make_shared<Tensor>(
      source->storage, source->sizes, source->strides, source->offset, source->dtype, false);
  return wrap_tensor(result);
}

int Tensor_set_data(PyTensor* self, PyObject* value, void*) {
  if (!is_tensor(value)) {
    PyErr_SetString(PyExc_TypeError, "data must be a Tensor");
    return -1;
  }
  try {
    auto target = self->value->get();
    auto source = tensor_ref(value);
    if (!mtorch::devices_equal(target->device, source->device)) {
      PyErr_SetString(PyExc_RuntimeError, "data assignment requires tensors to be on the same device");
      return -1;
    }
    if (target->requires_grad && !dtype_allows_requires_grad(source->dtype)) {
      PyErr_SetString(
          PyExc_RuntimeError,
          "data set to a tensor that requires gradients must be floating point or complex dtype");
      return -1;
    }
    target->storage = source->storage;
    target->sizes = source->sizes;
    target->strides = source->strides;
    target->offset = source->offset;
    target->dtype = source->dtype;
    target->device = source->device;
    return 0;
  } catch (...) {
    translate_exception();
    return -1;
  }
}

int Tensor_set_grad(PyTensor* self, PyObject* value, void*) {
  if (value == nullptr || value == Py_None) {
    self->value->get()->grad = nullptr;
    return 0;
  }
  if (!is_tensor(value)) {
    PyErr_SetString(PyExc_TypeError, "grad must be a Tensor or None");
    return -1;
  }
  auto target = self->value->get();
  auto gradient = tensor_ref(value);
  if (gradient->sizes != target->sizes) {
    PyErr_SetString(PyExc_ValueError, "assigned grad has an incompatible shape");
    return -1;
  }
  target->grad = gradient;
  return 0;
}

PyObject* Tensor_tolist(PyTensor* self, PyObject*) {
  std::vector<int64_t> index;
  return tensor_to_nested_list(*self->value->get(), 0, index);
}

PyObject* Tensor_item(PyTensor* self, PyObject*) {
  if (self->value->get()->numel() != 1) {
    PyErr_SetString(PyExc_ValueError, "only one-element tensors can be converted to Python scalars");
    return nullptr;
  }
  return scalar_to_py(self->value->get()->value_at_linear(0), self->value->get()->dtype);
}

PyObject* Tensor_stride(PyTensor* self, PyObject*) {
  return make_size_tuple(self->value->get()->strides);
}

PyObject* Tensor_element_size(PyTensor* self, PyObject*) {
  return PyLong_FromLongLong(self->value->get()->element_size());
}

PyObject* Tensor_size(PyTensor* self, PyObject* args) {
  if (PyTuple_GET_SIZE(args) == 0) {
    return make_size_tuple(self->value->get()->sizes);
  }
  long long dim = 0;
  if (!PyArg_ParseTuple(args, "L:size", &dim)) {
    return nullptr;
  }
  const int64_t rank = self->value->get()->dim();
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    PyErr_SetString(PyExc_IndexError, "dimension out of range");
    return nullptr;
  }
  return PyLong_FromLongLong(self->value->get()->sizes[static_cast<size_t>(dim)]);
}

PyObject* Tensor_dim_method(PyTensor* self, PyObject*) {
  return PyLong_FromLongLong(self->value->get()->dim());
}

PyObject* Tensor_numel(PyTensor* self, PyObject*) {
  return PyLong_FromLongLong(self->value->get()->numel());
}

PyObject* Tensor_is_contiguous(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"memory_format", nullptr};
  PyObject* memory_format = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|O:is_contiguous", const_cast<char**>(keywords), &memory_format)) {
    return nullptr;
  }
  try {
    if (memory_format == nullptr) {
      return PyBool_FromLong(self->value->get()->is_contiguous() ? 1 : 0);
    }
    const MemoryFormat format_value = memory_format_from_py(memory_format);
    return PyBool_FromLong(tensor_is_contiguous_memory_format(*self->value->get(), format_value) ? 1 : 0);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_is_floating_point_method(PyTensor* self, PyObject*) {
  const auto dtype = self->value->get()->dtype;
  return PyBool_FromLong(
      dtype == ScalarType::Float16 || dtype == ScalarType::Float32 || dtype == ScalarType::Float64 ? 1 : 0);
}

PyObject* Tensor_is_complex_method(PyTensor*, PyObject*) {
  Py_RETURN_FALSE;
}

PyObject* Tensor_is_conj_method(PyTensor*, PyObject*) {
  Py_RETURN_FALSE;
}

PyObject* Tensor_is_signed_method(PyTensor* self, PyObject*) {
  return PyBool_FromLong(dtype_is_signed(self->value->get()->dtype) ? 1 : 0);
}

PyObject* Tensor_clone(PyTensor* self, PyObject*) {
  return wrap_tensor(self->value->get()->clone());
}

PyObject* Tensor_contiguous(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"memory_format", nullptr};
  PyObject* memory_format = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|O:contiguous", const_cast<char**>(keywords), &memory_format)) {
    return nullptr;
  }
  try {
    if (memory_format == nullptr) {
      return wrap_tensor(self->value->get()->contiguous());
    }
    const MemoryFormat format_value = memory_format_from_py(memory_format);
    if (format_value == MemoryFormat::Preserve && !self->value->get()->is_contiguous()) {
      throw std::runtime_error("preserve memory format is unsupported by the contiguous operator");
    }
    return wrap_tensor(tensor_to_memory_format(*self->value, format_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_detach(PyTensor* self, PyObject*) {
  auto source = self->value->get();
  auto result = std::make_shared<Tensor>(
      source->storage, source->sizes, source->strides, source->offset, source->dtype, false);
  return wrap_tensor(result);
}

PyObject* Tensor_detach_inplace(PyTensor* self, PyObject*) {
  auto tensor = self->value->get();
  tensor->requires_grad = false;
  tensor->parents.clear();
  tensor->backward_fn = nullptr;
  tensor->grad.reset();
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* Tensor_requires_grad_inplace(PyTensor* self, PyObject* args) {
  PyObject* requires_grad = Py_True;
  if (!PyArg_ParseTuple(args, "|O:requires_grad_", &requires_grad)) {
    return nullptr;
  }
  const int truth = PyObject_IsTrue(requires_grad);
  if (truth < 0) {
    return nullptr;
  }
  self->value->get()->requires_grad = truth != 0;
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* Tensor_cpu(PyTensor* self, PyObject*) {
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* Tensor_to(PyTensor* self, PyObject* args, PyObject* kwargs) {
  try {
    const Tensor& source = *self->value->get();
    const auto request = parse_to_request(source, args, kwargs);
    auto result = mtorch::to(*self->value, request.dtype, request.device, request.copy);
    if (request.memory_format.has_value()) {
      result = tensor_to_memory_format(result, *request.memory_format, request.copy);
    }
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_float_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Float32, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_double_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Float64, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_half_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Float16, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_long_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Int64, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_int_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Int32, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_bool_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::to(*self->value, ScalarType::Bool, self->value->get()->device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_type_as(PyTensor* self, PyObject* other) {
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "type_as expected Tensor");
    return nullptr;
  }
  try {
    const Tensor& target = *tensor_ref(other);
    return wrap_tensor(mtorch::to(*self->value, target.dtype, target.device));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_new_tensor(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"data", "dtype", "device", "requires_grad", nullptr};
  PyObject* data = nullptr;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  int requires_grad = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|OOp:new_tensor", const_cast<char**>(keywords), &data, &dtype, &device, &requires_grad)) {
    return nullptr;
  }

  try {
    const Tensor& source = *self->value->get();
    const ScalarType target_dtype = dtype_from_py(dtype, source.dtype);
    const Device target_device = device_from_py(device, source.device);
    if (is_tensor(data)) {
      auto input = tensor_ref(data);
      auto copy = mtorch::make_tensor(input->contiguous_values(), input->sizes, target_dtype, false, target_device);
      copy->requires_grad = requires_grad != 0;
      return wrap_tensor(copy);
    }

    std::vector<double> values;
    std::vector<int64_t> shape;
    ScalarType inferred_dtype = source.dtype;
    parse_nested_data(data, values, shape, inferred_dtype, 0);
    auto result = mtorch::make_tensor(values, shape, target_dtype, requires_grad != 0, target_device);
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

struct NewFactoryOptions {
  std::vector<int64_t> shape;
  PyObject* dtype = Py_None;
  PyObject* device = Py_None;
  bool requires_grad = false;
};

NewFactoryOptions tensor_new_factory_options_from_args(
    const Tensor& source,
    PyObject* args,
    PyObject* kwargs,
    const char* name,
    bool has_fill_value,
    PyObject** fill_value = nullptr) {
  NewFactoryOptions options;
  PyObject* size_kw = nullptr;
  PyObject* fill_value_kw = nullptr;
  if (kwargs != nullptr) {
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    Py_ssize_t position = 0;
    while (PyDict_Next(kwargs, &position, &key, &value)) {
      const char* key_text = PyUnicode_AsUTF8(key);
      if (key_text == nullptr) {
        throw std::invalid_argument("factory keyword names must be strings");
      }
      if (std::strcmp(key_text, "size") == 0) {
        size_kw = value;
      } else if (has_fill_value && std::strcmp(key_text, "fill_value") == 0) {
        fill_value_kw = value;
      } else if (std::strcmp(key_text, "dtype") == 0) {
        options.dtype = value;
      } else if (std::strcmp(key_text, "device") == 0) {
        options.device = value;
      } else if (std::strcmp(key_text, "requires_grad") == 0) {
        const int truth = PyObject_IsTrue(value);
        if (truth < 0) {
          throw std::invalid_argument("requires_grad must be truthy or falsy");
        }
        options.requires_grad = truth != 0;
      } else if (std::strcmp(key_text, "layout") == 0) {
        if (value != Py_None) {
          throw NotImplementedException(std::string(name) + " layout is not implemented");
        }
      } else if (std::strcmp(key_text, "pin_memory") == 0) {
        const int truth = PyObject_IsTrue(value);
        if (truth < 0) {
          throw std::invalid_argument("pin_memory must be truthy or falsy");
        }
        if (truth != 0) {
          throw NotImplementedException(std::string(name) + " pin_memory=True is not implemented");
        }
      } else {
        throw std::invalid_argument(std::string(name) + " got an unexpected keyword argument '" + key_text + "'");
      }
    }
  }

  const Py_ssize_t positional_count = PyTuple_GET_SIZE(args);
  if (size_kw != nullptr) {
    const Py_ssize_t expected_positionals = has_fill_value && fill_value_kw == nullptr ? 1 : 0;
    if (positional_count != expected_positionals) {
      throw std::invalid_argument(std::string(name) + " got both positional size and size keyword");
    }
    options.shape = shape_from_object(size_kw);
    if (has_fill_value && fill_value != nullptr) {
      *fill_value = fill_value_kw == nullptr ? PyTuple_GET_ITEM(args, 0) : fill_value_kw;
    }
    return options;
  }
  if (has_fill_value) {
    const Py_ssize_t expected_positionals = fill_value_kw == nullptr ? 2 : 1;
    if (positional_count != expected_positionals) {
      throw std::invalid_argument(std::string(name) + " expected size and fill_value");
    }
    if (fill_value != nullptr) {
      *fill_value = fill_value_kw == nullptr ? PyTuple_GET_ITEM(args, 1) : fill_value_kw;
    }
    options.shape = shape_from_object(PyTuple_GET_ITEM(args, 0));
  } else {
    if (positional_count == 0) {
      throw std::invalid_argument(std::string(name) + " missing required size");
    }
    options.shape = shape_from_args(args);
  }
  (void)source;
  return options;
}

PyObject* Tensor_new_empty(PyTensor* self, PyObject* args, PyObject* kwargs) {
  try {
    const Tensor& source = *self->value->get();
    const auto options = tensor_new_factory_options_from_args(source, args, kwargs, "new_empty", false);
    auto result = mtorch::zeros(
        options.shape,
        dtype_from_py(options.dtype, source.dtype),
        device_from_py(options.device, source.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_new_zeros(PyTensor* self, PyObject* args, PyObject* kwargs) {
  try {
    const Tensor& source = *self->value->get();
    const auto options = tensor_new_factory_options_from_args(source, args, kwargs, "new_zeros", false);
    auto result = mtorch::zeros(
        options.shape,
        dtype_from_py(options.dtype, source.dtype),
        device_from_py(options.device, source.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_new_ones(PyTensor* self, PyObject* args, PyObject* kwargs) {
  try {
    const Tensor& source = *self->value->get();
    const auto options = tensor_new_factory_options_from_args(source, args, kwargs, "new_ones", false);
    auto result = mtorch::ones(
        options.shape,
        dtype_from_py(options.dtype, source.dtype),
        device_from_py(options.device, source.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_new_full(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* fill_value = nullptr;
  try {
    const Tensor& source = *self->value->get();
    if (kwargs == nullptr && PyTuple_GET_SIZE(args) == 2) {
      return wrap_tensor(mtorch::full(
          shape_from_object(PyTuple_GET_ITEM(args, 0)),
          scalar_from_py(PyTuple_GET_ITEM(args, 1)),
          source.dtype,
          source.device));
    }
    const auto options = tensor_new_factory_options_from_args(source, args, kwargs, "new_full", true, &fill_value);
    auto result = mtorch::full(
        options.shape,
        scalar_from_py(fill_value),
        dtype_from_py(options.dtype, source.dtype),
        device_from_py(options.device, source.device));
    result->requires_grad = options.requires_grad;
    return wrap_tensor(result);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_reshape(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::reshape(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_view(PyTensor* self, PyObject* args) {
  try {
    if (!self->value->get()->is_contiguous()) {
      throw std::runtime_error("view size is not compatible with input tensor's size and stride");
    }
    return wrap_tensor(mtorch::reshape(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_unflatten(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* sizes = nullptr;
  if (!PyArg_ParseTuple(args, "LO:unflatten", &dim, &sizes)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unflatten(*self->value, dim, shape_from_object(sizes)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_transpose(PyTensor* self, PyObject* args) {
  long long dim0 = 0;
  long long dim1 = 0;
  if (!PyArg_ParseTuple(args, "LL:transpose", &dim0, &dim1)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(*self->value, dim0, dim1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_permute(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::permute(*self->value, dims_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_movedim(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"source", "destination", nullptr};
  PyObject* source = nullptr;
  PyObject* destination = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO:movedim", const_cast<char**>(keywords), &source, &destination)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::movedim(*self->value, shape_from_object(source), shape_from_object(destination)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_swapaxes(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"axis0", "axis1", nullptr};
  long long axis0 = 0;
  long long axis1 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "LL:swapaxes", const_cast<char**>(keywords), &axis0, &axis1)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(*self->value, axis0, axis1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_swapdims(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"dim0", "dim1", nullptr};
  long long dim0 = 0;
  long long dim1 = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "LL:swapdims", const_cast<char**>(keywords), &dim0, &dim1)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::transpose(*self->value, dim0, dim1));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_flatten(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"start_dim", "end_dim", nullptr};
  long long start_dim = 0;
  long long end_dim = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|LL:flatten", const_cast<char**>(keywords), &start_dim, &end_dim)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::flatten(*self->value, start_dim, end_dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_ravel(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::ravel(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_t(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::t(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_expand(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::expand(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_expand_as(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:expand_as", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "expand_as expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::expand(*self->value, tensor_ref(other)->sizes));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_repeat(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::repeat(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_repeat_interleave(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"repeats", "dim", "output_size", nullptr};
  PyObject* repeats = nullptr;
  PyObject* dim = Py_None;
  PyObject* output_size = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "O|OO:repeat_interleave", const_cast<char**>(keywords), &repeats, &dim, &output_size)) {
    return nullptr;
  }
  try {
    const auto dim_value = optional_int64_from_py(dim, "dim");
    const auto output_size_value = optional_int64_from_py(output_size, "output_size");
    if (is_tensor(repeats)) {
      return wrap_tensor(mtorch::repeat_interleave(*self->value, tensor_ref(repeats), dim_value, output_size_value));
    }
    const int64_t repeat_count = PyLong_AsLongLong(repeats);
    if (PyErr_Occurred()) {
      throw std::invalid_argument("repeat_interleave repeats must be an integer or Tensor");
    }
    return wrap_tensor(mtorch::repeat_interleave(*self->value, repeat_count, dim_value, output_size_value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_tile(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::tile(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_flip(PyTensor* self, PyObject* args) {
  try {
    return wrap_tensor(mtorch::flip(*self->value, shape_from_args(args)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_fliplr(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::fliplr(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_flipud(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::flipud(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_rot90(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"k", "dims", nullptr};
  long long k = 1;
  PyObject* dims = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|LO:rot90", const_cast<char**>(keywords), &k, &dims)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::rot90(*self->value, k, dims == Py_None ? std::vector<int64_t>{0, 1} : shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_roll(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"shifts", "dims", nullptr};
  PyObject* shifts = nullptr;
  PyObject* dims = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O:roll", const_cast<char**>(keywords), &shifts, &dims)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::roll(*self->value, shape_from_object(shifts), dims == Py_None ? std::vector<int64_t>{} : shape_from_object(dims)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_squeeze(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"dim", nullptr};
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:squeeze", const_cast<char**>(keywords), &dim)) {
    return nullptr;
  }
  try {
    if (dim != Py_None) {
      return wrap_tensor(mtorch::squeeze(*self->value, PyLong_AsLongLong(dim)));
    }
    return wrap_tensor(mtorch::squeeze(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_unsqueeze(PyTensor* self, PyObject* args) {
  long long dim = 0;
  if (!PyArg_ParseTuple(args, "L:unsqueeze", &dim)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::unsqueeze(*self->value, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_narrow(PyTensor* self, PyObject* args) {
  long long dim = 0;
  long long start = 0;
  long long length = 0;
  if (!PyArg_ParseTuple(args, "LLL:narrow", &dim, &start, &length)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::narrow(*self->value, dim, start, length));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_select(PyTensor* self, PyObject* args) {
  long long dim = 0;
  long long index = 0;
  if (!PyArg_ParseTuple(args, "LL:select", &dim, &index)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::select(*self->value, dim, index));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_as_strided(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"size", "stride", "storage_offset", nullptr};
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  PyObject* storage_offset = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OO|O:as_strided", const_cast<char**>(keywords), &size, &stride, &storage_offset)) {
    return nullptr;
  }
  try {
    std::optional<int64_t> offset;
    if (storage_offset != Py_None) {
      offset = PyLong_AsLongLong(storage_offset);
      if (PyErr_Occurred()) {
        throw std::invalid_argument("storage_offset must be an integer or None");
      }
    }
    return wrap_tensor(mtorch::as_strided(*self->value, shape_from_object(size), shape_from_object(stride), offset));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_diagonal(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"offset", "dim1", "dim2", nullptr};
  long long offset = 0;
  long long dim1 = 0;
  long long dim2 = 1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|LLL:diagonal", const_cast<char**>(keywords), &offset, &dim1, &dim2)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diagonal(*self->value, offset, dim1, dim2));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_diag(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"diagonal", nullptr};
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L:diag", const_cast<char**>(keywords), &diagonal)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diag(*self->value, diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_diag_embed(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"offset", "dim1", "dim2", nullptr};
  long long offset = 0;
  long long dim1 = -2;
  long long dim2 = -1;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|LLL:diag_embed", const_cast<char**>(keywords), &offset, &dim1, &dim2)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::diag_embed(*self->value, offset, dim1, dim2));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_tril(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"diagonal", nullptr};
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L:tril", const_cast<char**>(keywords), &diagonal)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::tril(*self->value, diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_triu(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"diagonal", nullptr};
  long long diagonal = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L:triu", const_cast<char**>(keywords), &diagonal)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::triu(*self->value, diagonal));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_trace(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::trace(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_index_select(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "LO:index_select", &dim, &indices)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "index_select expected Tensor index");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::index_select(*self->value, dim, tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_gather(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "LO:gather", &dim, &indices)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "gather expected Tensor index");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::gather(*self->value, dim, tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_scatter(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "LOO:scatter", &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter expected Tensor index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(*self->value, source, "scatter");
    return wrap_tensor(mtorch::scatter(*self->value, dim, tensor_ref(indices), *source_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_scatter_inplace(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "LOO:scatter_", &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter_ expected Tensor index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(*self->value, source, "scatter_");
    mtorch::scatter_inplace(*self->value, dim, tensor_ref(indices), *source_tensor);
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_scatter_add(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "LOO:scatter_add", &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter_add expected Tensor index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(*self->value, source, "scatter_add");
    return wrap_tensor(mtorch::scatter_add(*self->value, dim, tensor_ref(indices), *source_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_scatter_add_inplace(PyTensor* self, PyObject* args) {
  long long dim = 0;
  PyObject* indices = nullptr;
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "LOO:scatter_add_", &dim, &indices, &source)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "scatter_add_ expected Tensor index");
    return nullptr;
  }
  try {
    auto source_tensor = scatter_source_from_py(*self->value, source, "scatter_add_");
    mtorch::scatter_add_inplace(*self->value, dim, tensor_ref(indices), *source_tensor);
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_take(PyTensor* self, PyObject* args) {
  PyObject* indices = nullptr;
  if (!PyArg_ParseTuple(args, "O:take", &indices)) {
    return nullptr;
  }
  if (!is_tensor(indices)) {
    PyErr_SetString(PyExc_TypeError, "take expected Tensor index");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::take(*self->value, tensor_ref(indices)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_masked_select(PyTensor* self, PyObject* args) {
  PyObject* mask = nullptr;
  if (!PyArg_ParseTuple(args, "O:masked_select", &mask)) {
    return nullptr;
  }
  if (!is_tensor(mask)) {
    PyErr_SetString(PyExc_TypeError, "masked_select expected Tensor mask");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::masked_select(*self->value, tensor_ref(mask)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_masked_fill(PyTensor* self, PyObject* args) {
  PyObject* mask = nullptr;
  PyObject* value = nullptr;
  if (!PyArg_ParseTuple(args, "OO:masked_fill", &mask, &value)) {
    return nullptr;
  }
  if (!is_tensor(mask)) {
    PyErr_SetString(PyExc_TypeError, "masked_fill expected Tensor mask");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::masked_fill(*self->value, tensor_ref(mask), scalar_from_py(value)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_masked_fill_inplace(PyTensor* self, PyObject* args) {
  PyObject* mask = nullptr;
  PyObject* value = nullptr;
  if (!PyArg_ParseTuple(args, "OO:masked_fill_", &mask, &value)) {
    return nullptr;
  }
  if (!is_tensor(mask)) {
    PyErr_SetString(PyExc_TypeError, "masked_fill_ expected Tensor mask");
    return nullptr;
  }
  try {
    auto filled = mtorch::masked_fill(*self->value, tensor_ref(mask), scalar_from_py(value));
    self->value->get()->copy_from(*filled);
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_nonzero(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"as_tuple", nullptr};
  int as_tuple = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p:nonzero", const_cast<char**>(keywords), &as_tuple)) {
    return nullptr;
  }
  try {
    if (as_tuple != 0) {
      return tuple_from_tensors(mtorch::nonzero_tuple(*self->value));
    }
    return wrap_tensor(mtorch::nonzero(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_argwhere(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::nonzero(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_count_nonzero(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"dim", nullptr};
  PyObject* dim = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O:count_nonzero", const_cast<char**>(keywords), &dim)) {
    return nullptr;
  }
  try {
    if (dim != Py_None) {
      return wrap_tensor(mtorch::count_nonzero_dim(*self->value, PyLong_AsLongLong(dim)));
    }
    return wrap_tensor(mtorch::count_nonzero(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_split(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"split_size", "dim", nullptr};
  PyObject* split_size = nullptr;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|L:split", const_cast<char**>(keywords), &split_size, &dim)) {
    return nullptr;
  }
  try {
    if (PyLong_Check(split_size)) {
      return tuple_from_tensors(mtorch::split(*self->value, PyLong_AsLongLong(split_size), dim));
    }
    return tuple_from_tensors(mtorch::split(*self->value, shape_from_object(split_size), dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_chunk(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"chunks", "dim", nullptr};
  long long chunks = 0;
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "L|L:chunk", const_cast<char**>(keywords), &chunks, &dim)) {
    return nullptr;
  }
  try {
    return tuple_from_tensors(mtorch::chunk(*self->value, chunks, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_unbind(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"dim", nullptr};
  long long dim = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|L:unbind", const_cast<char**>(keywords), &dim)) {
    return nullptr;
  }
  try {
    return tuple_from_tensors(mtorch::unbind(*self->value, dim));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_reshape_as(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:reshape_as", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "reshape_as expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::reshape(*self->value, tensor_ref(other)->sizes));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_view_as(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:view_as", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "view_as expected Tensor");
    return nullptr;
  }
  try {
    if (!self->value->get()->is_contiguous()) {
      throw std::runtime_error("view size is not compatible with input tensor's size and stride");
    }
    return wrap_tensor(mtorch::reshape(*self->value, tensor_ref(other)->sizes));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_subscript(PyObject* self, PyObject* key) {
  if (!is_tensor(self)) {
    PyErr_SetString(PyExc_TypeError, "__getitem__ expected Tensor");
    return nullptr;
  }
  try {
    auto& tensor = tensor_ref(self);
    auto integer_tuple_indices = int_tensor_tuple_from_key(key, tensor->device);
    if (!integer_tuple_indices.empty()) {
      return wrap_tensor(mtorch::index_integer_tuple(tensor, integer_tuple_indices));
    }
    MixedAdvancedKey mixed;
    if (parse_mixed_advanced_key(key, *tensor, mixed)) {
      if (mixed.mask) {
        return wrap_tensor(mtorch::index_bool_mask(tensor, mixed.mask, mixed.tail_indices));
      }
      return wrap_tensor(mtorch::index_int_tensor(tensor, mixed.int_indices, mixed.tail_indices));
    }
    if (auto mask = bool_mask_from_key(key, tensor->device)) {
      return wrap_tensor(mtorch::index_bool_mask(tensor, mask));
    }
    if (auto indices = int_tensor_from_key(key, tensor->device)) {
      return wrap_tensor(mtorch::index_int_tensor(tensor, indices));
    }
    DimIntAdvancedKey dim_int;
    if (parse_dim_int_advanced_key(key, tensor, dim_int)) {
      return wrap_tensor(mtorch::index_int_tensor_dim(dim_int.base, dim_int.int_indices, dim_int.dim));
    }
    return wrap_tensor(mtorch::index(tensor, parse_tensor_indices(key, *tensor)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

void assign_tensor_subscript(PyObject* self, PyObject* key, PyObject* value) {
  if (!is_tensor(self)) {
    throw std::invalid_argument("__setitem__ expected Tensor");
  }
  if (value == nullptr) {
    throw std::invalid_argument("cannot delete tensor elements");
  }

  auto& tensor = tensor_ref(self);
  auto integer_tuple_indices = int_tensor_tuple_from_key(key, tensor->device);
  if (!integer_tuple_indices.empty()) {
    double scalar = 0.0;
    if (pyobject_to_scalar(value, scalar)) {
      auto source = mtorch::make_tensor({scalar}, {}, tensor->dtype, false, tensor->device);
      mtorch::index_put_integer_tuple(tensor, integer_tuple_indices, *source, false);
      return;
    }
    if (is_tensor(value)) {
      mtorch::index_put_integer_tuple(tensor, integer_tuple_indices, *tensor_ref(value), false);
      return;
    }
    if (object_is_sequence(value)) {
      throw TypeErrorException("can't assign a sequence to an mtorch.Tensor");
    }
    throw TypeErrorException("__setitem__ value must be a scalar or Tensor");
  }
  if (auto column_indices = full_row_slice_int_columns_key(key, tensor)) {
    double scalar = 0.0;
    if (pyobject_to_scalar(value, scalar)) {
      auto source = mtorch::make_tensor({scalar}, {}, tensor->dtype, false, tensor->device);
      mtorch::index_put_int_tensor_dim(tensor, column_indices, 1, *source);
      return;
    }
    if (is_tensor(value)) {
      mtorch::index_put_int_tensor_dim(tensor, column_indices, 1, *tensor_ref(value));
      return;
    }
  }

  MixedAdvancedKey mixed;
  const bool has_mixed = parse_mixed_advanced_key(key, *tensor, mixed);
  DimIntAdvancedKey dim_int;
  const bool has_dim_int = !has_mixed && parse_dim_int_advanced_key(key, tensor, dim_int);
  TensorPtr mask = has_mixed ? mixed.mask : bool_mask_from_key(key, tensor->device);
  TensorPtr int_indices = has_mixed ? mixed.int_indices : nullptr;
  if (!has_mixed && !has_dim_int && mask == nullptr) {
    int_indices = int_tensor_from_key(key, tensor->device);
  }
  const std::vector<mtorch::TensorIndex> empty_tail_indices;
  const auto& tail_indices = has_mixed ? mixed.tail_indices : empty_tail_indices;
  TensorPtr view;
  if (mask == nullptr && int_indices == nullptr && !has_dim_int) {
    view = mtorch::index(tensor, parse_tensor_indices(key, *tensor));
  }
  double scalar = 0.0;
  if (pyobject_to_scalar(value, scalar)) {
    if (mask != nullptr) {
      auto source = mtorch::make_tensor({scalar}, {}, tensor->dtype, false, tensor->device);
      mtorch::index_put_bool_mask(tensor, mask, *source, tail_indices);
    } else if (int_indices != nullptr) {
      auto source = mtorch::make_tensor({scalar}, {}, tensor->dtype, false, tensor->device);
      mtorch::index_put_int_tensor(tensor, int_indices, *source, tail_indices);
    } else if (has_dim_int) {
      auto source = mtorch::make_tensor({scalar}, {}, tensor->dtype, false, tensor->device);
      mtorch::index_put_int_tensor_dim(dim_int.base, dim_int.int_indices, dim_int.dim, *source);
    } else {
      view->fill_inplace(scalar);
    }
    return;
  }
  if (is_tensor(value)) {
    if (mask != nullptr) {
      mtorch::index_put_bool_mask(tensor, mask, *tensor_ref(value), tail_indices);
    } else if (int_indices != nullptr) {
      mtorch::index_put_int_tensor(tensor, int_indices, *tensor_ref(value), tail_indices);
    } else if (has_dim_int) {
      mtorch::index_put_int_tensor_dim(dim_int.base, dim_int.int_indices, dim_int.dim, *tensor_ref(value));
    } else {
      view->copy_from(*tensor_ref(value));
    }
    return;
  }
  if (object_is_sequence(value)) {
    throw TypeErrorException("can't assign a sequence to an mtorch.Tensor");
  }
  throw TypeErrorException("__setitem__ value must be a scalar or Tensor");
}

int Tensor_ass_subscript(PyObject* self, PyObject* key, PyObject* value) {
  try {
    assign_tensor_subscript(self, key, value);
    return 0;
  } catch (...) {
    translate_exception();
    return -1;
  }
}

PyObject* Tensor_getitem_method(PyTensor* self, PyObject* args) {
  PyObject* key = nullptr;
  if (!PyArg_ParseTuple(args, "O:__getitem__", &key)) {
    return nullptr;
  }
  return Tensor_subscript(reinterpret_cast<PyObject*>(self), key);
}

PyObject* Tensor_setitem_method(PyTensor* self, PyObject* args) {
  PyObject* key = nullptr;
  PyObject* value = nullptr;
  if (!PyArg_ParseTuple(args, "OO:__setitem__", &key, &value)) {
    return nullptr;
  }
  try {
    assign_tensor_subscript(reinterpret_cast<PyObject*>(self), key, value);
    Py_RETURN_NONE;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* tensor_method_binary_dispatch(PyTensor* self, PyObject* other, const std::string& op) {
  try {
    if (is_tensor(other)) {
      return wrap_tensor(mtorch::binary_tensor_tensor(*self->value, tensor_ref(other), op));
    }
    double scalar = 0.0;
    ScalarType scalar_dtype = ScalarType::Float32;
    if (pyobject_to_scalar(other, scalar, &scalar_dtype)) {
      return wrap_tensor(mtorch::binary_tensor_scalar(*self->value, scalar, scalar_dtype, op));
    }
    Py_RETURN_NOTIMPLEMENTED;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_add_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:add", &other)) {
    return nullptr;
  }
  return tensor_method_binary_dispatch(self, other, "add");
}

PyObject* Tensor_binary_method(PyTensor* self, PyObject* args, const std::string& op) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O", &other)) {
    return nullptr;
  }
  return tensor_method_binary_dispatch(self, other, op);
}

PyObject* Tensor_unary_method(PyTensor* self, const std::string& op) {
  return unary_dispatch(reinterpret_cast<PyObject*>(self), op);
}

using TensorKeywordForward = PyObject* (*)(PyObject*, PyObject*, PyObject*);

PyObject* Tensor_forward_keyword_method(
    PyTensor* self,
    PyObject* args,
    PyObject* kwargs,
    TensorKeywordForward function) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = function(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_neg_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "neg");
}

PyObject* Tensor_abs_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "abs");
}

PyObject* Tensor_exp_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "exp");
}

PyObject* Tensor_expm1_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "expm1");
}

PyObject* Tensor_log_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "log");
}

PyObject* Tensor_log1p_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "log1p");
}

PyObject* Tensor_log2_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "log2");
}

PyObject* Tensor_log10_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "log10");
}

PyObject* Tensor_sqrt_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "sqrt");
}

PyObject* Tensor_rsqrt_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "rsqrt");
}

PyObject* Tensor_reciprocal_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::reciprocal(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_sign_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "sign");
}

PyObject* Tensor_floor_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "floor");
}

PyObject* Tensor_ceil_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "ceil");
}

PyObject* Tensor_trunc_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "trunc");
}

PyObject* Tensor_round_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "round");
}

PyObject* Tensor_positive_method(PyTensor* self, PyObject*) {
  Py_INCREF(self);
  return reinterpret_cast<PyObject*>(self);
}

PyObject* Tensor_sin_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "sin");
}

PyObject* Tensor_cos_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "cos");
}

PyObject* Tensor_tan_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "tan");
}

PyObject* Tensor_sinh_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "sinh");
}

PyObject* Tensor_cosh_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "cosh");
}

PyObject* Tensor_tanh_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "tanh");
}

PyObject* Tensor_asin_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "asin");
}

PyObject* Tensor_acos_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "acos");
}

PyObject* Tensor_atan_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "atan");
}

PyObject* Tensor_sigmoid_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "sigmoid");
}

PyObject* Tensor_erf_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "erf");
}

PyObject* Tensor_erfc_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "erfc");
}

PyObject* Tensor_deg2rad_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::deg2rad(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_rad2deg_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::rad2deg(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_frac_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::frac(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_isnan_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "isnan"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_isinf_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "isinf"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_isfinite_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "isfinite"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_signbit_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "signbit"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_isposinf_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "isposinf"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_isneginf_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::unary_predicate(*self->value, "isneginf"));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_logical_not_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::logical_not(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_bitwise_not_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::bitwise_not(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_square_method(PyTensor* self, PyObject*) {
  return Tensor_unary_method(self, "square");
}

PyObject* Tensor_nan_to_num_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_nan_to_num);
}

PyObject* Tensor_clamp_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_clamp);
}

PyObject* Tensor_clip_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_clip);
}

PyObject* Tensor_clamp_min_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_clamp_min);
}

PyObject* Tensor_clamp_max_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_clamp_max);
}

PyObject* Tensor_softmax_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_softmax);
}

PyObject* Tensor_log_softmax_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_log_softmax);
}

PyObject* Tensor_norm_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"p", "dim", "keepdim", "dtype", nullptr};
  PyObject* p = Py_None;
  PyObject* dim = Py_None;
  int keepdim = 0;
  PyObject* dtype = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|OOpO:norm", const_cast<char**>(keywords), &p, &dim, &keepdim, &dtype)) {
    return nullptr;
  }
  try {
    const auto input = *self->value;
    return wrap_tensor(norm_tensor(
        input,
        norm_order_from_py(p),
        optional_dims_from_py(dim),
        keepdim != 0,
        dtype_from_py(dtype, input->dtype)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_relu_method(PyTensor* self, PyObject*) {
  try {
    return wrap_tensor(mtorch::relu(*self->value));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_sub_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "sub");
}

PyObject* Tensor_mul_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "mul");
}

PyObject* Tensor_div_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "div");
}

PyObject* Tensor_pow_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "pow");
}

PyObject* Tensor_floor_divide_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "floor_divide");
}

PyObject* Tensor_float_power_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "float_power");
}

PyObject* Tensor_remainder_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "remainder");
}

PyObject* Tensor_fmod_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "fmod");
}

PyObject* Tensor_atan2_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "atan2");
}

PyObject* Tensor_hypot_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "hypot");
}

PyObject* Tensor_ldexp_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "ldexp");
}

PyObject* Tensor_nextafter_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "nextafter");
}

PyObject* Tensor_copysign_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "copysign");
}

PyObject* Tensor_heaviside_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "heaviside");
}

PyObject* Tensor_logaddexp_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "logaddexp");
}

PyObject* Tensor_logaddexp2_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "logaddexp2");
}

PyObject* Tensor_xlogy_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "xlogy");
}

PyObject* Tensor_fmax_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "fmax");
}

PyObject* Tensor_fmin_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "fmin");
}

PyObject* Tensor_addcmul_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_addcmul);
}

PyObject* Tensor_addcdiv_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_addcdiv);
}

PyObject* Tensor_maximum_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "maximum");
}

PyObject* Tensor_minimum_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "minimum");
}

PyObject* Tensor_logical_and_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "logical_and");
}

PyObject* Tensor_logical_or_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "logical_or");
}

PyObject* Tensor_logical_xor_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "logical_xor");
}

PyObject* Tensor_bitwise_and_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "bitwise_and");
}

PyObject* Tensor_bitwise_or_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "bitwise_or");
}

PyObject* Tensor_bitwise_xor_method(PyTensor* self, PyObject* args) {
  return Tensor_binary_method(self, args, "bitwise_xor");
}

PyObject* Tensor_matmul_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:matmul", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "matmul expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    ensure_same_dtype_matrix_contraction("matmul", *self->value, right);
    return wrap_tensor(mtorch::matmul(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_mm_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:mm", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "mm expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    if ((*self->value)->dim() != 2 || right->dim() != 2) {
      throw std::runtime_error("mm expects two 2-D tensors");
    }
    ensure_same_dtype_matrix_contraction("mm", *self->value, right);
    return wrap_tensor(mtorch::matmul(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_bmm_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:bmm", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "bmm expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    ensure_same_dtype_matrix_contraction("bmm", *self->value, right);
    return wrap_tensor(mtorch::bmm(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_addmm_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"mat1", "mat2", "beta", "alpha", nullptr};
  PyObject* mat1 = nullptr;
  PyObject* mat2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd:addmm", const_cast<char**>(keywords), &mat1, &mat2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(mat1) || !is_tensor(mat2)) {
    PyErr_SetString(PyExc_TypeError, "addmm expected tensors");
    return nullptr;
  }
  try {
    const auto mat1_tensor = tensor_ref(mat1);
    const auto mat2_tensor = tensor_ref(mat2);
    ensure_all_same_dtype_matrix_contraction("addmm", {*self->value, mat1_tensor, mat2_tensor});
    return wrap_tensor(mtorch::addmm(*self->value, mat1_tensor, mat2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_addmv_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"mat", "vec", "beta", "alpha", nullptr};
  PyObject* mat = nullptr;
  PyObject* vec = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd:addmv", const_cast<char**>(keywords), &mat, &vec, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(mat) || !is_tensor(vec)) {
    PyErr_SetString(PyExc_TypeError, "addmv expected tensors");
    return nullptr;
  }
  try {
    const auto mat_tensor = tensor_ref(mat);
    const auto vec_tensor = tensor_ref(vec);
    ensure_same_dtype_matrix_contraction("addmv", mat_tensor, vec_tensor);
    return wrap_tensor(mtorch::addmv(*self->value, mat_tensor, vec_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_addr_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"vec1", "vec2", "beta", "alpha", nullptr};
  PyObject* vec1 = nullptr;
  PyObject* vec2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd:addr", const_cast<char**>(keywords), &vec1, &vec2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(vec1) || !is_tensor(vec2)) {
    PyErr_SetString(PyExc_TypeError, "addr expected tensors");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::addr(*self->value, tensor_ref(vec1), tensor_ref(vec2), beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_baddbmm_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"batch1", "batch2", "beta", "alpha", nullptr};
  PyObject* batch1 = nullptr;
  PyObject* batch2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd:baddbmm", const_cast<char**>(keywords), &batch1, &batch2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(batch1) || !is_tensor(batch2)) {
    PyErr_SetString(PyExc_TypeError, "baddbmm expected tensors");
    return nullptr;
  }
  try {
    const auto batch1_tensor = tensor_ref(batch1);
    const auto batch2_tensor = tensor_ref(batch2);
    ensure_all_same_dtype_matrix_contraction("baddbmm", {*self->value, batch1_tensor, batch2_tensor});
    return wrap_tensor(mtorch::baddbmm(*self->value, batch1_tensor, batch2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_addbmm_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"batch1", "batch2", "beta", "alpha", nullptr};
  PyObject* batch1 = nullptr;
  PyObject* batch2 = nullptr;
  double beta = 1.0;
  double alpha = 1.0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd:addbmm", const_cast<char**>(keywords), &batch1, &batch2, &beta, &alpha)) {
    return nullptr;
  }
  if (!is_tensor(batch1) || !is_tensor(batch2)) {
    PyErr_SetString(PyExc_TypeError, "addbmm expected tensors");
    return nullptr;
  }
  try {
    const auto batch1_tensor = tensor_ref(batch1);
    const auto batch2_tensor = tensor_ref(batch2);
    ensure_all_same_dtype_matrix_contraction("addbmm", {*self->value, batch1_tensor, batch2_tensor});
    return wrap_tensor(mtorch::addbmm(*self->value, batch1_tensor, batch2_tensor, beta, alpha));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_vdot_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:vdot", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "vdot expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    ensure_same_dtype_matrix_contraction("vdot", *self->value, right);
    return wrap_tensor(mtorch::vdot(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_inner_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:inner", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "inner expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    if (!(*self->value)->is_scalar() && !right->is_scalar()) {
      ensure_same_dtype_matrix_contraction("inner", *self->value, right);
    }
    return wrap_tensor(mtorch::inner(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_kron_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:kron", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "kron expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::kron(*self->value, tensor_ref(other)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_matrix_power_method(PyTensor* self, PyObject* args) {
  long long n = 0;
  if (!PyArg_ParseTuple(args, "L:matrix_power", &n)) {
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::matrix_power(*self->value, static_cast<int64_t>(n)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_dot_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:dot", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "dot expected Tensor");
    return nullptr;
  }
  try {
    const auto right = tensor_ref(other);
    ensure_same_dtype_matrix_contraction("dot", *self->value, right);
    return wrap_tensor(mtorch::dot(*self->value, right));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_mv_method(PyTensor* self, PyObject* args) {
  PyObject* vector = nullptr;
  if (!PyArg_ParseTuple(args, "O:mv", &vector)) {
    return nullptr;
  }
  if (!is_tensor(vector)) {
    PyErr_SetString(PyExc_TypeError, "mv expected Tensor");
    return nullptr;
  }
  try {
    const auto vector_tensor = tensor_ref(vector);
    ensure_same_dtype_matrix_contraction("mv", *self->value, vector_tensor);
    return wrap_tensor(mtorch::mv(*self->value, vector_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_outer_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:outer", &other)) {
    return nullptr;
  }
  if (!is_tensor(other)) {
    PyErr_SetString(PyExc_TypeError, "outer expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::outer(*self->value, tensor_ref(other)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_eq_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:eq", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "eq");
}

PyObject* Tensor_ne_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:ne", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "ne");
}

PyObject* Tensor_lt_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:lt", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "lt");
}

PyObject* Tensor_le_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:le", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "le");
}

PyObject* Tensor_gt_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:gt", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "gt");
}

PyObject* Tensor_ge_method(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:ge", &other)) {
    return nullptr;
  }
  return binary_dispatch(reinterpret_cast<PyObject*>(self), other, "ge");
}

PyObject* Tensor_isclose_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_isclose);
}

PyObject* Tensor_allclose_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_allclose);
}

PyObject* Tensor_equal_method(PyTensor* self, PyObject* args) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_equal(nullptr, tuple);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_is_nonzero_method(PyTensor* self, PyObject*) {
  PyObject* tuple = PyTuple_New(1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  PyObject* result = py_is_nonzero(nullptr, tuple);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_lerp_method(PyTensor* self, PyObject* args) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_lerp(nullptr, tuple);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_sum_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_sum);
}

PyObject* Tensor_cumsum_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_cumsum);
}

PyObject* Tensor_cumprod_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_cumprod);
}

PyObject* Tensor_cummax_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_cummax);
}

PyObject* Tensor_cummin_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_cummin);
}

PyObject* Tensor_mean_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_mean(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_prod_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_prod);
}

PyObject* Tensor_var_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_var);
}

PyObject* Tensor_std_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_std);
}

PyObject* Tensor_all_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_all(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_any_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_any);
}

PyObject* Tensor_amax_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_amax);
}

PyObject* Tensor_amin_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_amin);
}

PyObject* Tensor_max_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_max(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_argmax_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_argmax(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_min_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_min(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_argmin_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  PyObject* tuple = PyTuple_New(PyTuple_GET_SIZE(args) + 1);
  if (tuple == nullptr) {
    return nullptr;
  }
  Py_INCREF(self);
  PyTuple_SET_ITEM(tuple, 0, reinterpret_cast<PyObject*>(self));
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(args); ++i) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(tuple, i + 1, item);
  }
  PyObject* result = py_argmin(nullptr, tuple, kwargs);
  Py_DECREF(tuple);
  return result;
}

PyObject* Tensor_sort_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_sort);
}

PyObject* Tensor_argsort_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_argsort);
}

PyObject* Tensor_topk_method(PyTensor* self, PyObject* args, PyObject* kwargs) {
  return Tensor_forward_keyword_method(self, args, kwargs, py_topk);
}

PyObject* Tensor_add_inplace(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:add_", &other)) {
    return nullptr;
  }
  try {
    double scalar = 0.0;
    if (pyobject_to_scalar(other, scalar)) {
      self->value->get()->add_inplace(scalar);
    } else if (is_tensor(other)) {
      self->value->get()->copy_from(*mtorch::binary_tensor_tensor(*self->value, tensor_ref(other), "add"));
    } else {
      PyErr_SetString(PyExc_TypeError, "add_ expected scalar or Tensor");
      return nullptr;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_mul_inplace(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:mul_", &other)) {
    return nullptr;
  }
  try {
    double scalar = 0.0;
    if (pyobject_to_scalar(other, scalar)) {
      self->value->get()->mul_inplace(scalar);
    } else if (is_tensor(other)) {
      self->value->get()->copy_from(*mtorch::binary_tensor_tensor(*self->value, tensor_ref(other), "mul"));
    } else {
      PyErr_SetString(PyExc_TypeError, "mul_ expected scalar or Tensor");
      return nullptr;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_sub_inplace(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:sub_", &other)) {
    return nullptr;
  }
  try {
    double scalar = 0.0;
    if (pyobject_to_scalar(other, scalar)) {
      self->value->get()->add_inplace(-scalar);
    } else if (is_tensor(other)) {
      self->value->get()->copy_from(*mtorch::binary_tensor_tensor(*self->value, tensor_ref(other), "sub"));
    } else {
      PyErr_SetString(PyExc_TypeError, "sub_ expected scalar or Tensor");
      return nullptr;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_div_inplace(PyTensor* self, PyObject* args) {
  PyObject* other = nullptr;
  if (!PyArg_ParseTuple(args, "O:div_", &other)) {
    return nullptr;
  }
  try {
    double scalar = 0.0;
    if (pyobject_to_scalar(other, scalar)) {
      self->value->get()->mul_inplace(1.0 / scalar);
    } else if (is_tensor(other)) {
      self->value->get()->copy_from(*mtorch::binary_tensor_tensor(*self->value, tensor_ref(other), "div"));
    } else {
      PyErr_SetString(PyExc_TypeError, "div_ expected scalar or Tensor");
      return nullptr;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_zero_inplace(PyTensor* self, PyObject*) {
  try {
    self->value->get()->fill_inplace(0.0);
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_copy_inplace(PyTensor* self, PyObject* args) {
  PyObject* source = nullptr;
  if (!PyArg_ParseTuple(args, "O:copy_", &source)) {
    return nullptr;
  }
  if (!is_tensor(source)) {
    PyErr_SetString(PyExc_TypeError, "copy_ expected Tensor");
    return nullptr;
  }
  try {
    self->value->get()->copy_from(*tensor_ref(source));
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_fill_inplace(PyTensor* self, PyObject* args) {
  PyObject* value = nullptr;
  if (!PyArg_ParseTuple(args, "O:fill_", &value)) {
    return nullptr;
  }
  try {
    self->value->get()->fill_inplace(scalar_from_py(value));
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_normal_inplace(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"mean", "std", "generator", nullptr};
  double mean = 0.0;
  double std = 1.0;
  PyObject* generator = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|ddO:normal_", const_cast<char**>(keywords), &mean, &std, &generator)) {
    return nullptr;
  }
  try {
    normal_inplace(*self->value->get(), mean, std, random_state_from_generator(generator));
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_uniform_inplace(PyTensor* self, PyObject* args, PyObject* kwargs) {
  static const char* keywords[] = {"from", "to", "generator", nullptr};
  double from = 0.0;
  double to = 1.0;
  PyObject* generator = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|ddO:uniform_", const_cast<char**>(keywords), &from, &to, &generator)) {
    return nullptr;
  }
  try {
    uniform_inplace(*self->value->get(), from, to, random_state_from_generator(generator));
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_backward(PyTensor* self, PyObject* args) {
  PyObject* gradient = Py_None;
  if (!PyArg_ParseTuple(args, "|O:backward", &gradient)) {
    return nullptr;
  }
  try {
    if (gradient == Py_None) {
      self->value->get()->backward();
    } else {
      if (!is_tensor(gradient)) {
        PyErr_SetString(PyExc_TypeError, "backward gradient must be a Tensor or None");
        return nullptr;
      }
      self->value->get()->backward_with(*tensor_ref(gradient));
    }
    Py_RETURN_NONE;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_nb_add(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "add");
}

PyObject* Tensor_nb_subtract(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "sub");
}

PyObject* Tensor_nb_multiply(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "mul");
}

PyObject* Tensor_nb_matrix_multiply(PyObject* left, PyObject* right) {
  if (!is_tensor(left) || !is_tensor(right)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  try {
    const auto left_tensor = tensor_ref(left);
    const auto right_tensor = tensor_ref(right);
    ensure_same_dtype_matrix_contraction("matmul", left_tensor, right_tensor);
    return wrap_tensor(mtorch::matmul(left_tensor, right_tensor));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_nb_and(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "bitwise_and");
}

PyObject* Tensor_nb_or(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "bitwise_or");
}

PyObject* Tensor_nb_xor(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "bitwise_xor");
}

PyObject* Tensor_nb_true_divide(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "div");
}

PyObject* Tensor_nb_floor_divide(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "floor_divide");
}

PyObject* Tensor_nb_remainder(PyObject* left, PyObject* right) {
  return binary_dispatch(left, right, "remainder");
}

PyObject* Tensor_nb_power(PyObject* left, PyObject* right, PyObject*) {
  return binary_dispatch(left, right, "pow");
}

PyObject* Tensor_nb_negative(PyObject* object) {
  return unary_dispatch(object, "neg");
}

PyObject* Tensor_nb_positive(PyObject* object) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  Py_INCREF(object);
  return object;
}

PyObject* Tensor_nb_absolute(PyObject* object) {
  return unary_dispatch(object, "abs");
}

PyObject* Tensor_nb_invert(PyObject* object) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  try {
    return wrap_tensor(mtorch::bitwise_not(tensor_ref(object)));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_nb_inplace_arithmetic(PyObject* left, PyObject* right, const std::string& op, const char* name) {
  if (!is_tensor(left)) {
    Py_RETURN_NOTIMPLEMENTED;
  }
  auto* self = reinterpret_cast<PyTensor*>(left);
  try {
    double scalar = 0.0;
    if (pyobject_to_scalar(right, scalar)) {
      if (op == "add") {
        self->value->get()->add_inplace(scalar);
      } else if (op == "sub") {
        self->value->get()->add_inplace(-scalar);
      } else if (op == "mul") {
        self->value->get()->mul_inplace(scalar);
      } else if (op == "div") {
        self->value->get()->mul_inplace(1.0 / scalar);
      } else {
        PyErr_Format(PyExc_TypeError, "%s does not support scalar operands", name);
        return nullptr;
      }
    } else if (is_tensor(right)) {
      self->value->get()->copy_from(*mtorch::binary_tensor_tensor(*self->value, tensor_ref(right), op));
    } else {
      Py_RETURN_NOTIMPLEMENTED;
    }
    Py_INCREF(self);
    return reinterpret_cast<PyObject*>(self);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_nb_inplace_add(PyObject* left, PyObject* right) {
  return Tensor_nb_inplace_arithmetic(left, right, "add", "+=");
}

PyObject* Tensor_nb_inplace_subtract(PyObject* left, PyObject* right) {
  return Tensor_nb_inplace_arithmetic(left, right, "sub", "-=");
}

PyObject* Tensor_nb_inplace_multiply(PyObject* left, PyObject* right) {
  return Tensor_nb_inplace_arithmetic(left, right, "mul", "*=");
}

PyObject* Tensor_nb_inplace_true_divide(PyObject* left, PyObject* right) {
  return Tensor_nb_inplace_arithmetic(left, right, "div", "/=");
}

int Tensor_nb_bool(PyObject* object) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return -1;
  }
  try {
    return tensor_truth_value(tensor_ref(object));
  } catch (...) {
    translate_exception();
    return -1;
  }
}

Py_ssize_t Tensor_sq_length(PyObject* object) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return -1;
  }
  try {
    const auto& tensor = tensor_ref(object);
    if (tensor->dim() == 0) {
      PyErr_SetString(PyExc_TypeError, "len() of a 0-d tensor");
      return -1;
    }
    const int64_t length = tensor->sizes[0];
    if (length > static_cast<int64_t>(std::numeric_limits<Py_ssize_t>::max())) {
      PyErr_SetString(PyExc_OverflowError, "tensor length does not fit in Py_ssize_t");
      return -1;
    }
    return static_cast<Py_ssize_t>(length);
  } catch (...) {
    translate_exception();
    return -1;
  }
}

PyObject* Tensor_sq_item(PyObject* object, Py_ssize_t index) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  try {
    const auto& tensor = tensor_ref(object);
    if (tensor->dim() == 0) {
      PyErr_SetString(PyExc_TypeError, "iteration over a 0-d tensor");
      return nullptr;
    }
    int64_t normalized = static_cast<int64_t>(index);
    const int64_t length = tensor->sizes[0];
    if (normalized < 0) {
      normalized += length;
    }
    if (normalized < 0 || normalized >= length) {
      PyErr_SetString(PyExc_IndexError, "index out of range");
      return nullptr;
    }
    return wrap_tensor(mtorch::select(tensor, 0, normalized));
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_iter(PyObject* object) {
  if (!is_tensor(object)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor");
    return nullptr;
  }
  try {
    const auto& tensor = tensor_ref(object);
    if (tensor->dim() == 0) {
      PyErr_SetString(PyExc_TypeError, "iteration over a 0-d tensor");
      return nullptr;
    }
    return PySeqIter_New(object);
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

PyObject* Tensor_richcompare(PyObject* left, PyObject* right, int compare_op) {
  const char* op = nullptr;
  switch (compare_op) {
    case Py_LT:
      op = "lt";
      break;
    case Py_LE:
      op = "le";
      break;
    case Py_EQ:
      op = "eq";
      break;
    case Py_NE:
      op = "ne";
      break;
    case Py_GT:
      op = "gt";
      break;
    case Py_GE:
      op = "ge";
      break;
    default:
      Py_RETURN_NOTIMPLEMENTED;
  }
  return binary_dispatch(left, right, op);
}

PyGetSetDef Tensor_getset[] = {
    {"shape", reinterpret_cast<getter>(Tensor_get_shape), nullptr, nullptr, nullptr},
    {"ndim", reinterpret_cast<getter>(Tensor_get_ndim), nullptr, nullptr, nullptr},
    {"dtype", reinterpret_cast<getter>(Tensor_get_dtype), nullptr, nullptr, nullptr},
    {"device", reinterpret_cast<getter>(Tensor_get_device), nullptr, nullptr, nullptr},
    {"requires_grad", reinterpret_cast<getter>(Tensor_get_requires_grad), nullptr, nullptr, nullptr},
    {"grad", reinterpret_cast<getter>(Tensor_get_grad), reinterpret_cast<setter>(Tensor_set_grad), nullptr, nullptr},
    {"data", reinterpret_cast<getter>(Tensor_get_data), reinterpret_cast<setter>(Tensor_set_data), nullptr, nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef Tensor_methods[] = {
    {"tolist", reinterpret_cast<PyCFunction>(Tensor_tolist), METH_NOARGS, nullptr},
    {"item", reinterpret_cast<PyCFunction>(Tensor_item), METH_NOARGS, nullptr},
    {"size", reinterpret_cast<PyCFunction>(Tensor_size), METH_VARARGS, nullptr},
    {"dim", reinterpret_cast<PyCFunction>(Tensor_dim_method), METH_NOARGS, nullptr},
    {"numel", reinterpret_cast<PyCFunction>(Tensor_numel), METH_NOARGS, nullptr},
    {"stride", reinterpret_cast<PyCFunction>(Tensor_stride), METH_NOARGS, nullptr},
    {"element_size", reinterpret_cast<PyCFunction>(Tensor_element_size), METH_NOARGS, nullptr},
    {"is_contiguous", reinterpret_cast<PyCFunction>(Tensor_is_contiguous), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"is_floating_point", reinterpret_cast<PyCFunction>(Tensor_is_floating_point_method), METH_NOARGS, nullptr},
    {"is_complex", reinterpret_cast<PyCFunction>(Tensor_is_complex_method), METH_NOARGS, nullptr},
    {"is_conj", reinterpret_cast<PyCFunction>(Tensor_is_conj_method), METH_NOARGS, nullptr},
    {"is_signed", reinterpret_cast<PyCFunction>(Tensor_is_signed_method), METH_NOARGS, nullptr},
    {"clone", reinterpret_cast<PyCFunction>(Tensor_clone), METH_NOARGS, nullptr},
    {"contiguous", reinterpret_cast<PyCFunction>(Tensor_contiguous), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"detach", reinterpret_cast<PyCFunction>(Tensor_detach), METH_NOARGS, nullptr},
    {"detach_", reinterpret_cast<PyCFunction>(Tensor_detach_inplace), METH_NOARGS, nullptr},
    {"requires_grad_", reinterpret_cast<PyCFunction>(Tensor_requires_grad_inplace), METH_VARARGS, nullptr},
    {"cpu", reinterpret_cast<PyCFunction>(Tensor_cpu), METH_NOARGS, nullptr},
    {"to", reinterpret_cast<PyCFunction>(Tensor_to), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"float", reinterpret_cast<PyCFunction>(Tensor_float_method), METH_NOARGS, nullptr},
    {"double", reinterpret_cast<PyCFunction>(Tensor_double_method), METH_NOARGS, nullptr},
    {"half", reinterpret_cast<PyCFunction>(Tensor_half_method), METH_NOARGS, nullptr},
    {"long", reinterpret_cast<PyCFunction>(Tensor_long_method), METH_NOARGS, nullptr},
    {"int", reinterpret_cast<PyCFunction>(Tensor_int_method), METH_NOARGS, nullptr},
    {"bool", reinterpret_cast<PyCFunction>(Tensor_bool_method), METH_NOARGS, nullptr},
    {"type_as", reinterpret_cast<PyCFunction>(Tensor_type_as), METH_O, nullptr},
    {"new_tensor", reinterpret_cast<PyCFunction>(Tensor_new_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"new_empty", reinterpret_cast<PyCFunction>(Tensor_new_empty), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"new_zeros", reinterpret_cast<PyCFunction>(Tensor_new_zeros), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"new_ones", reinterpret_cast<PyCFunction>(Tensor_new_ones), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"new_full", reinterpret_cast<PyCFunction>(Tensor_new_full), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"reshape", reinterpret_cast<PyCFunction>(Tensor_reshape), METH_VARARGS, nullptr},
    {"view", reinterpret_cast<PyCFunction>(Tensor_view), METH_VARARGS, nullptr},
    {"reshape_as", reinterpret_cast<PyCFunction>(Tensor_reshape_as), METH_VARARGS, nullptr},
    {"view_as", reinterpret_cast<PyCFunction>(Tensor_view_as), METH_VARARGS, nullptr},
    {"unflatten", reinterpret_cast<PyCFunction>(Tensor_unflatten), METH_VARARGS, nullptr},
    {"transpose", reinterpret_cast<PyCFunction>(Tensor_transpose), METH_VARARGS, nullptr},
    {"permute", reinterpret_cast<PyCFunction>(Tensor_permute), METH_VARARGS, nullptr},
    {"movedim", reinterpret_cast<PyCFunction>(Tensor_movedim), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"moveaxis", reinterpret_cast<PyCFunction>(Tensor_movedim), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"swapaxes", reinterpret_cast<PyCFunction>(Tensor_swapaxes), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"swapdims", reinterpret_cast<PyCFunction>(Tensor_swapdims), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"flatten", reinterpret_cast<PyCFunction>(Tensor_flatten), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"ravel", reinterpret_cast<PyCFunction>(Tensor_ravel), METH_NOARGS, nullptr},
    {"t", reinterpret_cast<PyCFunction>(Tensor_t), METH_NOARGS, nullptr},
    {"expand", reinterpret_cast<PyCFunction>(Tensor_expand), METH_VARARGS, nullptr},
    {"expand_as", reinterpret_cast<PyCFunction>(Tensor_expand_as), METH_VARARGS, nullptr},
    {"repeat", reinterpret_cast<PyCFunction>(Tensor_repeat), METH_VARARGS, nullptr},
    {"repeat_interleave", reinterpret_cast<PyCFunction>(Tensor_repeat_interleave), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"tile", reinterpret_cast<PyCFunction>(Tensor_tile), METH_VARARGS, nullptr},
    {"flip", reinterpret_cast<PyCFunction>(Tensor_flip), METH_VARARGS, nullptr},
    {"fliplr", reinterpret_cast<PyCFunction>(Tensor_fliplr), METH_NOARGS, nullptr},
    {"flipud", reinterpret_cast<PyCFunction>(Tensor_flipud), METH_NOARGS, nullptr},
    {"rot90", reinterpret_cast<PyCFunction>(Tensor_rot90), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"roll", reinterpret_cast<PyCFunction>(Tensor_roll), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"squeeze", reinterpret_cast<PyCFunction>(Tensor_squeeze), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unsqueeze", reinterpret_cast<PyCFunction>(Tensor_unsqueeze), METH_VARARGS, nullptr},
    {"narrow", reinterpret_cast<PyCFunction>(Tensor_narrow), METH_VARARGS, nullptr},
    {"select", reinterpret_cast<PyCFunction>(Tensor_select), METH_VARARGS, nullptr},
    {"as_strided", reinterpret_cast<PyCFunction>(Tensor_as_strided), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diagonal", reinterpret_cast<PyCFunction>(Tensor_diagonal), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diag", reinterpret_cast<PyCFunction>(Tensor_diag), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diag_embed", reinterpret_cast<PyCFunction>(Tensor_diag_embed), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"tril", reinterpret_cast<PyCFunction>(Tensor_tril), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"triu", reinterpret_cast<PyCFunction>(Tensor_triu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"split", reinterpret_cast<PyCFunction>(Tensor_split), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"chunk", reinterpret_cast<PyCFunction>(Tensor_chunk), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unbind", reinterpret_cast<PyCFunction>(Tensor_unbind), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"__getitem__", reinterpret_cast<PyCFunction>(Tensor_getitem_method), METH_VARARGS, nullptr},
    {"__setitem__", reinterpret_cast<PyCFunction>(Tensor_setitem_method), METH_VARARGS, nullptr},
    {"index_select", reinterpret_cast<PyCFunction>(Tensor_index_select), METH_VARARGS, nullptr},
    {"gather", reinterpret_cast<PyCFunction>(Tensor_gather), METH_VARARGS, nullptr},
    {"scatter", reinterpret_cast<PyCFunction>(Tensor_scatter), METH_VARARGS, nullptr},
    {"scatter_", reinterpret_cast<PyCFunction>(Tensor_scatter_inplace), METH_VARARGS, nullptr},
    {"scatter_add", reinterpret_cast<PyCFunction>(Tensor_scatter_add), METH_VARARGS, nullptr},
    {"scatter_add_", reinterpret_cast<PyCFunction>(Tensor_scatter_add_inplace), METH_VARARGS, nullptr},
    {"take", reinterpret_cast<PyCFunction>(Tensor_take), METH_VARARGS, nullptr},
    {"masked_select", reinterpret_cast<PyCFunction>(Tensor_masked_select), METH_VARARGS, nullptr},
    {"masked_fill", reinterpret_cast<PyCFunction>(Tensor_masked_fill), METH_VARARGS, nullptr},
    {"masked_fill_", reinterpret_cast<PyCFunction>(Tensor_masked_fill_inplace), METH_VARARGS, nullptr},
    {"nonzero", reinterpret_cast<PyCFunction>(Tensor_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argwhere", reinterpret_cast<PyCFunction>(Tensor_argwhere), METH_NOARGS, nullptr},
    {"count_nonzero", reinterpret_cast<PyCFunction>(Tensor_count_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"neg", reinterpret_cast<PyCFunction>(Tensor_neg_method), METH_NOARGS, nullptr},
    {"negative", reinterpret_cast<PyCFunction>(Tensor_neg_method), METH_NOARGS, nullptr},
    {"abs", reinterpret_cast<PyCFunction>(Tensor_abs_method), METH_NOARGS, nullptr},
    {"absolute", reinterpret_cast<PyCFunction>(Tensor_abs_method), METH_NOARGS, nullptr},
    {"exp", reinterpret_cast<PyCFunction>(Tensor_exp_method), METH_NOARGS, nullptr},
    {"expm1", reinterpret_cast<PyCFunction>(Tensor_expm1_method), METH_NOARGS, nullptr},
    {"log", reinterpret_cast<PyCFunction>(Tensor_log_method), METH_NOARGS, nullptr},
    {"log1p", reinterpret_cast<PyCFunction>(Tensor_log1p_method), METH_NOARGS, nullptr},
    {"log2", reinterpret_cast<PyCFunction>(Tensor_log2_method), METH_NOARGS, nullptr},
    {"log10", reinterpret_cast<PyCFunction>(Tensor_log10_method), METH_NOARGS, nullptr},
    {"sqrt", reinterpret_cast<PyCFunction>(Tensor_sqrt_method), METH_NOARGS, nullptr},
    {"rsqrt", reinterpret_cast<PyCFunction>(Tensor_rsqrt_method), METH_NOARGS, nullptr},
    {"reciprocal", reinterpret_cast<PyCFunction>(Tensor_reciprocal_method), METH_NOARGS, nullptr},
    {"sign", reinterpret_cast<PyCFunction>(Tensor_sign_method), METH_NOARGS, nullptr},
    {"floor", reinterpret_cast<PyCFunction>(Tensor_floor_method), METH_NOARGS, nullptr},
    {"ceil", reinterpret_cast<PyCFunction>(Tensor_ceil_method), METH_NOARGS, nullptr},
    {"trunc", reinterpret_cast<PyCFunction>(Tensor_trunc_method), METH_NOARGS, nullptr},
    {"fix", reinterpret_cast<PyCFunction>(Tensor_trunc_method), METH_NOARGS, nullptr},
    {"round", reinterpret_cast<PyCFunction>(Tensor_round_method), METH_NOARGS, nullptr},
    {"positive", reinterpret_cast<PyCFunction>(Tensor_positive_method), METH_NOARGS, nullptr},
    {"sin", reinterpret_cast<PyCFunction>(Tensor_sin_method), METH_NOARGS, nullptr},
    {"arcsin", reinterpret_cast<PyCFunction>(Tensor_asin_method), METH_NOARGS, nullptr},
    {"cos", reinterpret_cast<PyCFunction>(Tensor_cos_method), METH_NOARGS, nullptr},
    {"arccos", reinterpret_cast<PyCFunction>(Tensor_acos_method), METH_NOARGS, nullptr},
    {"tan", reinterpret_cast<PyCFunction>(Tensor_tan_method), METH_NOARGS, nullptr},
    {"arctan", reinterpret_cast<PyCFunction>(Tensor_atan_method), METH_NOARGS, nullptr},
    {"sinh", reinterpret_cast<PyCFunction>(Tensor_sinh_method), METH_NOARGS, nullptr},
    {"cosh", reinterpret_cast<PyCFunction>(Tensor_cosh_method), METH_NOARGS, nullptr},
    {"tanh", reinterpret_cast<PyCFunction>(Tensor_tanh_method), METH_NOARGS, nullptr},
    {"asin", reinterpret_cast<PyCFunction>(Tensor_asin_method), METH_NOARGS, nullptr},
    {"acos", reinterpret_cast<PyCFunction>(Tensor_acos_method), METH_NOARGS, nullptr},
    {"atan", reinterpret_cast<PyCFunction>(Tensor_atan_method), METH_NOARGS, nullptr},
    {"sigmoid", reinterpret_cast<PyCFunction>(Tensor_sigmoid_method), METH_NOARGS, nullptr},
    {"erf", reinterpret_cast<PyCFunction>(Tensor_erf_method), METH_NOARGS, nullptr},
    {"erfc", reinterpret_cast<PyCFunction>(Tensor_erfc_method), METH_NOARGS, nullptr},
    {"deg2rad", reinterpret_cast<PyCFunction>(Tensor_deg2rad_method), METH_NOARGS, nullptr},
    {"rad2deg", reinterpret_cast<PyCFunction>(Tensor_rad2deg_method), METH_NOARGS, nullptr},
    {"frac", reinterpret_cast<PyCFunction>(Tensor_frac_method), METH_NOARGS, nullptr},
    {"isnan", reinterpret_cast<PyCFunction>(Tensor_isnan_method), METH_NOARGS, nullptr},
    {"isinf", reinterpret_cast<PyCFunction>(Tensor_isinf_method), METH_NOARGS, nullptr},
    {"isfinite", reinterpret_cast<PyCFunction>(Tensor_isfinite_method), METH_NOARGS, nullptr},
    {"signbit", reinterpret_cast<PyCFunction>(Tensor_signbit_method), METH_NOARGS, nullptr},
    {"isposinf", reinterpret_cast<PyCFunction>(Tensor_isposinf_method), METH_NOARGS, nullptr},
    {"isneginf", reinterpret_cast<PyCFunction>(Tensor_isneginf_method), METH_NOARGS, nullptr},
    {"logical_not", reinterpret_cast<PyCFunction>(Tensor_logical_not_method), METH_NOARGS, nullptr},
    {"bitwise_not", reinterpret_cast<PyCFunction>(Tensor_bitwise_not_method), METH_NOARGS, nullptr},
    {"square", reinterpret_cast<PyCFunction>(Tensor_square_method), METH_NOARGS, nullptr},
    {"nan_to_num", reinterpret_cast<PyCFunction>(Tensor_nan_to_num_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp", reinterpret_cast<PyCFunction>(Tensor_clamp_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clip", reinterpret_cast<PyCFunction>(Tensor_clip_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp_min", reinterpret_cast<PyCFunction>(Tensor_clamp_min_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp_max", reinterpret_cast<PyCFunction>(Tensor_clamp_max_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"softmax", reinterpret_cast<PyCFunction>(Tensor_softmax_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"log_softmax", reinterpret_cast<PyCFunction>(Tensor_log_softmax_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"norm", reinterpret_cast<PyCFunction>(Tensor_norm_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"relu", reinterpret_cast<PyCFunction>(Tensor_relu_method), METH_NOARGS, nullptr},
    {"add", reinterpret_cast<PyCFunction>(Tensor_add_method), METH_VARARGS, nullptr},
    {"subtract", reinterpret_cast<PyCFunction>(Tensor_sub_method), METH_VARARGS, nullptr},
    {"sub", reinterpret_cast<PyCFunction>(Tensor_sub_method), METH_VARARGS, nullptr},
    {"multiply", reinterpret_cast<PyCFunction>(Tensor_mul_method), METH_VARARGS, nullptr},
    {"mul", reinterpret_cast<PyCFunction>(Tensor_mul_method), METH_VARARGS, nullptr},
    {"divide", reinterpret_cast<PyCFunction>(Tensor_div_method), METH_VARARGS, nullptr},
    {"true_divide", reinterpret_cast<PyCFunction>(Tensor_div_method), METH_VARARGS, nullptr},
    {"div", reinterpret_cast<PyCFunction>(Tensor_div_method), METH_VARARGS, nullptr},
    {"pow", reinterpret_cast<PyCFunction>(Tensor_pow_method), METH_VARARGS, nullptr},
    {"floor_divide", reinterpret_cast<PyCFunction>(Tensor_floor_divide_method), METH_VARARGS, nullptr},
    {"float_power", reinterpret_cast<PyCFunction>(Tensor_float_power_method), METH_VARARGS, nullptr},
    {"remainder", reinterpret_cast<PyCFunction>(Tensor_remainder_method), METH_VARARGS, nullptr},
    {"fmod", reinterpret_cast<PyCFunction>(Tensor_fmod_method), METH_VARARGS, nullptr},
    {"atan2", reinterpret_cast<PyCFunction>(Tensor_atan2_method), METH_VARARGS, nullptr},
    {"arctan2", reinterpret_cast<PyCFunction>(Tensor_atan2_method), METH_VARARGS, nullptr},
    {"hypot", reinterpret_cast<PyCFunction>(Tensor_hypot_method), METH_VARARGS, nullptr},
    {"ldexp", reinterpret_cast<PyCFunction>(Tensor_ldexp_method), METH_VARARGS, nullptr},
    {"nextafter", reinterpret_cast<PyCFunction>(Tensor_nextafter_method), METH_VARARGS, nullptr},
    {"copysign", reinterpret_cast<PyCFunction>(Tensor_copysign_method), METH_VARARGS, nullptr},
    {"heaviside", reinterpret_cast<PyCFunction>(Tensor_heaviside_method), METH_VARARGS, nullptr},
    {"logaddexp", reinterpret_cast<PyCFunction>(Tensor_logaddexp_method), METH_VARARGS, nullptr},
    {"logaddexp2", reinterpret_cast<PyCFunction>(Tensor_logaddexp2_method), METH_VARARGS, nullptr},
    {"xlogy", reinterpret_cast<PyCFunction>(Tensor_xlogy_method), METH_VARARGS, nullptr},
    {"fmax", reinterpret_cast<PyCFunction>(Tensor_fmax_method), METH_VARARGS, nullptr},
    {"fmin", reinterpret_cast<PyCFunction>(Tensor_fmin_method), METH_VARARGS, nullptr},
    {"addcmul", reinterpret_cast<PyCFunction>(Tensor_addcmul_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addcdiv", reinterpret_cast<PyCFunction>(Tensor_addcdiv_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"maximum", reinterpret_cast<PyCFunction>(Tensor_maximum_method), METH_VARARGS, nullptr},
    {"minimum", reinterpret_cast<PyCFunction>(Tensor_minimum_method), METH_VARARGS, nullptr},
    {"logical_and", reinterpret_cast<PyCFunction>(Tensor_logical_and_method), METH_VARARGS, nullptr},
    {"logical_or", reinterpret_cast<PyCFunction>(Tensor_logical_or_method), METH_VARARGS, nullptr},
    {"logical_xor", reinterpret_cast<PyCFunction>(Tensor_logical_xor_method), METH_VARARGS, nullptr},
    {"bitwise_and", reinterpret_cast<PyCFunction>(Tensor_bitwise_and_method), METH_VARARGS, nullptr},
    {"bitwise_or", reinterpret_cast<PyCFunction>(Tensor_bitwise_or_method), METH_VARARGS, nullptr},
    {"bitwise_xor", reinterpret_cast<PyCFunction>(Tensor_bitwise_xor_method), METH_VARARGS, nullptr},
    {"matmul", reinterpret_cast<PyCFunction>(Tensor_matmul_method), METH_VARARGS, nullptr},
    {"mm", reinterpret_cast<PyCFunction>(Tensor_mm_method), METH_VARARGS, nullptr},
    {"bmm", reinterpret_cast<PyCFunction>(Tensor_bmm_method), METH_VARARGS, nullptr},
    {"addmm", reinterpret_cast<PyCFunction>(Tensor_addmm_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addmv", reinterpret_cast<PyCFunction>(Tensor_addmv_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addr", reinterpret_cast<PyCFunction>(Tensor_addr_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"baddbmm", reinterpret_cast<PyCFunction>(Tensor_baddbmm_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addbmm", reinterpret_cast<PyCFunction>(Tensor_addbmm_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"vdot", reinterpret_cast<PyCFunction>(Tensor_vdot_method), METH_VARARGS, nullptr},
    {"inner", reinterpret_cast<PyCFunction>(Tensor_inner_method), METH_VARARGS, nullptr},
    {"kron", reinterpret_cast<PyCFunction>(Tensor_kron_method), METH_VARARGS, nullptr},
    {"matrix_power", reinterpret_cast<PyCFunction>(Tensor_matrix_power_method), METH_VARARGS, nullptr},
    {"dot", reinterpret_cast<PyCFunction>(Tensor_dot_method), METH_VARARGS, nullptr},
    {"mv", reinterpret_cast<PyCFunction>(Tensor_mv_method), METH_VARARGS, nullptr},
    {"outer", reinterpret_cast<PyCFunction>(Tensor_outer_method), METH_VARARGS, nullptr},
    {"ger", reinterpret_cast<PyCFunction>(Tensor_outer_method), METH_VARARGS, nullptr},
    {"eq", reinterpret_cast<PyCFunction>(Tensor_eq_method), METH_VARARGS, nullptr},
    {"ne", reinterpret_cast<PyCFunction>(Tensor_ne_method), METH_VARARGS, nullptr},
    {"not_equal", reinterpret_cast<PyCFunction>(Tensor_ne_method), METH_VARARGS, nullptr},
    {"lt", reinterpret_cast<PyCFunction>(Tensor_lt_method), METH_VARARGS, nullptr},
    {"less", reinterpret_cast<PyCFunction>(Tensor_lt_method), METH_VARARGS, nullptr},
    {"le", reinterpret_cast<PyCFunction>(Tensor_le_method), METH_VARARGS, nullptr},
    {"less_equal", reinterpret_cast<PyCFunction>(Tensor_le_method), METH_VARARGS, nullptr},
    {"gt", reinterpret_cast<PyCFunction>(Tensor_gt_method), METH_VARARGS, nullptr},
    {"greater", reinterpret_cast<PyCFunction>(Tensor_gt_method), METH_VARARGS, nullptr},
    {"ge", reinterpret_cast<PyCFunction>(Tensor_ge_method), METH_VARARGS, nullptr},
    {"greater_equal", reinterpret_cast<PyCFunction>(Tensor_ge_method), METH_VARARGS, nullptr},
    {"isclose", reinterpret_cast<PyCFunction>(Tensor_isclose_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"allclose", reinterpret_cast<PyCFunction>(Tensor_allclose_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"equal", reinterpret_cast<PyCFunction>(Tensor_equal_method), METH_VARARGS, nullptr},
    {"is_nonzero", reinterpret_cast<PyCFunction>(Tensor_is_nonzero_method), METH_NOARGS, nullptr},
    {"lerp", reinterpret_cast<PyCFunction>(Tensor_lerp_method), METH_VARARGS, nullptr},
    {"sum", reinterpret_cast<PyCFunction>(Tensor_sum_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"trace", reinterpret_cast<PyCFunction>(Tensor_trace), METH_NOARGS, nullptr},
    {"cumsum", reinterpret_cast<PyCFunction>(Tensor_cumsum_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cumprod", reinterpret_cast<PyCFunction>(Tensor_cumprod_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cummax", reinterpret_cast<PyCFunction>(Tensor_cummax_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cummin", reinterpret_cast<PyCFunction>(Tensor_cummin_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"mean", reinterpret_cast<PyCFunction>(Tensor_mean_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"prod", reinterpret_cast<PyCFunction>(Tensor_prod_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"var", reinterpret_cast<PyCFunction>(Tensor_var_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"std", reinterpret_cast<PyCFunction>(Tensor_std_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"all", reinterpret_cast<PyCFunction>(Tensor_all_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"any", reinterpret_cast<PyCFunction>(Tensor_any_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"amax", reinterpret_cast<PyCFunction>(Tensor_amax_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"amin", reinterpret_cast<PyCFunction>(Tensor_amin_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"max", reinterpret_cast<PyCFunction>(Tensor_max_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argmax", reinterpret_cast<PyCFunction>(Tensor_argmax_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"min", reinterpret_cast<PyCFunction>(Tensor_min_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argmin", reinterpret_cast<PyCFunction>(Tensor_argmin_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"sort", reinterpret_cast<PyCFunction>(Tensor_sort_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argsort", reinterpret_cast<PyCFunction>(Tensor_argsort_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"topk", reinterpret_cast<PyCFunction>(Tensor_topk_method), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"add_", reinterpret_cast<PyCFunction>(Tensor_add_inplace), METH_VARARGS, nullptr},
    {"sub_", reinterpret_cast<PyCFunction>(Tensor_sub_inplace), METH_VARARGS, nullptr},
    {"mul_", reinterpret_cast<PyCFunction>(Tensor_mul_inplace), METH_VARARGS, nullptr},
    {"div_", reinterpret_cast<PyCFunction>(Tensor_div_inplace), METH_VARARGS, nullptr},
    {"zero_", reinterpret_cast<PyCFunction>(Tensor_zero_inplace), METH_NOARGS, nullptr},
    {"copy_", reinterpret_cast<PyCFunction>(Tensor_copy_inplace), METH_VARARGS, nullptr},
    {"fill_", reinterpret_cast<PyCFunction>(Tensor_fill_inplace), METH_VARARGS, nullptr},
    {"normal_", reinterpret_cast<PyCFunction>(Tensor_normal_inplace), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"uniform_", reinterpret_cast<PyCFunction>(Tensor_uniform_inplace), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"backward", reinterpret_cast<PyCFunction>(Tensor_backward), METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef module_methods[] = {
    {"tensor", reinterpret_cast<PyCFunction>(py_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"as_tensor", reinterpret_cast<PyCFunction>(py_as_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"zeros", reinterpret_cast<PyCFunction>(py_zeros), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"ones", reinterpret_cast<PyCFunction>(py_ones), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"empty", reinterpret_cast<PyCFunction>(py_empty), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"empty_strided", reinterpret_cast<PyCFunction>(py_empty_strided), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"full", reinterpret_cast<PyCFunction>(py_full), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"empty_like", reinterpret_cast<PyCFunction>(py_empty_like), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"zeros_like", reinterpret_cast<PyCFunction>(py_zeros_like), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"ones_like", reinterpret_cast<PyCFunction>(py_ones_like), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"full_like", reinterpret_cast<PyCFunction>(py_full_like), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"arange", reinterpret_cast<PyCFunction>(py_arange), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"linspace", reinterpret_cast<PyCFunction>(py_linspace), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"eye", reinterpret_cast<PyCFunction>(py_eye), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"randint", reinterpret_cast<PyCFunction>(py_randint), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"rand", reinterpret_cast<PyCFunction>(py_rand), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"randn", reinterpret_cast<PyCFunction>(py_randn), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"randperm", reinterpret_cast<PyCFunction>(py_randperm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"trunc_normal_", reinterpret_cast<PyCFunction>(py_trunc_normal_inplace), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"manual_seed", reinterpret_cast<PyCFunction>(py_manual_seed), METH_VARARGS, nullptr},
    {"initial_seed", reinterpret_cast<PyCFunction>(py_initial_seed), METH_NOARGS, nullptr},
    {"multinomial", reinterpret_cast<PyCFunction>(py_multinomial), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"bernoulli", reinterpret_cast<PyCFunction>(py_bernoulli), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"neg", py_neg, METH_VARARGS, nullptr},
    {"abs", py_abs, METH_VARARGS, nullptr},
    {"exp", py_exp, METH_VARARGS, nullptr},
    {"expm1", py_expm1, METH_VARARGS, nullptr},
    {"log", py_log, METH_VARARGS, nullptr},
    {"log1p", py_log1p, METH_VARARGS, nullptr},
    {"log2", py_log2, METH_VARARGS, nullptr},
    {"log10", py_log10, METH_VARARGS, nullptr},
    {"sqrt", py_sqrt, METH_VARARGS, nullptr},
    {"rsqrt", py_rsqrt, METH_VARARGS, nullptr},
    {"reciprocal", py_reciprocal, METH_VARARGS, nullptr},
    {"sign", py_sign, METH_VARARGS, nullptr},
    {"floor", py_floor, METH_VARARGS, nullptr},
    {"ceil", py_ceil, METH_VARARGS, nullptr},
    {"trunc", py_trunc, METH_VARARGS, nullptr},
    {"round", py_round, METH_VARARGS, nullptr},
    {"sin", py_sin, METH_VARARGS, nullptr},
    {"cos", py_cos, METH_VARARGS, nullptr},
    {"tan", py_tan, METH_VARARGS, nullptr},
    {"sinh", py_sinh, METH_VARARGS, nullptr},
    {"cosh", py_cosh, METH_VARARGS, nullptr},
    {"tanh", py_tanh, METH_VARARGS, nullptr},
    {"asin", py_asin, METH_VARARGS, nullptr},
    {"acos", py_acos, METH_VARARGS, nullptr},
    {"atan", py_atan, METH_VARARGS, nullptr},
    {"sigmoid", py_sigmoid, METH_VARARGS, nullptr},
    {"erf", py_erf, METH_VARARGS, nullptr},
    {"erfc", py_erfc, METH_VARARGS, nullptr},
    {"deg2rad", py_deg2rad, METH_VARARGS, nullptr},
    {"rad2deg", py_rad2deg, METH_VARARGS, nullptr},
    {"frac", py_frac, METH_VARARGS, nullptr},
    {"isnan", py_isnan, METH_VARARGS, nullptr},
    {"isinf", py_isinf, METH_VARARGS, nullptr},
    {"isfinite", py_isfinite, METH_VARARGS, nullptr},
    {"signbit", py_signbit, METH_VARARGS, nullptr},
    {"isposinf", py_isposinf, METH_VARARGS, nullptr},
    {"isneginf", py_isneginf, METH_VARARGS, nullptr},
    {"logical_not", py_logical_not, METH_VARARGS, nullptr},
    {"bitwise_not", py_bitwise_not, METH_VARARGS, nullptr},
    {"square", py_square, METH_VARARGS, nullptr},
    {"nan_to_num", reinterpret_cast<PyCFunction>(py_nan_to_num), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"gelu", reinterpret_cast<PyCFunction>(py_gelu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp", reinterpret_cast<PyCFunction>(py_clamp), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clip", reinterpret_cast<PyCFunction>(py_clip), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp_min", reinterpret_cast<PyCFunction>(py_clamp_min), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clamp_max", reinterpret_cast<PyCFunction>(py_clamp_max), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"softmax", reinterpret_cast<PyCFunction>(py_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"log_softmax", reinterpret_cast<PyCFunction>(py_log_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"norm", reinterpret_cast<PyCFunction>(py_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_normalize_l2", reinterpret_cast<PyCFunction>(py_normalize_l2), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"add", py_add, METH_VARARGS, nullptr},
    {"sub", py_sub, METH_VARARGS, nullptr},
    {"mul", py_mul, METH_VARARGS, nullptr},
    {"div", py_div, METH_VARARGS, nullptr},
    {"pow", py_pow, METH_VARARGS, nullptr},
    {"floor_divide", py_floor_divide, METH_VARARGS, nullptr},
    {"float_power", py_float_power, METH_VARARGS, nullptr},
    {"remainder", py_remainder, METH_VARARGS, nullptr},
    {"fmod", py_fmod, METH_VARARGS, nullptr},
    {"atan2", py_atan2, METH_VARARGS, nullptr},
    {"hypot", py_hypot, METH_VARARGS, nullptr},
    {"ldexp", py_ldexp, METH_VARARGS, nullptr},
    {"nextafter", py_nextafter, METH_VARARGS, nullptr},
    {"copysign", py_copysign, METH_VARARGS, nullptr},
    {"heaviside", py_heaviside, METH_VARARGS, nullptr},
    {"logaddexp", py_logaddexp, METH_VARARGS, nullptr},
    {"logaddexp2", py_logaddexp2, METH_VARARGS, nullptr},
    {"xlogy", py_xlogy, METH_VARARGS, nullptr},
    {"fmax", py_fmax, METH_VARARGS, nullptr},
    {"fmin", py_fmin, METH_VARARGS, nullptr},
    {"addcmul", reinterpret_cast<PyCFunction>(py_addcmul), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addcdiv", reinterpret_cast<PyCFunction>(py_addcdiv), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"maximum", py_maximum, METH_VARARGS, nullptr},
    {"minimum", py_minimum, METH_VARARGS, nullptr},
    {"eq", py_eq, METH_VARARGS, nullptr},
    {"ne", py_ne, METH_VARARGS, nullptr},
    {"lt", py_lt, METH_VARARGS, nullptr},
    {"le", py_le, METH_VARARGS, nullptr},
    {"gt", py_gt, METH_VARARGS, nullptr},
    {"ge", py_ge, METH_VARARGS, nullptr},
    {"logical_and", py_logical_and, METH_VARARGS, nullptr},
    {"logical_or", py_logical_or, METH_VARARGS, nullptr},
    {"logical_xor", py_logical_xor, METH_VARARGS, nullptr},
    {"bitwise_and", py_bitwise_and, METH_VARARGS, nullptr},
    {"bitwise_or", py_bitwise_or, METH_VARARGS, nullptr},
    {"bitwise_xor", py_bitwise_xor, METH_VARARGS, nullptr},
    {"isclose", reinterpret_cast<PyCFunction>(py_isclose), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"allclose", reinterpret_cast<PyCFunction>(py_allclose), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"equal", py_equal, METH_VARARGS, nullptr},
    {"is_nonzero", py_is_nonzero, METH_VARARGS, nullptr},
    {"lerp", py_lerp, METH_VARARGS, nullptr},
    {"sum", reinterpret_cast<PyCFunction>(py_sum), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_diff_float32", reinterpret_cast<PyCFunction>(py_diff_float32), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"trace", py_trace, METH_VARARGS, nullptr},
    {"cumsum", reinterpret_cast<PyCFunction>(py_cumsum), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cumprod", reinterpret_cast<PyCFunction>(py_cumprod), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cummax", reinterpret_cast<PyCFunction>(py_cummax), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cummin", reinterpret_cast<PyCFunction>(py_cummin), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_trapezoid_dx", reinterpret_cast<PyCFunction>(py_trapezoid_dx), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_cumulative_trapezoid_dx", reinterpret_cast<PyCFunction>(py_cumulative_trapezoid_dx), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_gradient_uniform", reinterpret_cast<PyCFunction>(py_gradient_uniform), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"mean", reinterpret_cast<PyCFunction>(py_mean), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"prod", reinterpret_cast<PyCFunction>(py_prod), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"var", reinterpret_cast<PyCFunction>(py_var), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"std", reinterpret_cast<PyCFunction>(py_std), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_var_tail", reinterpret_cast<PyCFunction>(py_var_tail), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_std_tail", reinterpret_cast<PyCFunction>(py_std_tail), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_var_mean", reinterpret_cast<PyCFunction>(py_var_mean), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_std_mean", reinterpret_cast<PyCFunction>(py_std_mean), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"all", reinterpret_cast<PyCFunction>(py_all), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"any", reinterpret_cast<PyCFunction>(py_any), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"amax", reinterpret_cast<PyCFunction>(py_amax), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"amin", reinterpret_cast<PyCFunction>(py_amin), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"max", reinterpret_cast<PyCFunction>(py_max), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argmax", reinterpret_cast<PyCFunction>(py_argmax), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"min", reinterpret_cast<PyCFunction>(py_min), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argmin", reinterpret_cast<PyCFunction>(py_argmin), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"sort", reinterpret_cast<PyCFunction>(py_sort), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argsort", reinterpret_cast<PyCFunction>(py_argsort), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_quantile_flat", reinterpret_cast<PyCFunction>(py_quantile_flat), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"_quantile_dim_2d", reinterpret_cast<PyCFunction>(py_quantile_dim_2d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"topk", reinterpret_cast<PyCFunction>(py_topk), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"searchsorted", reinterpret_cast<PyCFunction>(py_searchsorted), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unique", reinterpret_cast<PyCFunction>(py_unique), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unique_consecutive", reinterpret_cast<PyCFunction>(py_unique_consecutive), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"reshape", py_reshape, METH_VARARGS, nullptr},
    {"unflatten", py_unflatten, METH_VARARGS, nullptr},
    {"transpose", py_transpose, METH_VARARGS, nullptr},
    {"permute", py_permute, METH_VARARGS, nullptr},
    {"movedim", reinterpret_cast<PyCFunction>(py_movedim), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"moveaxis", reinterpret_cast<PyCFunction>(py_movedim), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"swapaxes", reinterpret_cast<PyCFunction>(py_swapaxes), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"swapdims", reinterpret_cast<PyCFunction>(py_swapdims), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"flatten", reinterpret_cast<PyCFunction>(py_flatten), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"ravel", py_ravel, METH_VARARGS, nullptr},
    {"t", py_t, METH_VARARGS, nullptr},
    {"broadcast_to", py_broadcast_to, METH_VARARGS, nullptr},
    {"broadcast_shapes", py_broadcast_shapes, METH_VARARGS, nullptr},
    {"broadcast_tensors", py_broadcast_tensors, METH_VARARGS, nullptr},
    {"tile", py_tile, METH_VARARGS, nullptr},
    {"repeat_interleave", reinterpret_cast<PyCFunction>(py_repeat_interleave), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"pad", reinterpret_cast<PyCFunction>(py_pad), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"adaptive_avg_pool1d", py_adaptive_avg_pool1d, METH_VARARGS, nullptr},
    {"adaptive_avg_pool2d", py_adaptive_avg_pool2d, METH_VARARGS, nullptr},
    {"pixel_shuffle", py_pixel_shuffle, METH_VARARGS, nullptr},
    {"pixel_unshuffle", py_pixel_unshuffle, METH_VARARGS, nullptr},
    {"channel_shuffle", py_channel_shuffle, METH_VARARGS, nullptr},
    {"interpolate", reinterpret_cast<PyCFunction>(py_interpolate), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"grid_sample", reinterpret_cast<PyCFunction>(py_grid_sample), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"affine_grid", reinterpret_cast<PyCFunction>(py_affine_grid), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"flip", py_flip, METH_VARARGS, nullptr},
    {"fliplr", py_fliplr, METH_VARARGS, nullptr},
    {"flipud", py_flipud, METH_VARARGS, nullptr},
    {"rot90", reinterpret_cast<PyCFunction>(py_rot90), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"roll", reinterpret_cast<PyCFunction>(py_roll), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"squeeze", reinterpret_cast<PyCFunction>(py_squeeze), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unsqueeze", py_unsqueeze, METH_VARARGS, nullptr},
    {"narrow", py_narrow, METH_VARARGS, nullptr},
    {"select", py_select, METH_VARARGS, nullptr},
    {"as_strided", reinterpret_cast<PyCFunction>(py_as_strided), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diagonal", reinterpret_cast<PyCFunction>(py_diagonal), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diag", reinterpret_cast<PyCFunction>(py_diag), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diagflat", reinterpret_cast<PyCFunction>(py_diagflat), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"diag_embed", reinterpret_cast<PyCFunction>(py_diag_embed), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"block_diag", py_block_diag, METH_VARARGS, nullptr},
    {"tril", reinterpret_cast<PyCFunction>(py_tril), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"triu", reinterpret_cast<PyCFunction>(py_triu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"split", reinterpret_cast<PyCFunction>(py_split), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"chunk", reinterpret_cast<PyCFunction>(py_chunk), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unbind", reinterpret_cast<PyCFunction>(py_unbind), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cat", reinterpret_cast<PyCFunction>(py_cat), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"stack", reinterpret_cast<PyCFunction>(py_stack), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"hstack", py_hstack, METH_O, nullptr},
    {"vstack", py_vstack, METH_O, nullptr},
    {"row_stack", py_vstack, METH_O, nullptr},
    {"dstack", py_dstack, METH_O, nullptr},
    {"column_stack", py_column_stack, METH_O, nullptr},
    {"cartesian_prod", py_cartesian_prod, METH_VARARGS, nullptr},
    {"matmul", py_matmul, METH_VARARGS, nullptr},
    {"mm", py_mm, METH_VARARGS, nullptr},
    {"einsum", py_einsum, METH_VARARGS, nullptr},
    {"bmm", py_bmm, METH_VARARGS, nullptr},
    {"addmm", reinterpret_cast<PyCFunction>(py_addmm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addmv", reinterpret_cast<PyCFunction>(py_addmv), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addr", reinterpret_cast<PyCFunction>(py_addr), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"baddbmm", reinterpret_cast<PyCFunction>(py_baddbmm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"addbmm", reinterpret_cast<PyCFunction>(py_addbmm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"vdot", py_vdot, METH_VARARGS, nullptr},
    {"inner", py_inner, METH_VARARGS, nullptr},
    {"tensordot", reinterpret_cast<PyCFunction>(py_tensordot), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"kron", py_kron, METH_VARARGS, nullptr},
    {"chain_matmul", py_chain_matmul, METH_VARARGS, nullptr},
    {"matrix_power", py_matrix_power, METH_VARARGS, nullptr},
    {"dot", py_dot, METH_VARARGS, nullptr},
    {"mv", py_mv, METH_VARARGS, nullptr},
    {"outer", py_outer, METH_VARARGS, nullptr},
    {"ger", py_outer, METH_VARARGS, nullptr},
    {"linear", reinterpret_cast<PyCFunction>(py_linear), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv1d", reinterpret_cast<PyCFunction>(py_conv1d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv_transpose1d", reinterpret_cast<PyCFunction>(py_conv_transpose1d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv2d", reinterpret_cast<PyCFunction>(py_conv2d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv3d", reinterpret_cast<PyCFunction>(py_conv3d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv_transpose2d", reinterpret_cast<PyCFunction>(py_conv_transpose2d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"conv_transpose3d", reinterpret_cast<PyCFunction>(py_conv_transpose3d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"max_pool1d", reinterpret_cast<PyCFunction>(py_max_pool1d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"avg_pool1d", reinterpret_cast<PyCFunction>(py_avg_pool1d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"max_pool2d", reinterpret_cast<PyCFunction>(py_max_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"avg_pool2d", reinterpret_cast<PyCFunction>(py_avg_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"unfold", reinterpret_cast<PyCFunction>(py_unfold), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"fold", reinterpret_cast<PyCFunction>(py_fold), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"scaled_dot_product_attention", reinterpret_cast<PyCFunction>(py_scaled_dot_product_attention), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"layer_norm", reinterpret_cast<PyCFunction>(py_layer_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"rms_norm", reinterpret_cast<PyCFunction>(py_rms_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"batch_norm", reinterpret_cast<PyCFunction>(py_batch_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"group_norm", reinterpret_cast<PyCFunction>(py_group_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"embedding", reinterpret_cast<PyCFunction>(py_embedding), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"dropout", reinterpret_cast<PyCFunction>(py_dropout), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"mse_loss", reinterpret_cast<PyCFunction>(py_mse_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"l1_loss", reinterpret_cast<PyCFunction>(py_l1_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"nll_loss", reinterpret_cast<PyCFunction>(py_nll_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cross_entropy", reinterpret_cast<PyCFunction>(py_cross_entropy), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"binary_cross_entropy", reinterpret_cast<PyCFunction>(py_binary_cross_entropy), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"binary_cross_entropy_with_logits", reinterpret_cast<PyCFunction>(py_binary_cross_entropy_with_logits), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"where", py_where, METH_VARARGS, nullptr},
    {"isin", reinterpret_cast<PyCFunction>(py_isin), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"take", py_take, METH_VARARGS, nullptr},
    {"index_select", py_index_select, METH_VARARGS, nullptr},
    {"gather", py_gather, METH_VARARGS, nullptr},
    {"index_put", reinterpret_cast<PyCFunction>(py_index_put), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"scatter", py_scatter, METH_VARARGS, nullptr},
    {"scatter_add", py_scatter_add, METH_VARARGS, nullptr},
    {"masked_select", py_masked_select, METH_VARARGS, nullptr},
    {"masked_fill", py_masked_fill, METH_VARARGS, nullptr},
    {"nonzero", reinterpret_cast<PyCFunction>(py_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"argwhere", py_argwhere, METH_VARARGS, nullptr},
    {"count_nonzero", reinterpret_cast<PyCFunction>(py_count_nonzero), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"bincount", reinterpret_cast<PyCFunction>(py_bincount), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"one_hot", py_one_hot, METH_VARARGS, nullptr},
    {"relu", py_relu, METH_VARARGS, nullptr},
    {"leaky_relu", reinterpret_cast<PyCFunction>(py_leaky_relu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"silu", reinterpret_cast<PyCFunction>(py_silu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"elu", reinterpret_cast<PyCFunction>(py_elu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"selu", reinterpret_cast<PyCFunction>(py_selu), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"softplus", reinterpret_cast<PyCFunction>(py_softplus), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"hardtanh", reinterpret_cast<PyCFunction>(py_hardtanh), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"relu6", reinterpret_cast<PyCFunction>(py_relu6), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"hardsigmoid", reinterpret_cast<PyCFunction>(py_hardsigmoid), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"hardswish", reinterpret_cast<PyCFunction>(py_hardswish), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"softsign", py_softsign, METH_VARARGS, nullptr},
    {"mish", reinterpret_cast<PyCFunction>(py_mish), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"clone", py_clone, METH_VARARGS, nullptr},
    {"numel", py_numel, METH_VARARGS, nullptr},
    {"is_tensor", py_is_tensor, METH_VARARGS, nullptr},
    {"_mark_parameter", py_mark_parameter, METH_VARARGS, nullptr},
    {"_is_parameter", py_is_parameter, METH_VARARGS, nullptr},
    {"is_floating_point", py_is_floating_point, METH_VARARGS, nullptr},
    {"is_complex", py_is_complex, METH_VARARGS, nullptr},
    {"is_conj", py_is_conj, METH_VARARGS, nullptr},
    {"is_signed", py_is_signed, METH_VARARGS, nullptr},
    {"_is_grad_enabled", py_is_grad_enabled, METH_NOARGS, nullptr},
    {"_set_grad_enabled", py_set_grad_enabled, METH_VARARGS, nullptr},
    {"_autograd_grad", reinterpret_cast<PyCFunction>(py_autograd_grad), METH_VARARGS | METH_KEYWORDS, nullptr},
    {"save", py_not_implemented, METH_VARARGS, nullptr},
    {"load", py_not_implemented, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef Generator_methods[] = {
    {"manual_seed", reinterpret_cast<PyCFunction>(Generator_manual_seed), METH_VARARGS, nullptr},
    {"initial_seed", reinterpret_cast<PyCFunction>(Generator_initial_seed), METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyType_Slot Generator_slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(Generator_new)},
    {Py_tp_init, reinterpret_cast<void*>(Generator_init)},
    {Py_tp_repr, reinterpret_cast<void*>(Generator_repr)},
    {Py_tp_methods, reinterpret_cast<void*>(Generator_methods)},
    {0, nullptr},
};

PyType_Spec Generator_spec = {
    "mtorch.Generator",
    sizeof(PyGenerator),
    0,
    Py_TPFLAGS_DEFAULT,
    Generator_slots,
};

PyType_Slot Tensor_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(Tensor_dealloc)},
    {Py_tp_new, reinterpret_cast<void*>(Tensor_new)},
    {Py_tp_repr, reinterpret_cast<void*>(Tensor_repr)},
    {Py_tp_methods, reinterpret_cast<void*>(Tensor_methods)},
    {Py_tp_getset, reinterpret_cast<void*>(Tensor_getset)},
    {Py_tp_richcompare, reinterpret_cast<void*>(Tensor_richcompare)},
    {Py_tp_iter, reinterpret_cast<void*>(Tensor_iter)},
    {Py_sq_length, reinterpret_cast<void*>(Tensor_sq_length)},
    {Py_sq_item, reinterpret_cast<void*>(Tensor_sq_item)},
    {Py_mp_subscript, reinterpret_cast<void*>(Tensor_subscript)},
    {Py_mp_ass_subscript, reinterpret_cast<void*>(Tensor_ass_subscript)},
    {Py_nb_add, reinterpret_cast<void*>(Tensor_nb_add)},
    {Py_nb_subtract, reinterpret_cast<void*>(Tensor_nb_subtract)},
    {Py_nb_multiply, reinterpret_cast<void*>(Tensor_nb_multiply)},
    {Py_nb_matrix_multiply, reinterpret_cast<void*>(Tensor_nb_matrix_multiply)},
    {Py_nb_and, reinterpret_cast<void*>(Tensor_nb_and)},
    {Py_nb_or, reinterpret_cast<void*>(Tensor_nb_or)},
    {Py_nb_xor, reinterpret_cast<void*>(Tensor_nb_xor)},
    {Py_nb_true_divide, reinterpret_cast<void*>(Tensor_nb_true_divide)},
    {Py_nb_floor_divide, reinterpret_cast<void*>(Tensor_nb_floor_divide)},
    {Py_nb_remainder, reinterpret_cast<void*>(Tensor_nb_remainder)},
    {Py_nb_power, reinterpret_cast<void*>(Tensor_nb_power)},
    {Py_nb_negative, reinterpret_cast<void*>(Tensor_nb_negative)},
    {Py_nb_positive, reinterpret_cast<void*>(Tensor_nb_positive)},
    {Py_nb_absolute, reinterpret_cast<void*>(Tensor_nb_absolute)},
    {Py_nb_invert, reinterpret_cast<void*>(Tensor_nb_invert)},
    {Py_nb_inplace_add, reinterpret_cast<void*>(Tensor_nb_inplace_add)},
    {Py_nb_inplace_subtract, reinterpret_cast<void*>(Tensor_nb_inplace_subtract)},
    {Py_nb_inplace_multiply, reinterpret_cast<void*>(Tensor_nb_inplace_multiply)},
    {Py_nb_inplace_true_divide, reinterpret_cast<void*>(Tensor_nb_inplace_true_divide)},
    {Py_nb_bool, reinterpret_cast<void*>(Tensor_nb_bool)},
    {0, nullptr},
};

PyType_Spec Tensor_spec = {
    "mtorch.Tensor",
    sizeof(PyTensor),
    0,
    Py_TPFLAGS_DEFAULT,
    Tensor_slots,
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "mtorch._C",
    "Native C++ core bindings for mtorch.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__C() {
  PyObject* module = PyModule_Create(&module_def);
  if (module == nullptr) {
    return nullptr;
  }

  TensorType = PyType_FromSpec(&Tensor_spec);
  if (TensorType == nullptr) {
    Py_DECREF(module);
    return nullptr;
  }

  Py_INCREF(TensorType);
  if (PyModule_AddObject(module, "Tensor", TensorType) < 0) {
    Py_DECREF(TensorType);
    Py_DECREF(module);
    return nullptr;
  }

  GeneratorType = PyType_FromSpec(&Generator_spec);
  if (GeneratorType == nullptr) {
    Py_DECREF(module);
    return nullptr;
  }

  Py_INCREF(GeneratorType);
  if (PyModule_AddObject(module, "Generator", GeneratorType) < 0) {
    Py_DECREF(GeneratorType);
    Py_DECREF(module);
    return nullptr;
  }

  return module;
}
