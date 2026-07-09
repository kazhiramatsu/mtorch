#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mtorch {

enum class ScalarType {
  Float16,
  Float32,
  Float64,
  Int32,
  Int64,
  Bool,
};

enum class DeviceType {
  CPU,
  Metal,
};

struct Device {
  DeviceType type = DeviceType::CPU;
  int64_t index = -1;
};

struct Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

enum class TensorIndexKind {
  Select,
  Slice,
  NewAxis,
};

struct TensorIndex {
  TensorIndexKind kind;
  int64_t index = 0;
  int64_t start = 0;
  int64_t length = 0;
  int64_t step = 1;
};

int64_t element_size(ScalarType dtype);
std::string dtype_name(ScalarType dtype);
ScalarType promote_dtype(ScalarType left, ScalarType right);
std::string device_type_name(DeviceType type);
std::string device_name(const Device& device);
Device cpu_device();
Device metal_device(int64_t index = 0);
bool devices_equal(const Device& left, const Device& right);
bool is_grad_enabled();
bool set_grad_enabled(bool enabled);

struct Storage {
  ScalarType dtype;
  Device device;
  std::vector<uint8_t> bytes;
  uint64_t version = 0;
  mutable uint64_t half_float_cache_version = ~uint64_t{0};
  mutable std::vector<float> half_float_cache;
  mutable uint64_t half_conv2d_nhwc_weight_cache_version = ~uint64_t{0};
  mutable int64_t half_conv2d_nhwc_weight_cache_offset = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_numel = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_out_channels = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_in_channels = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_kernel_height = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_kernel_width = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_stride0 = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_stride1 = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_stride2 = -1;
  mutable int64_t half_conv2d_nhwc_weight_cache_stride3 = -1;
  mutable std::vector<float> half_conv2d_nhwc_weight_cache;

  Storage(ScalarType dtype, int64_t elements, Device device = Device{}, bool zero_initialize = true);

  int64_t numel() const;
  double get(int64_t index) const;
  void set(int64_t index, double value);
};

struct Tensor {
  std::shared_ptr<Storage> storage;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset = 0;
  ScalarType dtype = ScalarType::Float32;
  Device device;
  bool requires_grad = false;
  TensorPtr grad;
  std::vector<TensorPtr> parents;
  std::function<void(const Tensor&)> backward_fn;

  Tensor(
      std::shared_ptr<Storage> storage,
      std::vector<int64_t> sizes,
      std::vector<int64_t> strides,
      int64_t offset,
      ScalarType dtype,
      bool requires_grad);

  int64_t dim() const;
  int64_t numel() const;
  int64_t element_size() const;
  bool is_scalar() const;
  bool is_contiguous() const;
  double value_at_index(const std::vector<int64_t>& index) const;
  void set_at_index(const std::vector<int64_t>& index, double value);
  double value_at_linear(int64_t linear) const;
  void set_at_linear(int64_t linear, double value);
  std::vector<double> contiguous_values() const;
  TensorPtr clone() const;
  TensorPtr contiguous() const;
  void fill_inplace(double value);
  void add_inplace(double value);
  void mul_inplace(double value);
  void copy_from(const Tensor& source);
  void mark_storage_modified();
  void backward();
  void backward_with(const Tensor& upstream);
};

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sizes);
TensorPtr make_tensor(
    const std::vector<double>& values,
    const std::vector<int64_t>& sizes,
    ScalarType dtype,
    bool requires_grad = false,
    Device device = Device{});
TensorPtr full(const std::vector<int64_t>& sizes, double value, ScalarType dtype, Device device = Device{});
TensorPtr zeros(const std::vector<int64_t>& sizes, ScalarType dtype, Device device = Device{});
TensorPtr ones(const std::vector<int64_t>& sizes, ScalarType dtype, Device device = Device{});
TensorPtr empty_strided(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    ScalarType dtype,
    bool requires_grad = false,
    Device device = Device{});
TensorPtr empty_like(const TensorPtr& input, ScalarType dtype, Device device, bool requires_grad = false);
TensorPtr zeros_like(const TensorPtr& input, ScalarType dtype, Device device, bool requires_grad = false);
TensorPtr ones_like(const TensorPtr& input, ScalarType dtype, Device device, bool requires_grad = false);
TensorPtr full_like(
    const TensorPtr& input,
    double value,
    ScalarType dtype,
    Device device,
    bool requires_grad = false);
TensorPtr arange(double start, double end, double step, ScalarType dtype, Device device = Device{});
TensorPtr linspace(double start, double end, int64_t steps, ScalarType dtype, Device device = Device{});
TensorPtr eye(int64_t n, ScalarType dtype, Device device = Device{});

TensorPtr to(const TensorPtr& input, ScalarType dtype, Device device, bool copy = false);
TensorPtr reshape(const TensorPtr& input, const std::vector<int64_t>& sizes);
TensorPtr unflatten(const TensorPtr& input, int64_t dim, const std::vector<int64_t>& sizes);
TensorPtr transpose(const TensorPtr& input, int64_t dim0, int64_t dim1);
TensorPtr permute(const TensorPtr& input, const std::vector<int64_t>& dims);
TensorPtr movedim(const TensorPtr& input, const std::vector<int64_t>& source, const std::vector<int64_t>& destination);
TensorPtr flatten(const TensorPtr& input, int64_t start_dim = 0, int64_t end_dim = -1);
TensorPtr ravel(const TensorPtr& input);
TensorPtr t(const TensorPtr& input);
TensorPtr expand(const TensorPtr& input, const std::vector<int64_t>& sizes);
TensorPtr broadcast_to(const TensorPtr& input, const std::vector<int64_t>& sizes);
TensorPtr repeat(const TensorPtr& input, const std::vector<int64_t>& repeats);
TensorPtr repeat_interleave(
    const TensorPtr& input,
    int64_t repeats,
    std::optional<int64_t> dim = std::nullopt,
    std::optional<int64_t> output_size = std::nullopt);
TensorPtr repeat_interleave(
    const TensorPtr& input,
    const TensorPtr& repeats,
    std::optional<int64_t> dim = std::nullopt,
    std::optional<int64_t> output_size = std::nullopt);
TensorPtr tile(const TensorPtr& input, const std::vector<int64_t>& repeats);
TensorPtr pad(const TensorPtr& input, const std::vector<int64_t>& padding, double value = 0.0);
TensorPtr pad(const TensorPtr& input, const std::vector<int64_t>& padding, const std::string& mode, double value = 0.0);
TensorPtr adaptive_avg_pool1d(const TensorPtr& input, const std::vector<int64_t>& output_size);
TensorPtr adaptive_avg_pool2d(const TensorPtr& input, const std::vector<int64_t>& output_size);
TensorPtr pixel_shuffle(const TensorPtr& input, int64_t upscale_factor);
TensorPtr pixel_unshuffle(const TensorPtr& input, int64_t downscale_factor);
TensorPtr channel_shuffle(const TensorPtr& input, int64_t groups);
TensorPtr interpolate(
    const TensorPtr& input,
    const std::vector<int64_t>& size,
    const std::string& mode = "nearest",
    bool align_corners = false);
TensorPtr grid_sample(
    const TensorPtr& input,
    const TensorPtr& grid,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool align_corners = false);
TensorPtr affine_grid(
    const TensorPtr& theta,
    const std::vector<int64_t>& size,
    bool align_corners = false);
TensorPtr flip(const TensorPtr& input, const std::vector<int64_t>& dims);
TensorPtr fliplr(const TensorPtr& input);
TensorPtr flipud(const TensorPtr& input);
TensorPtr rot90(const TensorPtr& input, int64_t k = 1, const std::vector<int64_t>& dims = {0, 1});
TensorPtr roll(const TensorPtr& input, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims = {});
std::vector<int64_t> broadcast_shapes(const std::vector<std::vector<int64_t>>& shapes);
std::vector<TensorPtr> broadcast_tensors(const std::vector<TensorPtr>& tensors);
TensorPtr squeeze(const TensorPtr& input);
TensorPtr squeeze(const TensorPtr& input, int64_t dim);
TensorPtr unsqueeze(const TensorPtr& input, int64_t dim);
TensorPtr narrow(const TensorPtr& input, int64_t dim, int64_t start, int64_t length);
TensorPtr select(const TensorPtr& input, int64_t dim, int64_t index);
TensorPtr as_strided(
    const TensorPtr& input,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    std::optional<int64_t> storage_offset = std::nullopt);
TensorPtr diagonal(const TensorPtr& input, int64_t offset = 0, int64_t dim1 = 0, int64_t dim2 = 1);
TensorPtr diag(const TensorPtr& input, int64_t diagonal = 0);
TensorPtr diagflat(const TensorPtr& input, int64_t offset = 0);
TensorPtr diag_embed(const TensorPtr& input, int64_t offset = 0, int64_t dim1 = -2, int64_t dim2 = -1);
TensorPtr block_diag(const std::vector<TensorPtr>& tensors);
TensorPtr tril(const TensorPtr& input, int64_t diagonal = 0);
TensorPtr triu(const TensorPtr& input, int64_t diagonal = 0);
TensorPtr trace(const TensorPtr& input);
std::vector<TensorPtr> split(const TensorPtr& input, int64_t split_size, int64_t dim = 0);
std::vector<TensorPtr> split(const TensorPtr& input, const std::vector<int64_t>& split_sizes, int64_t dim = 0);
std::vector<TensorPtr> chunk(const TensorPtr& input, int64_t chunks, int64_t dim = 0);
std::vector<TensorPtr> unbind(const TensorPtr& input, int64_t dim = 0);
TensorPtr index(const TensorPtr& input, const std::vector<TensorIndex>& indices);
TensorPtr index_integer_tuple(const TensorPtr& input, const std::vector<TensorPtr>& indices);
TensorPtr index_bool_mask(
    const TensorPtr& input,
    const TensorPtr& mask,
    const std::vector<TensorIndex>& tail_indices = {});
void index_put_bool_mask(
    const TensorPtr& input,
    const TensorPtr& mask,
    const Tensor& source,
    const std::vector<TensorIndex>& tail_indices = {});
TensorPtr index_int_tensor(
    const TensorPtr& input,
    const TensorPtr& indices,
    const std::vector<TensorIndex>& tail_indices = {});
void index_put_int_tensor(
    const TensorPtr& input,
    const TensorPtr& indices,
    const Tensor& source,
    const std::vector<TensorIndex>& tail_indices = {});
TensorPtr index_int_tensor_dim(const TensorPtr& input, const TensorPtr& indices, int64_t dim);
void index_put_int_tensor_dim(const TensorPtr& input, const TensorPtr& indices, int64_t dim, const Tensor& source);
void index_put_integer_tuple(
    const TensorPtr& input,
    const std::vector<TensorPtr>& indices,
    const Tensor& source,
    bool accumulate = false);
TensorPtr index_put(const TensorPtr& input, const std::vector<TensorPtr>& indices, const Tensor& source, bool accumulate = false);
TensorPtr one_hot(const TensorPtr& input, int64_t num_classes = -1);
TensorPtr index_select(const TensorPtr& input, int64_t dim, const TensorPtr& indices);
TensorPtr gather(const TensorPtr& input, int64_t dim, const TensorPtr& indices);
TensorPtr masked_select(const TensorPtr& input, const TensorPtr& mask);
TensorPtr masked_fill(const TensorPtr& input, const TensorPtr& mask, double value);
TensorPtr nonzero(const TensorPtr& input);
std::vector<TensorPtr> nonzero_tuple(const TensorPtr& input);
TensorPtr count_nonzero(const TensorPtr& input);
TensorPtr count_nonzero_dim(const TensorPtr& input, int64_t dim);
TensorPtr bincount(const TensorPtr& input, const TensorPtr& weights = nullptr, int64_t minlength = 0);
TensorPtr isin(const TensorPtr& elements, const TensorPtr& test_elements, bool assume_unique = false, bool invert = false);
TensorPtr scatter(const TensorPtr& input, int64_t dim, const TensorPtr& indices, const Tensor& source);
void scatter_inplace(const TensorPtr& input, int64_t dim, const TensorPtr& indices, const Tensor& source);
TensorPtr scatter_add(const TensorPtr& input, int64_t dim, const TensorPtr& indices, const Tensor& source);
void scatter_add_inplace(const TensorPtr& input, int64_t dim, const TensorPtr& indices, const Tensor& source);

TensorPtr unary(const TensorPtr& input, const std::string& op);
TensorPtr reciprocal(const TensorPtr& input);
TensorPtr unary_predicate(const TensorPtr& input, const std::string& op);
TensorPtr logical_not(const TensorPtr& input);
TensorPtr bitwise_not(const TensorPtr& input);
TensorPtr deg2rad(const TensorPtr& input);
TensorPtr rad2deg(const TensorPtr& input);
TensorPtr frac(const TensorPtr& input);
TensorPtr nan_to_num(
    const TensorPtr& input,
    double nan = 0.0,
    std::optional<double> posinf = std::nullopt,
    std::optional<double> neginf = std::nullopt);
TensorPtr relu(const TensorPtr& input);
TensorPtr leaky_relu(const TensorPtr& input, double negative_slope = 0.01);
TensorPtr silu(const TensorPtr& input);
TensorPtr elu(const TensorPtr& input, double alpha = 1.0);
TensorPtr selu(const TensorPtr& input);
TensorPtr softplus(const TensorPtr& input, double beta = 1.0, double threshold = 20.0);
TensorPtr hardtanh(const TensorPtr& input, double min_value = -1.0, double max_value = 1.0);
TensorPtr relu6(const TensorPtr& input);
TensorPtr hardsigmoid(const TensorPtr& input);
TensorPtr hardswish(const TensorPtr& input);
TensorPtr softsign(const TensorPtr& input);
TensorPtr mish(const TensorPtr& input);
TensorPtr gelu(const TensorPtr& input, const std::string& approximate = "none");
TensorPtr layer_norm(
    const TensorPtr& input,
    const std::vector<int64_t>& normalized_shape,
    const TensorPtr& weight = nullptr,
    const TensorPtr& bias = nullptr,
    double eps = 1e-5);
TensorPtr rms_norm(
    const TensorPtr& input,
    const std::vector<int64_t>& normalized_shape,
    const TensorPtr& weight = nullptr,
    double eps = 1e-5);
TensorPtr normalize_l2(const TensorPtr& input, int64_t dim, double eps = 1e-12);
TensorPtr batch_norm(
    const TensorPtr& input,
    const TensorPtr& running_mean = nullptr,
    const TensorPtr& running_var = nullptr,
    const TensorPtr& weight = nullptr,
    const TensorPtr& bias = nullptr,
    bool training = false,
    double momentum = 0.1,
    double eps = 1e-5);
TensorPtr group_norm(
    const TensorPtr& input,
    int64_t num_groups,
    const TensorPtr& weight = nullptr,
    const TensorPtr& bias = nullptr,
    double eps = 1e-5);
TensorPtr clamp(const TensorPtr& input, std::optional<double> min, std::optional<double> max);
TensorPtr clamp_min(const TensorPtr& input, double min);
TensorPtr clamp_max(const TensorPtr& input, double max);
TensorPtr softmax(const TensorPtr& input, int64_t dim, ScalarType dtype);
TensorPtr log_softmax(const TensorPtr& input, int64_t dim, ScalarType dtype);
TensorPtr mse_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction = "mean");
TensorPtr l1_loss(const TensorPtr& input, const TensorPtr& target, const std::string& reduction = "mean");
TensorPtr nll_loss(
    const TensorPtr& input,
    const TensorPtr& target,
    const std::string& reduction = "mean",
    int64_t ignore_index = -100,
    const TensorPtr& weight = nullptr);
TensorPtr cross_entropy_loss(
    const TensorPtr& input,
    const TensorPtr& target,
    const std::string& reduction = "mean",
    int64_t ignore_index = -100,
    double label_smoothing = 0.0,
    const TensorPtr& weight = nullptr);
TensorPtr binary_cross_entropy_loss(
    const TensorPtr& input,
    const TensorPtr& target,
    const std::string& reduction = "mean",
    const TensorPtr& weight = nullptr);
TensorPtr binary_cross_entropy_with_logits_loss(
    const TensorPtr& input,
    const TensorPtr& target,
    const std::string& reduction = "mean",
    const TensorPtr& weight = nullptr,
    const TensorPtr& pos_weight = nullptr);
TensorPtr binary_tensor_tensor(const TensorPtr& left, const TensorPtr& right, const std::string& op);
TensorPtr binary_tensor_scalar(const TensorPtr& left, double scalar, ScalarType scalar_dtype, const std::string& op);
TensorPtr binary_scalar_tensor(double scalar, ScalarType scalar_dtype, const TensorPtr& right, const std::string& op);
TensorPtr isclose(
    const TensorPtr& left,
    const TensorPtr& right,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false);
bool allclose(
    const TensorPtr& left,
    const TensorPtr& right,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false);
TensorPtr lerp(const TensorPtr& input, const TensorPtr& end, const TensorPtr& weight);
TensorPtr lerp(const TensorPtr& input, const TensorPtr& end, double weight, ScalarType weight_dtype = ScalarType::Float32);
TensorPtr addcmul(
    const TensorPtr& input,
    const TensorPtr& tensor1,
    const TensorPtr& tensor2,
    double value = 1.0);
TensorPtr addcdiv(
    const TensorPtr& input,
    const TensorPtr& tensor1,
    const TensorPtr& tensor2,
    double value = 1.0);
TensorPtr reduce_sum(const TensorPtr& input, ScalarType dtype);
TensorPtr reduce_sum_dim(const TensorPtr& input, int64_t dim, bool keepdim, ScalarType dtype);
TensorPtr diff_float32(const TensorPtr& input, int64_t dim);
TensorPtr cumsum(const TensorPtr& input, int64_t dim, ScalarType dtype);
TensorPtr cumprod(const TensorPtr& input, int64_t dim, ScalarType dtype);
std::pair<TensorPtr, TensorPtr> cummax(const TensorPtr& input, int64_t dim);
std::pair<TensorPtr, TensorPtr> cummin(const TensorPtr& input, int64_t dim);
TensorPtr trapezoid_dx(const TensorPtr& input, double dx, int64_t dim);
TensorPtr cumulative_trapezoid_dx(const TensorPtr& input, double dx, int64_t dim);
TensorPtr gradient_uniform(const TensorPtr& input, double spacing, int64_t dim, int64_t edge_order);
TensorPtr reduce_mean(const TensorPtr& input, ScalarType dtype);
TensorPtr reduce_mean_dim(const TensorPtr& input, int64_t dim, bool keepdim, ScalarType dtype);
TensorPtr reduce_prod(const TensorPtr& input, ScalarType dtype);
TensorPtr reduce_prod_dim(const TensorPtr& input, int64_t dim, bool keepdim, ScalarType dtype);
TensorPtr reduce_var(const TensorPtr& input, double correction);
TensorPtr reduce_var_dim(const TensorPtr& input, int64_t dim, bool keepdim, double correction);
TensorPtr reduce_var_tail(const TensorPtr& input, int64_t start_dim, bool keepdim, double correction);
TensorPtr reduce_std(const TensorPtr& input, double correction);
TensorPtr reduce_std_dim(const TensorPtr& input, int64_t dim, bool keepdim, double correction);
TensorPtr reduce_std_tail(const TensorPtr& input, int64_t start_dim, bool keepdim, double correction);
std::pair<TensorPtr, TensorPtr> reduce_var_mean(const TensorPtr& input, double correction);
std::pair<TensorPtr, TensorPtr> reduce_var_mean_dim(const TensorPtr& input, int64_t dim, bool keepdim, double correction);
std::pair<TensorPtr, TensorPtr> reduce_std_mean(const TensorPtr& input, double correction);
std::pair<TensorPtr, TensorPtr> reduce_std_mean_dim(const TensorPtr& input, int64_t dim, bool keepdim, double correction);
TensorPtr reduce_all(const TensorPtr& input);
TensorPtr reduce_all_dim(const TensorPtr& input, int64_t dim, bool keepdim);
TensorPtr reduce_any(const TensorPtr& input);
TensorPtr reduce_any_dim(const TensorPtr& input, int64_t dim, bool keepdim);
TensorPtr reduce_max(const TensorPtr& input);
TensorPtr amax(const TensorPtr& input, const std::vector<int64_t>& dims, bool keepdim = false);
std::pair<TensorPtr, TensorPtr> reduce_max_dim(const TensorPtr& input, int64_t dim, bool keepdim);
TensorPtr argmax(const TensorPtr& input, bool keepdim);
TensorPtr argmax_dim(const TensorPtr& input, int64_t dim, bool keepdim);
TensorPtr reduce_min(const TensorPtr& input);
TensorPtr amin(const TensorPtr& input, const std::vector<int64_t>& dims, bool keepdim = false);
std::pair<TensorPtr, TensorPtr> reduce_min_dim(const TensorPtr& input, int64_t dim, bool keepdim);
TensorPtr argmin(const TensorPtr& input, bool keepdim);
TensorPtr argmin_dim(const TensorPtr& input, int64_t dim, bool keepdim);
std::pair<TensorPtr, TensorPtr> sort(const TensorPtr& input, int64_t dim = -1, bool descending = false, bool stable = false);
TensorPtr argsort(const TensorPtr& input, int64_t dim = -1, bool descending = false, bool stable = false);
TensorPtr quantile_flat(const TensorPtr& input, double q, const std::string& interpolation);
TensorPtr quantile_dim_2d(const TensorPtr& input, double q, int64_t dim, const std::string& interpolation);
std::pair<TensorPtr, TensorPtr> topk(
    const TensorPtr& input,
    int64_t k,
    int64_t dim = -1,
    bool largest = true,
    bool sorted = true);
TensorPtr searchsorted(const TensorPtr& sorted_sequence, const TensorPtr& values, bool out_int32 = false, bool right = false);
struct UniqueResult {
  TensorPtr output;
  TensorPtr inverse_indices;
  TensorPtr counts;
};
UniqueResult unique(
    const TensorPtr& input,
    bool sorted = true,
    bool return_inverse = false,
    bool return_counts = false,
    std::optional<int64_t> dim = std::nullopt);
UniqueResult unique_consecutive(
    const TensorPtr& input,
    bool return_inverse = false,
    bool return_counts = false,
    std::optional<int64_t> dim = std::nullopt);
TensorPtr dot(const TensorPtr& left, const TensorPtr& right);
TensorPtr vdot(const TensorPtr& left, const TensorPtr& right);
TensorPtr inner(const TensorPtr& left, const TensorPtr& right);
TensorPtr tensordot(
    const TensorPtr& left,
    const TensorPtr& right,
    const std::vector<int64_t>& left_dims,
    const std::vector<int64_t>& right_dims);
TensorPtr kron(const TensorPtr& left, const TensorPtr& right);
TensorPtr mv(const TensorPtr& matrix, const TensorPtr& vector);
TensorPtr outer(const TensorPtr& left, const TensorPtr& right);
TensorPtr matmul(const TensorPtr& left, const TensorPtr& right);
TensorPtr bmm(const TensorPtr& left, const TensorPtr& right);
TensorPtr addmm(const TensorPtr& input, const TensorPtr& mat1, const TensorPtr& mat2, double beta = 1.0, double alpha = 1.0);
TensorPtr addmv(const TensorPtr& input, const TensorPtr& mat, const TensorPtr& vec, double beta = 1.0, double alpha = 1.0);
TensorPtr addr(const TensorPtr& input, const TensorPtr& vec1, const TensorPtr& vec2, double beta = 1.0, double alpha = 1.0);
TensorPtr baddbmm(const TensorPtr& input, const TensorPtr& batch1, const TensorPtr& batch2, double beta = 1.0, double alpha = 1.0);
TensorPtr addbmm(const TensorPtr& input, const TensorPtr& batch1, const TensorPtr& batch2, double beta = 1.0, double alpha = 1.0);
TensorPtr chain_matmul(const std::vector<TensorPtr>& matrices);
TensorPtr matrix_power(const TensorPtr& input, int64_t n);
TensorPtr linear(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias = nullptr);
TensorPtr conv1d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1},
    const std::vector<int64_t>& padding = {0},
    const std::vector<int64_t>& dilation = {1},
    int64_t groups = 1);
TensorPtr conv2d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1, 1},
    const std::vector<int64_t>& padding = {0, 0},
    const std::vector<int64_t>& dilation = {1, 1},
    int64_t groups = 1);
TensorPtr conv3d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1, 1, 1},
    const std::vector<int64_t>& padding = {0, 0, 0},
    const std::vector<int64_t>& dilation = {1, 1, 1},
    int64_t groups = 1);
TensorPtr conv_transpose1d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1},
    const std::vector<int64_t>& padding = {0},
    const std::vector<int64_t>& output_padding = {0},
    int64_t groups = 1,
    const std::vector<int64_t>& dilation = {1});
TensorPtr conv_transpose2d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1, 1},
    const std::vector<int64_t>& padding = {0, 0},
    const std::vector<int64_t>& output_padding = {0, 0},
    int64_t groups = 1,
    const std::vector<int64_t>& dilation = {1, 1});
TensorPtr conv_transpose3d(
    const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& bias = nullptr,
    const std::vector<int64_t>& stride = {1, 1, 1},
    const std::vector<int64_t>& padding = {0, 0, 0},
    const std::vector<int64_t>& output_padding = {0, 0, 0},
    int64_t groups = 1,
    const std::vector<int64_t>& dilation = {1, 1, 1});
TensorPtr max_pool1d(
    const TensorPtr& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride = {},
    const std::vector<int64_t>& padding = {0},
    const std::vector<int64_t>& dilation = {1},
    bool ceil_mode = false);
TensorPtr avg_pool1d(
    const TensorPtr& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride = {},
    const std::vector<int64_t>& padding = {0},
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int64_t> divisor_override = std::nullopt);
TensorPtr max_pool2d(
    const TensorPtr& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride = {},
    const std::vector<int64_t>& padding = {0, 0},
    const std::vector<int64_t>& dilation = {1, 1},
    bool ceil_mode = false);
TensorPtr avg_pool2d(
    const TensorPtr& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride = {},
    const std::vector<int64_t>& padding = {0, 0},
    bool ceil_mode = false,
    bool count_include_pad = true,
    std::optional<int64_t> divisor_override = std::nullopt);
TensorPtr unfold2d(
    const TensorPtr& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation = {1, 1},
    const std::vector<int64_t>& padding = {0, 0},
    const std::vector<int64_t>& stride = {1, 1});
TensorPtr fold2d(
    const TensorPtr& input,
    const std::vector<int64_t>& output_size,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation = {1, 1},
    const std::vector<int64_t>& padding = {0, 0},
    const std::vector<int64_t>& stride = {1, 1});
TensorPtr scaled_dot_product_attention(
    const TensorPtr& query,
    const TensorPtr& key,
    const TensorPtr& value,
    const TensorPtr& attn_mask = nullptr,
    double dropout_p = 0.0,
    bool is_causal = false,
    std::optional<double> scale = std::nullopt,
    bool enable_gqa = false);
TensorPtr embedding(
    const TensorPtr& weight,
    const TensorPtr& indices,
    std::optional<int64_t> padding_idx = std::nullopt,
    std::optional<double> max_norm = std::nullopt,
    double norm_type = 2.0,
    bool scale_grad_by_freq = false);
TensorPtr cat(const std::vector<TensorPtr>& tensors, int64_t dim);
TensorPtr cat_pair(const TensorPtr& left, const TensorPtr& right, int64_t dim);
TensorPtr stack(const std::vector<TensorPtr>& tensors, int64_t dim);
TensorPtr hstack(const std::vector<TensorPtr>& tensors);
TensorPtr vstack(const std::vector<TensorPtr>& tensors);
TensorPtr dstack(const std::vector<TensorPtr>& tensors);
TensorPtr column_stack(const std::vector<TensorPtr>& tensors);
TensorPtr cartesian_prod(const std::vector<TensorPtr>& tensors);
TensorPtr where(const TensorPtr& condition, const TensorPtr& left, const TensorPtr& right);
TensorPtr take(const TensorPtr& input, const TensorPtr& indices);

}  // namespace mtorch
