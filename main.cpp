#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include "csrc/macros.h"

// Function to preprocess the image (placeholder)
std::tuple<cv::Mat, int, int> preprocess_image(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  int h = img.rows;
  int w = img.cols;
  cv::Mat tt, t;
  cv::resize(img, tt,
             cv::Size(DEPTH_PRO_FIXED_RESOLUTION, DEPTH_PRO_FIXED_RESOLUTION));

  tt.convertTo(t, CV_32F, 1.0 / 255);
  cv::Mat mean = cv::Mat(t.size(), CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
  cv::Mat std = cv::Mat(t.size(), CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
  t = (t - mean) / std;

  cv::dnn::blobFromImage(t, t);

  // Debugging output
  std::cout << "Blob dimensions: [" << t.size[0] << ", " << t.size[1] << ", "
            << t.size[2] << ", " << t.size[3] << "]" << std::endl;
  return {t, h, w};
}

cv::Mat tensor_to_mat(float* data, int batchIndex, int channels, int height,
                      int width) {
  std::vector<cv::Mat> channelMats;
  int singleImageElements = channels * height * width;

  // Calculate the starting pointer for the desired image
  float* startPtr = data + batchIndex * singleImageElements;

  // Create cv::Mat for each channel
  for (int c = 0; c < channels; ++c) {
    // Pointer to the start of the current channel
    float* channelData = startPtr + c * height * width;
    cv::Mat channelMat(height, width, CV_32FC1, channelData);
    channelMats.push_back(channelMat.clone()); // Clone the data into a new Mat
  }

  // Merge all channel Mats into one Mat
  cv::Mat image;
  cv::merge(channelMats, image);

  // Optional: Convert to more typical 8-bit image
  image.convertTo(image, CV_8UC3, 255.0); // Scale to 0-255 if needed

  return image;
}

cv::Mat tensor_to_mat_dnn(float* data, int height, int width, int channels) {
  // Create a 4D cv::Mat from the float pointer assuming single batch
  cv::Mat tensor(4, new int[4]{1, channels, height, width}, CV_32F, data);

  // Convert 4D Mat to standard 2D Mat (assuming single channel or merging
  // channels)
  std::vector<cv::Mat> channels_vec;
  for (int c = 0; c < channels; ++c) {
    // Create a mat for each channel
    cv::Mat channel(height, width, CV_32F, tensor.ptr(0, c));
    channels_vec.push_back(channel.clone()); // Clone to separate Mat
  }

  cv::Mat output;
  if (channels > 1) {
    cv::merge(channels_vec, output); // Merge if more than one channel
  } else {
    output = channels_vec[0]; // Directly use the single channel
  }

  return output;
}

Ort::Value mat_to_tensor(cv::Mat& img, Ort::MemoryInfo& memory_info) {
  cv::Mat img_continuous = img;
  if (!img.isContinuous()) {
    img_continuous = img.clone();
  }
  std::vector<int64_t> input_node_dims = {
      img_continuous.size[0], img_continuous.size[1], img_continuous.size[2],
      img_continuous.size[3]};
  return Ort::Value::CreateTensor<float>(
      memory_info,
      reinterpret_cast<float*>(img_continuous.data), // direct pointer to data
      img_continuous.total() *
          img_continuous.elemSize1(), // total number of elements, adjusted by
                                      // element size
      input_node_dims.data(), input_node_dims.size());
}

int main() {
  const std::string& image_path =
      "/home/william/Codes/ml-depth-pro/data/example.jpg";
  const std::string& onnx_path =
      "/home/william/Codes/ml-depth-pro/onnx_exp/depth_pro.onnx";
  float f_px = 0.0f;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  cuda_options.arena_extend_strategy = 1;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
  cuda_options.gpu_mem_limit = SIZE_MAX;
  cuda_options.do_copy_in_default_stream = 1;
  session_options.AppendExecutionProvider_CUDA(cuda_options);
  session_options.SetIntraOpNumThreads(
      10); // Set number of threads for CPU parallelism
  session_options.SetLogSeverityLevel(1);
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
  Ort::Session session(env, onnx_path.c_str(), session_options);

  // for debug log

  //   session_options.SetExecutionMode(ORT_SEQUENTIAL);

  //   // Configure CPU memory allocation strategy
  //   OrtArenaCfg* arena_cfg = Ort::ArenaCfg(
  //       4096, // Base size of the arena block (in kilobytes).
  //       20,   // Max percentage of free memory that can be allocated.
  //       1,    // Initial chunk size (in kilobytes) the allocator starts with.
  //       -1 // Maximum size of the arena (in kilobytes). Default value is -1,
  //       which
  //          // means unlimited.
  //   );
  //   session_options.EnableCpuMemArena();
  //   session_options.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_USE_ARENA,
  //   "1");
  //   session_options.AddConfigEntry(ORT_SESSION_OPTIONS_CONFIG_ARENA_CFG,
  //                                  arena_cfg);

  auto [rgb_fp32_t, h, w] = preprocess_image(image_path);
  auto input_node_allocated =
      session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
  std::vector<const char*> input_node_names = {input_node_allocated.get()};

  std::vector<Ort::AllocatedStringPtr> output_node_allocated;
  std::vector<const char*> output_node_names;
  size_t num_outputs = session.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    output_node_allocated.push_back(
        session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()));
    output_node_names.push_back(output_node_allocated.back().get());
  }

  // Debug print of output names

  std::cout << "Expected inputs: ";
  for (const auto& name : input_node_names) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  std::cout << "Expected outputs: ";
  for (const auto& name : output_node_names) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> input_tensors;
  auto rgb_tensor = mat_to_tensor(rgb_fp32_t, memory_info);
  input_tensors.emplace_back(std::move(rgb_tensor));

  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                  input_tensors.data(), session.GetInputCount(),
                  output_node_names.data(), session.GetOutputCount());

  float* canonical_inverse_depth_data =
      output_tensors[0].GetTensorMutableData<float>();
  auto canonoical_inverse_depth = tensor_to_mat_dnn(
      canonical_inverse_depth_data, DEPTH_PRO_FIXED_RESOLUTION,
      DEPTH_PRO_FIXED_RESOLUTION, DEPTH_PRO_FIXED_OUT_CHANNELS);
  float fov_deg = *output_tensors[1].GetTensorMutableData<float>();

  if (f_px == 0.0f) {
    f_px = 0.5 * w / std::tan(0.5 * fov_deg * CV_PI / 180.0);
  }

  std::cout << "Inverse Depth Shape: [" << canonoical_inverse_depth.rows << ", "
            << canonoical_inverse_depth.cols << "]" << std::endl;

  cv::Mat inverse_depth_resized;
  cv::resize(canonoical_inverse_depth, inverse_depth_resized, cv::Size(w, h));

  std::cout << "Inverse Depth Shape: [" << inverse_depth_resized.rows << ", "
            << inverse_depth_resized.cols << "]" << std::endl;
  std::cout << "Estimated focal length is: " << f_px << std::endl;

  return 0;
}