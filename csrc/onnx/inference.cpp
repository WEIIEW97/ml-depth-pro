/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inference.h"

#include "../macros.h"
#include <vector>

bool CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

Ort::Value mat_to_tensor(cv::Mat& img, const Ort::MemoryInfo& memory_info) {
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

std::shared_ptr<OrtSetupHolders> warmup(const std::string& onnx_path,
                                        int cpu_num_thread, bool verbose) {
  std::shared_ptr<OrtSetupHolders> holder_ptr =
      std::make_shared<OrtSetupHolders>();

  holder_ptr->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
  holder_ptr->cuda_options.device_id = 0;
  holder_ptr->cuda_options.arena_extend_strategy = 1;
  holder_ptr->cuda_options.cudnn_conv_algo_search =
      OrtCudnnConvAlgoSearchDefault;
  holder_ptr->cuda_options.gpu_mem_limit = SIZE_MAX;
  holder_ptr->cuda_options.do_copy_in_default_stream = 1;

  holder_ptr->session_options.AppendExecutionProvider_CUDA(
      holder_ptr->cuda_options);
  holder_ptr->session_options.SetIntraOpNumThreads(cpu_num_thread);
  if (verbose)
    holder_ptr->session_options.SetLogSeverityLevel(1);
  holder_ptr->session_options.SetGraphOptimizationLevel(
      ORT_ENABLE_BASIC); // something needed for GPU allocation
  holder_ptr->session = Ort::Session(holder_ptr->env, onnx_path.c_str(),
                                     holder_ptr->session_options);

  auto input_node_allocated = holder_ptr->session.GetInputNameAllocated(
      0, Ort::AllocatorWithDefaultOptions());
  holder_ptr->input_node_names = {input_node_allocated.get()};

  std::vector<Ort::AllocatedStringPtr> output_node_allocated;
  size_t num_outputs = holder_ptr->session.GetOutputCount();
  for (size_t i = 0; i < num_outputs; ++i) {
    output_node_allocated.push_back(holder_ptr->session.GetOutputNameAllocated(
        i, Ort::AllocatorWithDefaultOptions()));
    holder_ptr->output_node_names.push_back(output_node_allocated.back().get());
  }

  if (verbose) {
    std::cout << "Expected inputs: ";
    for (const auto& name : holder_ptr->input_node_names) {
      std::cout << name << " ";
    }
    std::cout << std::endl;

    std::cout << "Expected outputs: ";
    for (const auto& name : holder_ptr->output_node_names) {
      std::cout << name << " ";
    }
    std::cout << std::endl;
  }

  holder_ptr->memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  return holder_ptr;
}

void infer(std::shared_ptr<OrtSetupHolders>& holders,
           const std::string& img_path, cv::Mat& inverse_depth_full,
           float& f_px) {
  auto [rgb_fp32_t, h, w] = preprocess_image(img_path);
  std::vector<Ort::Value> input_tensors;
  auto rgb_tensor = mat_to_tensor(rgb_fp32_t, holders->memory_info);
  input_tensors.emplace_back(std::move(rgb_tensor));

  auto output_tensors = holders->session.Run(
      Ort::RunOptions{nullptr}, holders->input_node_names.data(),
      input_tensors.data(), holders->session.GetInputCount(),
      holders->output_node_names.data(), holders->session.GetOutputCount());

  float* canonical_inverse_depth_data =
      output_tensors[0].GetTensorMutableData<float>();

  auto canonoical_inverse_depth = tensor_to_mat_dnn(
      canonical_inverse_depth_data, DEPTH_PRO_FIXED_RESOLUTION,
      DEPTH_PRO_FIXED_RESOLUTION, DEPTH_PRO_FIXED_OUT_CHANNELS);
  float fov_deg = *output_tensors[1].GetTensorMutableData<float>();

  if (f_px == 0.0f) {
    f_px = 0.5 * w / std::tan(0.5 * fov_deg * CV_PI / 180.0);
  }

  cv::resize(canonoical_inverse_depth, inverse_depth_full, cv::Size(w, h));
}