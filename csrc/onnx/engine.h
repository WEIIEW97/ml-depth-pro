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

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <cassert>
#include <vector>

class OnnxRuntimeEngine {
public:
#ifdef _WIN32
  OnnxRuntimeEngine(const wchar_t* model_path);
#else
  OnnxRuntimeEngine(const char* model_path);
#endif

  ~OnnxRuntimeEngine();

  OrtApi* getOrtApi() { return g_ort_rt; }
  OrtSession* getOrtSession() { return session; }
  OrtAllocator* getOrtAllocator() { return allocator; }
  OrtMemoryInfo* getOrtMemoryInfo() { return memory_info; }

  Ort::Value mat_to_tensor(cv::Mat& img);

private:
  OrtApi* g_ort_rt = nullptr;
  OrtSession* session = nullptr;
  OrtAllocator* allocator = nullptr;
  OrtMemoryInfo* memory_info = nullptr;

  size_t num_input_nodes;
  std::vector<const char*> input_node_names;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<OrtValue*> input_tensors;
  size_t num_output_nodes;
  std::vector<const char*> output_node_names;
  std::vector<std::vector<int64_t>> output_node_dims;
  std::vector<OrtValue*> output_tensors;

  bool CheckStatus(const OrtApi* g_ort, OrtStatus* status);
};
