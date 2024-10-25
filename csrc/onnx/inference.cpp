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

bool CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

Ort::Value mat_to_tensor(cv::Mat& img, OrtMemoryInfo* memory_info) {
  std::vector<int64_t> input_node_dims = {1, 3, img.rows, img.cols};
  size_t num_elements = img.total() * img.channels();
  std::vector<float> array;
  array.assign(img.begin<float>(), img.end<float>());
  return Ort::Value::CreateTensor<float>(
      memory_info, array.data(), num_elements, input_node_dims.data(), 4);
}

const OrtApi* warmup(const std::string& onnx_path, bool is_fp32 = true) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  OrtSessionOptions* session_options;
  CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
  CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
  CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options,
                                                             ORT_ENABLE_BASIC));

  std::vector<const char*> options_keys = {};
  std::vector<const char*> options_values = {};

  // Need to set the HTP FP16 precision for float32 model, otherwise it's FP32
  // precision and runs very slow No need to set it for float16 model
  const std::string ENABLE_HTP_FP16_PRECISION = "enable_htp_fp16_precision";
  const std::string ENABLE_HTP_FP16_PRECISION_VALUE = "1";
  if (is_fp32) {
    options_keys.push_back(ENABLE_HTP_FP16_PRECISION.c_str());
    options_values.push_back(ENABLE_HTP_FP16_PRECISION_VALUE.c_str());
  }

  CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider(
                         session_options, "depth-pro", options_keys.data(),
                         options_values.data(), options_keys.size()));

  OrtSession* session;
  CheckStatus(g_ort, g_ort->CreateSession(env, onnx_path.c_str(),
                                          session_options, &session));

  OrtAllocator* allocator;
  CheckStatus(g_ort, g_ort->GetAllocatorWithDefaultOptions(&allocator));
  size_t num_input_nodes;
  CheckStatus(g_ort, g_ort->SessionGetInputCount(session, &num_input_nodes));

  std::vector<const char*> input_node_names;
  std::vector<std::vector<int64_t>> input_node_dims;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<OrtValue*> input_tensors;

  input_node_names.resize(num_input_nodes);
  input_node_dims.resize(num_input_nodes);
  input_types.resize(num_input_nodes);
  input_tensors.resize(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    // Get input node names
    char* input_name;
    CheckStatus(g_ort,
                g_ort->SessionGetInputName(session, i, allocator, &input_name));
    input_node_names[i] = input_name;

    // Get input node types
    OrtTypeInfo* type_info;
    CheckStatus(g_ort, g_ort->SessionGetInputTypeInfo(session, i, &type_info));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort,
                g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    ONNXTensorElementDataType type;
    CheckStatus(g_ort, g_ort->GetTensorElementType(tensor_info, &type));
    input_types[i] = type;

    // Get input shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    input_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, input_node_dims[i].data(), num_dims));

    if (type_info)
      g_ort->ReleaseTypeInfo(type_info);
  }

  size_t num_output_nodes;
  std::vector<const char*> output_node_names;
  std::vector<std::vector<int64_t>> output_node_dims;
  std::vector<OrtValue*> output_tensors;
  CheckStatus(g_ort, g_ort->SessionGetOutputCount(session, &num_output_nodes));
  output_node_names.resize(num_output_nodes);
  output_node_dims.resize(num_output_nodes);
  output_tensors.resize(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    // Get output node names
    char* output_name;
    CheckStatus(g_ort, g_ort->SessionGetOutputName(session, i, allocator,
                                                   &output_name));
    output_node_names[i] = output_name;

    OrtTypeInfo* type_info;
    CheckStatus(g_ort, g_ort->SessionGetOutputTypeInfo(session, i, &type_info));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort,
                g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));

    // Get output shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    output_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, (int64_t*)output_node_dims[i].data(),
                           num_dims));

    if (type_info)
      g_ort->ReleaseTypeInfo(type_info);
  }

  return g_ort;
}

void infer(const std::string& img_path, const OrtApi* g_ort, cv::Mat& out_depth,
           float pix_f) {

  OrtMemoryInfo* memory_info;
  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  auto [rgb, h, w] = preprocess_image(img_path);
  auto rgb_tensor = mat_to_tensor(rgb, memory_info);

  g_ort->ReleaseMemoryInfo(memory_info);

  
}