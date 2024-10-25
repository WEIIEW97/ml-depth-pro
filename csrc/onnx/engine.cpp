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

#include "engine.h"

#ifdef _WIN32
OnnxRuntimeEngine::OnnxRuntimeEngine(const wchar_t* model_path)
#else
OnnxRuntimeEngine::OnnxRuntimeEngine(const char* model_path)
#endif
{
  g_ort_rt = nullptr;
  session = nullptr;
  allocator = nullptr;

  const OrtApi* g_ort = nullptr;
  const OrtApiBase* ptr_api_base = OrtGetApiBase();
  g_ort = ptr_api_base->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_options;
  CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
  CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
  CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options,
                                                             ORT_ENABLE_BASIC));
  Ort::MemoryInfo memory_info_temp =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#ifndef _WIN32
  // CUDA option set
  OrtCUDAProviderOptions cuda_option;
  cuda_option.device_id = 0;
  cuda_option.arena_extend_strategy = 0;
  cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_option.gpu_mem_limit = SIZE_MAX;
  cuda_option.do_copy_in_default_stream = 1;
  // CUDA acceleration
  CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider_CUDA(
                         session_options, &cuda_option));
#endif

  // load  model and creat session
  // Model file path
  CheckStatus(g_ort,
              g_ort->CreateSession(env, model_path, session_options, &session));
  CheckStatus(g_ort, g_ort->GetAllocatorWithDefaultOptions(&allocator));
  //**********Input information**********//
  // size_t num_input_nodes; //Enter the number of nodes
  CheckStatus(g_ort, g_ort->SessionGetInputCount(session, &num_input_nodes));
  // initialize memory info
  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

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
    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CheckStatus(g_ort, g_ort->GetTensorElementType(tensor_info, &type));
    input_types[i] = type;

    // Get input shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    input_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, input_node_dims[i].data(), num_dims));

    size_t tensor_size;
    CheckStatus(g_ort,
                g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

    if (typeinfo)
      g_ort->ReleaseTypeInfo(typeinfo);
  }
  //---------------------------------------------------//

  //***********output info****************//

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

    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    // Get output shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    output_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, (int64_t*)output_node_dims[i].data(),
                           num_dims));

    if (typeinfo)
      g_ort->ReleaseTypeInfo(typeinfo);
  }

  g_ort_rt = (OrtApi*)g_ort;
}

OnnxRuntimeEngine::~OnnxRuntimeEngine() {
  if (memory_info) {
    g_ort_rt->ReleaseMemoryInfo(memory_info);
  }

  if (allocator) {
    g_ort_rt->ReleaseAllocator(allocator);
  }

  if (session) {
    g_ort_rt->ReleaseSession(session);
  }
}

bool OnnxRuntimeEngine::CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

Ort::Value OnnxRuntimeEngine::mat_to_tensor(cv::Mat& img) {
  std::vector<int64_t> input_node_dims = {1, 3, img.rows, img.cols};
  size_t num_elements = img.total() * img.channels();
  std::vector<float> array;
  array.assign(img.begin<float>(), img.end<float>());
  return Ort::Value::CreateTensor<float>(
      memory_info, array.data(), num_elements, input_node_dims.data(), 4);
}
