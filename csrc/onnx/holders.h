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

#include <onnxruntime_cxx_api.h>

struct OrtSetupHolders {
  Ort::Env env;
  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;
  Ort::Session session;
  Ort::MemoryInfo memory_info;

  std::vector<const char*> input_node_names = {};
  std::vector<const char*> output_node_names = {};

  OrtSetupHolders()
      : env(ORT_LOGGING_LEVEL_WARNING), // assuming ORT_LOGGING_LEVEL_WARNING is
                                        // a valid argument
        session_options(),              // default constructible assumed
        cuda_options(), // need to ensure it's default constructible
        session(env, "model_path",
                session_options), // This requires valid model path and env
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};