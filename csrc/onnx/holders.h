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
#ifdef _WIN32
#include <Windows.h>
#endif

#ifdef _WIN32

// bad implementation, would cause potential memory leak if without deallocation
inline wchar_t* charToWChar(const char* text) {
  if (text == nullptr)
    return nullptr;

  // Get the length needed for the wide string
  int count = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
  wchar_t* wText = new wchar_t[count];

  // Convert the string from multi-byte to wide character
  MultiByteToWideChar(CP_UTF8, 0, text, -1, wText, count);

  return wText;
}
#endif

struct OrtSetupHolders {
  Ort::Env env;
  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions cuda_options;
  Ort::Session session;
  Ort::MemoryInfo memory_info;

#ifdef _WIN32
  // do not need wchar_t* here
  std::vector<const char*> input_node_names = {};
  std::vector<const char*> output_node_names = {};
#else
  std::vector<const char*> input_node_names = {};
  std::vector<const char*> output_node_names = {};
#endif

#ifdef _WIN32
  OrtSetupHolders()
      : env(ORT_LOGGING_LEVEL_WARNING), // assuming ORT_LOGGING_LEVEL_WARNING is
                                        // a valid argument
        session_options(),              // default constructible assumed
        cuda_options(), // need to ensure it's default constructible
        session(env, std::wstring().c_str(),
                session_options), // This requires valid model path and env
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

#else
  OrtSetupHolders()
      : env(ORT_LOGGING_LEVEL_WARNING), // assuming ORT_LOGGING_LEVEL_WARNING is
                                        // a valid argument
        session_options(),              // default constructible assumed
        cuda_options(), // need to ensure it's default constructible
        session(env, std::string().c_str(),
                session_options), // This requires valid model path and env
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
#endif
};