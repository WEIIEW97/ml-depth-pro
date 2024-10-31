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

#include <memory>

#include "../utils.h"
#include "holders.h"

// for c api compatible

#ifdef _WIN32
OrtSetupHolders* c_warmup(const wchar_t* onnx_path, int cpu_num_thread,
                          bool verbose, bool use_cuda);
std::shared_ptr<OrtSetupHolders> warmup(const wchar_t* onnx_path,
                                        int cpu_num_thread, bool verbose,
                                        bool use_cuda);
void infer(std::shared_ptr<OrtSetupHolders>& holders, const wchar_t* img_path,
           cv::Mat& inverse_depth_full, float& f_px);
void c_infer(OrtSetupHolders* holders, const wchar_t* img_path,
             cv::Mat& inverse_depth_full, float& f_px);
#else
OrtSetupHolders* c_warmup(const char* onnx_path, int cpu_num_thread,
                          bool verbose, bool use_cuda);
std::shared_ptr<OrtSetupHolders>
warmup(const char* onnx_path, int cpu_num_thread, bool verbose, bool use_cuda);
void infer(std::shared_ptr<OrtSetupHolders>& holders, const char* img_path,
           cv::Mat& inverse_depth_full, float& f_px);
void c_infer(OrtSetupHolders* holders, const char* img_path,
             cv::Mat& inverse_depth_full, float& f_px);
#endif

void delete_ptr(OrtSetupHolders* ptr);
/////////////////////////// c++ calls ///////////////////////////

// for starting up the engine, do some warmup, time consuming

#ifdef _WIN32
std::shared_ptr<OrtSetupHolders> warmup(const std::wstring& onnx_path,
                                        int cpu_num_thread, bool verbose,
                                        bool use_cuda);

// inference a single image by warmed-up onnx runtime engines
void infer(std::shared_ptr<OrtSetupHolders>& holders,
           const std::wstring& img_path, cv::Mat& inverse_depth_full,
           float& f_px);
#else
std::shared_ptr<OrtSetupHolders> warmup(const std::string& onnx_path,
                                        int cpu_num_thread, bool verbose,
                                        bool use_cuda);

// inference a single image by warmed-up onnx runtime engines
void infer(std::shared_ptr<OrtSetupHolders>& holders,
           const std::string& img_path, cv::Mat& inverse_depth_full,
           float& f_px);
#endif