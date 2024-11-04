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

#include "macros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <tuple>
#include <string>

std::tuple<cv::Mat, int, int> preprocess_image(const std::string& img_path);
cv::Mat tensor_to_mat(float* data, int batchIndex, int channels, int height,
                      int width);
cv::Mat tensor_to_mat_dnn(float* data, int height, int width, int channels);
std::string wideToString(const std::wstring& wstr);
int get_maximum_threads_of_cpu();
int safe_get_maximum_threads_of_cpu();