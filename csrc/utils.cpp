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

#include "utils.h"
#include "macros.h"

// meet the same transform protocol with pytorch code
cv::Mat transform(cv::Mat& img, bool use_dnn_blob = false) {
  cv::Mat img_s;

  if (img.type() != CV_32F) {
    img.convertTo(img_s, CV_32F);
  } else {
    img_s = img;
  }

  img_s /= 255.0f;

  cv::Mat mean = cv::Mat(img.size(), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
  cv::Mat std = cv::Mat(img.size(), CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
  img_s = (img_s - mean) / std;

  // change data layout from HWC to BCHW;
  if (use_dnn_blob) {
    cv::Mat img_transpose;
    cv::dnn::blobFromImage(img_s, img_transpose);
    return img_transpose;
  } else {

    return img_s;
  }
}


std::tuple<cv::Mat, int, int> preprocess_image(const std::string& img_path) {
  cv::Mat img = cv::imread(img_path, cv::IMREAD_ANYCOLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  int h = img.rows, w = img.cols;

  cv::resize(img, img,
             cv::Size(DEPTH_PRO_FIXED_RESOLUTION, DEPTH_PRO_FIXED_RESOLUTION));

  cv::Mat img_fp32(img.rows, img.cols, CV_32F);

  auto BCHW_img_fp32 = transform(img_fp32); //return HxW by default

  return std::make_tuple(BCHW_img_fp32, h, w);
}