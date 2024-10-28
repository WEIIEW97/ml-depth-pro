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

  auto BCHW_img_fp32 = transform(img_fp32); // return HxW by default

  return std::make_tuple(BCHW_img_fp32, h, w);
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