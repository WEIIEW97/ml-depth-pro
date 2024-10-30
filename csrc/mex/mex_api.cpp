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

#include "mex_api.h"

mxArray* convert_mat_to_mx_array(const cv::Mat& mat) {
  if (mat.empty() || mat.channels() != 1) {
    mexErrMsgIdAndTxt("MATLAB:convert_mat_to_mx_array:invalidInput",
                      "Input must be a non-empty numeric matrix.");
  }

  auto cv_type = mat.type();
  if (cv_type != CV_32F || cv_type != CV_64F) {
    mexErrMsgIdAndTxt("MATLAB:convert_mat_to_mx_array:invalidInput",
                      "Input type must be float or double.");
  }

  auto class_id = (cv_type == CV_32F) ? mxSINGLE_CLASS : mxDOUBLE_CLASS;

  mxArray* out = mxCreateNumericMatrix(mat.rows, mat.cols, class_id, mxREAL);

  // Copy data from cv::Mat to mxArray, considering the difference in data
  // storage order
  if (cv_type == CV_32F) {
    float* out_data = static_cast<float*>(mxGetData(out));

    for (int i = 0; i < mat.cols; i++) {
      for (int j = 0; j < mat.rows; j++) {
        out_data[i * mat.rows + j] = mat.at<float>(j, i);
      }
    }
  } else {
    double* out_data = static_cast<double*>(mxGetData(out));
    for (int i = 0; i < mat.cols; i++) {
      for (int j = 0; j < mat.rows; j++) {
        out_data[i * mat.rows + j] = mat.at<double>(j, i);
      }
    }
  }
  return out;
}

cv::Mat mx_array_to_mat(const mxArray* array) {
  if (!mxIsDouble(array) && !mxIsSingle(array)) {
    mexErrMsgIdAndTxt("MATLAB:mx_array_to_mat:inputNotSupported",
                      "Input type must be double or single.");
  }

  // Determine dimensions and data type
  const mwSize* dims = mxGetDimensions(array);
  int rows = static_cast<int>(dims[0]);
  int cols = static_cast<int>(dims[1]);

  auto type = (mxIsDouble(array)) ? CV_64F : CV_32F;

  cv::Mat mat(cols, rows, type); // transpose to match row-major order

  // copy data from mxArray to cv::Mat
  if (mxIsDouble(array)) {
    double* data = mxGetPr(array);
    for (int i = 0; i < cols; i++) {
      for (int j = 0; j < rows; j++) {
        mat.at<double>(i, j) = data[j + i * rows]; // Transpose data
      }
    }
  } else if (mxIsSingle(array)) {
    float* data = reinterpret_cast<float*>(mxGetData(array));
    for (int i = 0; i < cols; i++) {
      for (int j = 0; j < rows; j++) {
        mat.at<float>(i, j) = data[j + i * rows]; // Transpose data
      }
    }
  }

  return mat;
}