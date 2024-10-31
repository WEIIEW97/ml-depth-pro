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

#include "../onnx/inference.h"

// functions params:
// 0: onnx path
// 1: image path
// 2: number of used cpu thread
// 3: manully input focal pixel length(default should be 0)
// 4: if show processing output
// 5: if use cuda

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs != 5) {
    mexErrMsgIdAndTxt("MATLAB:onnxInfer:invalidNumInputs",
                      "five inputs required.");
  }

  if (nlhs != 2) {
    mexErrMsgIdAndTxt("MATLAB:onnxInfer:invalidNumOutputs",
                      "Two outputs required.");
  }

  const char* onnx_path = mxArrayToString(prhs[0]);
  const char* img_path = mxArrayToString(prhs[1]);
  int cpu_num_thread = static_cast<int>(mxGetScalar(prhs[2]));
  float f_px = static_cast<float>(mxGetScalar(prhs[3]));
  bool verbose = mxGetLogicals(prhs[4])[0];
  bool use_cuda = mxGetLogicals(prhs[5])[0];

  auto holders = warmup(onnx_path, cpu_num_thread, verbose, use_cuda);

  cv::Mat inverse_depth_full;
  infer(holders, std::string(img_path), inverse_depth_full, f_px);

  mxFree((void*)onnx_path);
  mxFree((void*)img_path);

  plhs[0] = convert_mat_to_mx_array(inverse_depth_full);

  int cv_mat_type = inverse_depth_full.type();
  auto mx_class_type =
      (cv_mat_type == CV_32F) ? mxSINGLE_CLASS : mxDOUBLE_CLASS;

  plhs[1] = mxCreateNumericMatrix(1, 1, mx_class_type, mxREAL);

  if (mx_class_type == mxSINGLE_CLASS) {
    *static_cast<float*>(mxGetData(plhs[1])) = f_px;
  } else {
    *static_cast<double*>(mxGetData(plhs[1])) = f_px;
  }
}