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

#include <mex.h>

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  // Check for proper number of arguments
  if (nrhs != 2) {
    mexErrMsgIdAndTxt("MATLAB:add_two_numbers:nrhs", "Two inputs required.");
  }
  if (nlhs != 1) {
    mexErrMsgIdAndTxt("MATLAB:add_two_numbers:nlhs", "One output required.");
  }

  // Make sure both input arguments are type double
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1])) {
    mexErrMsgIdAndTxt("MATLAB:add_two_numbers:typeError",
                      "Inputs must be double.");
  }

  // Pointers to the input matrices
  double* a = mxGetPr(prhs[0]);
  double* b = mxGetPr(prhs[1]);

  // Allocate matrix for the return argument
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double* out = mxGetPr(plhs[0]);

  // Perform the addition operation
  *out = *a + *b;
}