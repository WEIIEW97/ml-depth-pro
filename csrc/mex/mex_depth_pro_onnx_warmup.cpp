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


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("MATLAB:warmup:invalidNumInputs", "Three inputs required.");
    }

    const char* onnx_path = mxArrayToString(prhs[0]);
    int cpu_num_thread = static_cast<int>(mxGetScalar(prhs[1]));
    bool verbose = mxGetLogicals(prhs[2])[0];

    auto holders = warmup(onnx_path, cpu_num_thread, verbose);

    
}