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

#include "encoder.h"
#include "configs.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>

class DepthProEncoderImpl : public torch::nn::Module {
public:
  std::vector<int> dims_encoder;
  dinov2l16_384 patch_encoder, image_encoder;
  std::vector<int> hook_block_ids;
  torch::Tensor backbone_highres_hook0;
  torch::Tensor backbone_highres_hook1;
  DepthProEncoderImpl(const std::vector<int>& dims_encoder,
                      dinov2l16_384 patch_encoder, dinov2l16_384 image_encoder,
                      const std::vector<int>& hook_block_ids,
                      int decoder_features)
      : dims_encoder(dims_encoder), patch_encoder(patch_encoder),
        image_encoder(image_encoder), hook_block_ids(hook_block_ids) {
    // Define the upsample and projection layers
    auto upsample_latent0 = register_module(
        "upsample_latent0",
        create_project_upsample_block(patch_encoder->embed_dim, dims_encoder[0],
                                      decoder_features, 3));
    auto upsample_latent1 = register_module(
        "upsample_latent1", create_project_upsample_block(
                                patch_encoder->embed_dim, dims_encoder[0], 2));
    auto upsample0 = register_module(
        "upsample0", create_project_upsample_block(patch_encoder->embed_dim,
                                                   dims_encoder[1], 1));
    auto upsample1 = register_module(
        "upsample1", create_project_upsample_block(patch_encoder->embed_dim,
                                                   dims_encoder[2], 1));
    auto upsample2 = register_module(
        "upsample2", create_project_upsample_block(patch_encoder->embed_dim,
                                                   dims_encoder[3], 1));
    auto upsample_lowres = register_module(
        "upsample_lowres", torch::nn::ConvTranspose2d(
                               torch::nn::ConvTranspose2dOptions(
                                   image_encoder->embed_dim, dims_encoder[3], 2)
                                   .stride(2)
                                   .padding(0)
                                   .bias(true)));

    auto fuse_lowres = register_module(
        "fuse_lowres",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions((dims_encoder[3] + dims_encoder[3]),
                                     dims_encoder[3], 1)
                .stride(1)
                .padding(0)
                .bias(true)));
    // Define the hooks on the specified block indices of the patch_encoder
    // model
    //     patch_encoder->register_forward_hook(
    //         hook_block_ids[0],
    //         [this](const torch::Tensor& input, const torch::Tensor& output) {
    //           this->backbone_highres_hook0 = output;
    //         });
    //     patch_encoder->register_forward_hook(
    //         hook_block_ids[1],
    //         [this](const torch::Tensor& input, const torch::Tensor& output) {
    //           this->backbone_highres_hook1 = output;
    //         });
  }

private:
  torch::nn::Sequential
  create_project_upsample_block(int dim_in, int dim_out, int upsample_layers,
                                std::optional<int> dim_int = std::nullopt) {
    if (!dim_int) {
      dim_int = dim_out;
    }
    std::vector<torch::nn::Module> blocks;
    blocks.emplace_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(dim_in, *dim_int, 1).stride(1).bias(false)));
    for (int i = 0; i < upsample_layers; ++i) {
      blocks.emplace_back(torch::nn::ConvTranspose2d(
          torch::nn::ConvTranspose2dOptions(*dim_int, dim_out, 2)
              .stride(2)
              .bias(false)));
    }
    return torch::nn::Sequential(blocks);
  }

public:
  torch::Tensor forward(torch::Tensor& x) {
    // Your forward implementation based on the Python code
  }
};

TORCH_MODULE(DepthProEncoder);
