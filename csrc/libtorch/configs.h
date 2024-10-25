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

#include <vector>
#include <string>
#include <torch/torch.h>

/// placeholder, just align to python codes. No actual effects
struct ViTConfig {
  int in_chans, embed_dim;
  int img_size = 384;
  int patch_size = 16;

  int timm_img_size = 384;
  int timm_patch_size = 16;

  std::vector<int> encoder_feature_layer_ids, encoder_feature_dims;
};

struct dinov2l16_384Impl : public torch::nn::Module {
  int in_chans = 3;
  int embed_dim = 1024;
  std::vector<int> encoder_feature_layer_ids = {5, 11, 17, 23};
  std::vector<int> encoder_feature_dims = {256, 512, 1024, 1024};
  int img_size = 384;
  int patch_size = 16;
  std::string timm_preset = "vit_large_patch14_dinov2";
  int timm_img_size = 518;
  int timm_patch_size = 14;
};

TORCH_MODULE(dinov2l16_384);

struct DepthProConfig {
  dinov2l16_384 patch_encoder_preset;
  dinov2l16_384 image_encoder_preset;
  dinov2l16_384 fov_encoder_preset;
  std::string checkpoint_uri;
  int decoder_features = 256;
  bool use_fov_head = true;
};
