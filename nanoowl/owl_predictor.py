# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
import PIL.Image
import subprocess
import tempfile
import os
from torchvision.ops import roi_align
from transformers.models.owlvit.modeling_owlvit import OwlViTForObjectDetection
from transformers.models.owlvit.processing_owlvit import OwlViTProcessor
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple
from .image_preprocessor import ImagePreprocessor

__all__ = [
    "OwlPredictor",
    "OwlEncodeTextOutput",
    "OwlEncodeImageOutput",
    "OwlDecodeOutput"
]


def _owl_center_to_corners_format_torch(bboxes_center):
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        [
            (center_x - 0.5 * width), 
            (center_y - 0.5 * height), 
            (center_x + 0.5 * width), 
            (center_y + 0.5 * height)
        ],
        dim=-1,
    )
    return bbox_corners


def _owl_get_image_size(hf_name: str):

    image_sizes = {
        "google/owlvit-base-patch32": 768,
        "google/owlvit-base-patch16": 768,
        "google/owlvit-large-patch14": 840,
    }

    return image_sizes[hf_name]


def _owl_get_patch_size(hf_name: str):

    patch_sizes = {
        "google/owlvit-base-patch32": 32,
        "google/owlvit-base-patch16": 16,
        "google/owlvit-large-patch14": 14,
    }

    return patch_sizes[hf_name]


# This function is modified from https://github.com/huggingface/transformers/blob/e8fdd7875def7be59e2c9b823705fbf003163ea0/src/transformers/models/owlvit/modeling_owlvit.py#L1333
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def _owl_normalize_grid_corner_coordinates(num_patches_per_side):
    box_coordinates = np.stack(
        np.meshgrid(np.arange(1, num_patches_per_side + 1), np.arange(1, num_patches_per_side + 1)), axis=-1
    ).astype(np.float32)
    box_coordinates /= np.array([num_patches_per_side, num_patches_per_side], np.float32)

    box_coordinates = box_coordinates.reshape(
        box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
    )
    box_coordinates = torch.from_numpy(box_coordinates)

    return box_coordinates


# This function is modified from https://github.com/huggingface/transformers/blob/e8fdd7875def7be59e2c9b823705fbf003163ea0/src/transformers/models/owlvit/modeling_owlvit.py#L1354
# Copyright 2022 Google AI and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
def _owl_compute_box_bias(num_patches_per_side):
    box_coordinates = _owl_normalize_grid_corner_coordinates(num_patches_per_side)
    box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

    box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

    box_size = torch.full_like(box_coord_bias, 1.0 / num_patches_per_side)
    box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

    box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)

    return box_bias


def _owl_box_roi_to_box_global(boxes, rois):
    # print("boxes shape", boxes)
    # print("rois", rois)
    x0y0 = rois[..., :2]
    x1y1 = rois[..., 2:]
    wh = (x1y1 - x0y0).repeat(1, 1, 2)
    x0y0 = x0y0.repeat(1, 1, 2)
    return (boxes * wh) + x0y0


@dataclass
class OwlEncodeTextOutput:
    text_embeds: torch.Tensor

    def slice(self, start_index, end_index):
        return OwlEncodeTextOutput(
            text_embeds=self.text_embeds[start_index:end_index]
        )


@dataclass
class OwlEncodeImageOutput:
    image_embeds: torch.Tensor
    image_class_embeds: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    pred_boxes: torch.Tensor


@dataclass
class OwlDecodeOutput:
    labels: List[torch.Tensor]
    scores: List[torch.Tensor]
    boxes: List[torch.Tensor]
    input_indices: List[torch.Tensor]

class OwlPredictor(torch.nn.Module):
    
    def __init__(self,
            model_name: str = "google/owlvit-base-patch32",
            device: str = "cuda",
            image_encoder_engine: Optional[str] = None,
            image_encoder_engine_max_batch_size: int = 1,
            image_preprocessor: Optional[ImagePreprocessor] = None
        ):

        super().__init__()

        self.image_size = _owl_get_image_size(model_name)
        self.device = device
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device).eval()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.patch_size = _owl_get_patch_size(model_name)
        self.num_patches_per_side = self.image_size // self.patch_size
        self.box_bias = _owl_compute_box_bias(self.num_patches_per_side).to(self.device)
        self.num_patches = (self.num_patches_per_side)**2
        # print(self.patch_size)
        # print(self.device)
        # print(image_encoder_engine)
        # print(model_name)
        # print(image_encoder_engine_max_batch_size)
        self.mesh_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0., 1., self.image_size),
                torch.linspace(0., 1., self.image_size)
            )
        ).to(self.device).float()
        self.image_encoder_engine = None
        if image_encoder_engine is not None:
            image_encoder_engine = OwlPredictor.load_image_encoder_engine(image_encoder_engine, image_encoder_engine_max_batch_size)
        self.image_encoder_engine = image_encoder_engine
        self.image_preprocessor = image_preprocessor.to(self.device).eval() if image_preprocessor else ImagePreprocessor().to(self.device).eval()
       

    def get_num_patches(self):
        return self.num_patches

    def get_device(self):
        return self.device
        
    def get_image_size(self):
        return (self.image_size, self.image_size)
    
    def encode_text(self, text: List[str]) -> OwlEncodeTextOutput:
        text_input = self.processor(text=text, return_tensors="pt")
        input_ids = text_input['input_ids'].to(self.device)
        attention_mask = text_input['attention_mask'].to(self.device)
        text_outputs = self.model.owlvit.text_model(input_ids, attention_mask)
        text_embeds = text_outputs[1]
        text_embeds = self.model.owlvit.text_projection(text_embeds)
        return OwlEncodeTextOutput(text_embeds=text_embeds)

    def encode_image_torch(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        
        vision_outputs = self.model.owlvit.vision_model(image)
        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)
        class_token_out = image_embeds[:, :1, :]
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)  # 768 dim

        # Box Head
        pred_boxes = self.model.box_head(image_embeds)
        pred_boxes += self.box_bias
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_boxes = _owl_center_to_corners_format_torch(pred_boxes)

        # Class Head
        image_class_embeds = self.model.class_head.dense0(image_embeds)
        logit_shift = self.model.class_head.logit_shift(image_embeds)
        logit_scale = self.model.class_head.logit_scale(image_embeds)
        logit_scale = self.model.class_head.elu(logit_scale) + 1

        output = OwlEncodeImageOutput(
            image_embeds=image_embeds,
            image_class_embeds=image_class_embeds,
            logit_shift=logit_shift,
            logit_scale=logit_scale,
            pred_boxes=pred_boxes
        )

        return output
    
    def encode_image_trt(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        return self.image_encoder_engine(image)

    def encode_image(self, image: torch.Tensor) -> OwlEncodeImageOutput:
        if self.image_encoder_engine is not None:
            return self.encode_image_trt(image)
        else:
            return self.encode_image_torch(image)

    def extract_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float = 1.0):
        if len(rois) == 0:
            return torch.empty(
                (0, image.shape[1], self.image_size, self.image_size),
                dtype=image.dtype,
                device=image.device
            )
        # print("Check Extract Roi")
        # print(rois.shape)   
        # print(image.shape)
        # if pad_square:
        #     # pad square
        #     w = padding_scale * (rois[..., 2] - rois[..., 0]) / 2
        #     h = padding_scale * (rois[..., 3] - rois[..., 1]) / 2
        #     cx = (rois[..., 0] + rois[..., 2]) / 2
        #     cy = (rois[..., 1] + rois[..., 3]) / 2
        #     s = torch.max(w, h)
        #     rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)

        #     # compute mask
        #     pad_x = (s - w) / (2 * s)
        #     pad_y = (s - h) / (2 * s)
        #     mask_x = (self.mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self.mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
        #     mask_y = (self.mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self.mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
        #     mask = (mask_x & mask_y)

        # roi_images_list = []
        # print(image.shape)
        # print(rois.shape)
        # for images, roi in zip(image, rois):
        #     print("Inside")
        #     if pad_square:
        #         # pad square
        #         w = padding_scale * (roi[..., 2] - roi[..., 0]) / 2
        #         h = padding_scale * (roi[..., 3] - roi[..., 1]) / 2
        #         cx = (roi[..., 0] + roi[..., 2]) / 2
        #         cy = (roi[..., 1] + roi[..., 3]) / 2
        #         s = torch.max(w, h)
        #         roi = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)
        #         print(f"roi : {roi.shape}")
        #         # compute mask
        #         pad_x = (s - w) / (2 * s)
        #         pad_y = (s - h) / (2 * s)
        #         mask_x = (self.mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self.mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
        #         mask_y = (self.mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self.mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
        #         mask = (mask_x & mask_y)
        #         print(f"Mask : {mask.shape}")

        #     # print([roi])
        #     # print(roi.shape)
        #     # extract rois
        #     print(images.unsqueeze(0).shape)
        #     print(roi.unsqueeze(0).shape)
        #     roi_images = roi_align(images.unsqueeze(0), [roi.unsqueeze(0)], output_size=self.get_image_size())
        #     if pad_square:
        #         print(mask[:, None, :, :].shape)
        #         roi_images = (roi_images * mask[:, None, :, :])
        #         print(f"Roi : {roi_images.shape}")
        #     roi_images_list.append(roi_images)

        #     print(f" List  {torch.cat(roi_images_list, dim=0).shape} {rois.shape}")
        # return torch.cat(roi_images_list, dim=0), rois

        # print(f"rois {rois.shape}")
        # # print(f"roi images{roi_images_list}")
        # print("Extract roi last")
        # print(f"roi images{roi_images.shape}")
        # print(roi_images.shape)
        # print(image.shape)
        
        # # print([rois])
        # # print(rois)
        if pad_square:
            # pad square
            w = padding_scale * (rois[..., 2] - rois[..., 0]) / 2
            h = padding_scale * (rois[..., 3] - rois[..., 1]) / 2
            cx = (rois[..., 0] + rois[..., 2]) / 2
            cy = (rois[..., 1] + rois[..., 3]) / 2
            s = torch.max(w, h)
            rois = torch.stack([cx-s, cy-s, cx+s, cy+s], dim=-1)
            # print(f"Rois 1 { rois}")
            
            # Create batch indices tensor
            batch_indices = torch.arange(rois.size(0)).unsqueeze(1).to(rois.device)  # Shape: (batch_size, 1)

            # Concatenate batch indices with ROIs along the last dimension
            rois_with_index = torch.cat([batch_indices, rois], dim=-1)  # Shape: (batch_size, 5)
            # print(f"Rois { rois_with_index}")
            
            # compute mask
            pad_x = (s - w) / (2 * s)
            pad_y = (s - h) / (2 * s)
            mask_x = (self.mesh_grid[1][None, ...] > pad_x[..., None, None]) & (self.mesh_grid[1][None, ...] < (1. - pad_x[..., None, None]))
            mask_y = (self.mesh_grid[0][None, ...] > pad_y[..., None, None]) & (self.mesh_grid[0][None, ...] < (1. - pad_y[..., None, None]))
            mask = (mask_x & mask_y)
            # print(f"Mask : {mask.shape}")


        roi_images = roi_align(image, rois_with_index, output_size=self.get_image_size())
        # print(f"img {roi_images.shape}")

        if pad_square:
            roi_images = (roi_images * mask[:, None, :, :])

        return roi_images, rois
        # # Add batch indices to rois
        # batch_size = image.shape[0]
        # batch_indices = torch.arange(batch_size, dtype=rois.dtype, device=rois.device).view(-1, 1)
        # rois = torch.cat([batch_indices, rois], dim=1)
        # # print(rois.shape)
        # # extract rois
        # roi_images = roi_align(image, [rois], output_size=self.get_image_size())
        # # print("Extract roi")
        # # print(f"roi images{roi_images.shape}")
        # # print(roi_images.shape)
        # # print(image.shape)

        # # mask rois
        # if pad_square:
        #     roi_images = (roi_images * mask[:, None, :, :])
        # # print("Mask roi")
        # # print(roi_images.shape)
        # # print(image.shape)
        # # print(rois.shape)
        # # print(f"rois {rois}")
        # # print(f"roi images{roi_images}")
        # return roi_images, rois
    
    def encode_rois(self, image: torch.Tensor, rois: torch.Tensor, pad_square: bool = True, padding_scale: float=1.0):
        # with torch_timeit_sync("extract rois"):
        # print(f"Roi Image : {image.shape}")
        # print(f"Rois Shape : {rois.shape}")
        roi_images, rois = self.extract_rois(image, rois, pad_square, padding_scale)
        # with torch_timeit_sync("encode images"):
        # print("Check roi")
        # print(roi_images.shape)
        output = self.encode_image(roi_images)
        # # print(f"output engine : {output}")
        # print("Output pred boxes shape", output.pred_boxes.shape)
        pred_boxes = _owl_box_roi_to_box_global(output.pred_boxes, rois[:, None, :])
        output.pred_boxes = pred_boxes
        return output

    # def decode(self, 
    #         image_output: OwlEncodeImageOutput, 
    #         text_output: OwlEncodeTextOutput,
    #         threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    #     ) -> OwlDecodeOutput:

    #     if isinstance(threshold, (int, float)):
    #         threshold = [threshold] * len(text_output.text_embeds) #apply single threshold to all labels 

    #     # print(f"Num Input :{image_output.image_class_embeds.shape}")
    #     # print(f"Num Input :{image_output.image_embeds.shape}")
    #     # print(f"Num Input :{image_output.pred_boxes.shape}")
    #     # print(f"Num Input :{image_output.logit_scale.shape}")
    #     # print(f"Num Input :{image_output.logit_shift.shape}")

    #     num_input_images = image_output.image_class_embeds.shape[1]
        
    #     image_class_embeds = image_output.image_class_embeds
    #     image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
    #     query_embeds = text_output.text_embeds
    #     query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
    #     logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
    #     logits = (logits + image_output.logit_shift) * image_output.logit_scale
        
    #     scores_sigmoid = torch.sigmoid(logits)
    #     scores_max = scores_sigmoid.max(dim=-1)
    #     labels = scores_max.indices
    #     scores = scores_max.values
    #     masks = []
    #     for i, thresh in enumerate(threshold):
    #         label_mask = labels == i
    #         score_mask = scores > thresh
    #         obj_mask = torch.logical_and(label_mask,score_mask)
    #         masks.append(obj_mask) 
        
    #     mask = masks[0]
    #     for mask_t in masks[1:]:
    #         mask = torch.logical_or(mask, mask_t)

    #     input_indices = torch.arange(0, num_input_images, dtype=labels.dtype, device=labels.device)
    #     input_indices = input_indices[:, None].repeat(1, self.num_patches)

    #     return OwlDecodeOutput(
    #         labels=labels[mask],
    #         scores=scores[mask],
    #         boxes=image_output.pred_boxes[mask],
    #         input_indices=input_indices[mask]
    #     )

    # def decode(self, 
    #            image_output: OwlEncodeImageOutput, 
    #            text_output: OwlEncodeTextOutput,
    #            threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    #           ) -> OwlDecodeOutput:

    #     if isinstance(threshold, (int, float)):
    #         threshold = [threshold] * text_output.text_embeds.shape[1]  # apply single threshold to all labels 

    #     # print(f"Num Input :{image_output.image_class_embeds.shape}")
    #     # print(f"Num Input :{image_output.image_embeds.shape}")
    #     # print(f"Num Input :{image_output.pred_boxes.shape}")
    #     # print(f"Num Input :{image_output.logit_scale.shape}")
    #     # print(f"Num Input :{image_output.logit_shift.shape}")

    #     all_labels = []
    #     all_scores = []
    #     all_boxes = []
    #     all_input_indices = []

    #     for i in range(8):  # Assuming there are 8 images in the batch
    #         num_input_images = image_output.image_class_embeds[i].shape[0]
    #         image_class_embeds = image_output.image_class_embeds[i]
    #         # print(f"Image : {num_input_images}")
    #         # print(f"Image : {image_class_embeds.shape}")
    #         image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=(-1,), keepdim=True) + 1e-6)
    #         query_embeds = text_output.text_embeds
    #         # print(f"Query : {query_embeds.shape}")
    #         query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
    #         logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
    #         logits = (logits + image_output.logit_shift[i]) * image_output.logit_scale[i]

    #         scores_sigmoid = torch.sigmoid(logits)
    #         scores_max = scores_sigmoid.max(dim=-1)
    #         labels = scores_max.indices
    #         scores = scores_max.values
    #         masks = []
    #         for j, thresh in enumerate(threshold):
    #             label_mask = labels == j
    #             score_mask = scores > thresh
    #             obj_mask = torch.logical_and(label_mask, score_mask)
    #             masks.append(obj_mask) 

    #         mask = masks[0]
    #         for mask_t in masks[1:]:
    #             mask = torch.logical_or(mask, mask_t)
    #         # print(f"Mask: {mask}")
    #         input_indices = torch.arange(0, num_input_images, dtype=labels.dtype, device=labels.device)
    #         input_indices = input_indices[:, None].repeat(1, self.num_patches)
    #         # print(f"Labels : {labels[mask]}")
    #         # print(f"Labels : {scores[mask]}")
    #         # print(f"Labels : {input_indices[mask]}")
    #         # print(f"Labels : {image_output.pred_boxes[i][mask]}")
    #         all_labels.append(labels[mask])
    #         all_scores.append(scores[mask])
    #         all_boxes.append(image_output.pred_boxes[i][mask])
    #         all_input_indices.append(input_indices[mask])
        
            
    #     # print(f"labels {(all_labels)}")
    #     # print(f"Scores {(all_scores)}")
    #     # print(f"boxes {(all_boxes)}")
    #     # print(f"input_indices {(all_scores)}")

    #     return OwlDecodeOutput(
    #         labels=torch.cat(all_labels),
    #         scores=torch.cat(all_scores),
    #         boxes=torch.cat(all_boxes),
    #         input_indices=torch.cat(all_input_indices)
    #     )

    def decode(self, 
            image_output: OwlEncodeImageOutput, 
            text_output: OwlEncodeTextOutput,
            threshold: Union[int, float, List[Union[int, float]]] = 0.1,
        ) -> OwlDecodeOutput:

        if isinstance(threshold, (int, float)):
            threshold = [threshold] * len(text_output.text_embeds) #apply single threshold to all labels 

        # print(f"Num Input :{image_output.image_class_embeds.shape}")
        # print(f"Num Input :{image_output.image_embeds.shape}")
        # print(f"Num Input :{image_output.pred_boxes.shape}")
        # print(f"Num Input :{image_output.logit_scale.shape}")
        # print(f"Num Input :{image_output.logit_shift.shape}")
        # # print(f"Num Input :{image_output.image_class_embeds}")
        # # print(f"Num Input :{image_output.image_embeds}")
        # # print(f"Num Input :{image_output.pred_boxes}")
        # # print(f"Num Input :{image_output.logit_scale}")
        # # print(f"Num Input :{image_output.logit_shift}")

        num_input_images = image_output.image_class_embeds.shape[0]

        image_class_embeds = image_output.image_class_embeds
        # print("Image class embeddings shape:", image_class_embeds.shape)  # Debugging
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        # # print(f"image_class_embeds : {image_class_embeds}")
        query_embeds = text_output.text_embeds
        # print("Query embeddings shape:", query_embeds.shape)  # Debugging
        # # print(f"query_embeds : {query_embeds}")
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)
        # # print(f"query_embeds : {query_embeds}")
        logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
        # print("Logits shape:", logits.shape)  # Debugging
        logits = (logits + image_output.logit_shift) * image_output.logit_scale
        # print(f"logits : {logits.shape}")
        # # print(f"logits : {logits}")
        scores_sigmoid = torch.sigmoid(logits)
        # print(f"scores_sigmoid : {scores_sigmoid.shape}")
        # # print(f"scores_sigmoid : {scores_sigmoid}")
        scores_max = scores_sigmoid.max(dim=-1)
        # # print(f"scores_max : {scores_max}")
        labels = scores_max.indices
        # print(f"labels : {labels.shape}")
        # # print(f"labels : {labels}")
        scores = scores_max.values
        # # print(f"scores : {scores}")
        # print(f"scores : {scores.shape}")
        masks = []
        for i, thresh in enumerate(threshold):
            label_mask = labels == i
            score_mask = scores > thresh
            obj_mask = torch.logical_and(label_mask,score_mask)
            masks.append(obj_mask) 
        # # print(f"masks : {masks}")
        mask = masks[0]
        # print(f"mask : {mask}")
        for mask_t in masks[1:]:
            mask = torch.logical_or(mask, mask_t)
        # print(f"mask : {mask.shape}")
        # # print(f"mask : {mask}")

        # Check if the mask is empty
        if mask.sum() == 0:
            print("Warning: The mask is empty. No elements satisfy the conditions.")
            return OwlDecodeOutput(
                labels=torch.tensor([], dtype=labels.dtype, device=labels.device),
                scores=torch.tensor([], dtype=scores.dtype, device=scores.device),
                boxes=torch.tensor([], dtype=image_output.pred_boxes.dtype, device=image_output.pred_boxes.device),
                input_indices=torch.tensor([], dtype=labels.dtype, device=labels.device)
            )

        input_indices = torch.arange(0, num_input_images, dtype=labels.dtype, device=labels.device)
        # print(input_indices.shape)
        # print(self.num_patches)
        input_indices = input_indices[:, None].repeat(1, self.num_patches)
        # print(input_indices.shape)
        # print("Labels shape before masking:", labels.shape)  # Debugging
        # print("Scores shape before masking:", scores.shape)  # Debugging
        # print("Image output boxes shape before masking:", image_output.pred_boxes.shape)  # Debugging
        # print("Input indices shape before masking:", input_indices.shape)  # Debugging
        # # print(input_indices)
        # print("Labels shape after masking:", len([labels[i][mask[i]] for i in range(labels.size(0))]))  # Debugging
        # # Print the shapes of the resulting tensors
        # for i, tensor in enumerate([labels[i][mask[i]] for i in range(labels.size(0))]):
        #     # print(f"Batch {i} masked labels shape: {tensor.shape}")
        # print("Scores shape after masking:", len([scores[i][mask[i]] for i in range(scores.size(0))]))  # Debugging
        # print("Image output boxes shape after masking:", ([image_output.pred_boxes[mask]][0].shape))  # Debugging
        # print("Input indices shape after masking:", [input_indices[mask]][0].shape)  # Debugging

        return OwlDecodeOutput(
            labels=[labels[i][mask[i]] for i in range(labels.size(0))],
            scores=[scores[i][mask[i]] for i in range(scores.size(0))],
            boxes=[image_output.pred_boxes[i][mask[i]] for i in range(image_output.pred_boxes.size(0))],
            input_indices=[input_indices[i][mask[i]] for i in range(input_indices.size(0))]
        )

    @staticmethod
    def get_image_encoder_input_names():
        return ["image"]

    @staticmethod
    def get_image_encoder_output_names():
        names = [
            "image_embeds",
            "image_class_embeds",
            "logit_shift",
            "logit_scale",
            "pred_boxes"
        ]
        return names


    def export_image_encoder_onnx(self, 
            output_path: str,
            use_dynamic_axes: bool = True,
            batch_size: int = 1,
            onnx_opset=17
        ):
        
        class TempModule(torch.nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def forward(self, image):
                output = self.parent.encode_image_torch(image)
                return (
                    output.image_embeds,
                    output.image_class_embeds,
                    output.logit_shift,
                    output.logit_scale,
                    output.pred_boxes
                )

        data = torch.randn(batch_size, 3, self.image_size, self.image_size).to(self.device)

        if use_dynamic_axes:
            dynamic_axes =  {
                "image": {0: "batch"},
                "image_embeds": {0: "batch"},
                "image_class_embeds": {0: "batch"},
                "logit_shift": {0: "batch"},
                "logit_scale": {0: "batch"},
                "pred_boxes": {0: "batch"}       
            }
        else:
            dynamic_axes = {}

        model = TempModule(self)

        torch.onnx.export(
            model, 
            data, 
            output_path, 
            input_names=self.get_image_encoder_input_names(), 
            output_names=self.get_image_encoder_output_names(),
            dynamic_axes=dynamic_axes,
            opset_version=onnx_opset
        )
    
    @staticmethod
    def load_image_encoder_engine(engine_path: str, max_batch_size: int = 1):
        import tensorrt as trt
        from torch2trt import TRTModule

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(engine_path, 'rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        base_module = TRTModule(
            engine,
            input_names=OwlPredictor.get_image_encoder_input_names(),
            output_names=OwlPredictor.get_image_encoder_output_names()
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, base_module: TRTModule, max_batch_size: int):
                super().__init__()
                self.base_module = base_module
                self.max_batch_size = max_batch_size
                print(self.max_batch_size)
                self.num = 1

            @torch.no_grad()
            def forward(self, image):
                # print("Check Engine")
                # print(image.shape)
                b = image.shape[0]
                # print(b)

                results = []
                # print(f"Batch : {self.max_batch_size} {b}")

                for start_index in range(0, b, self.max_batch_size):
                    
                    end_index = min(b, start_index + self.max_batch_size)
                    image_slice = image[start_index:end_index]
                    # print(f"Num : {self.num}")
                    # self.num += 1
                    # print(f"Image {image.shape} {image_slice.shape}")
                    # with torch_timeit_sync("run_engine"):
                    output = self.base_module(image_slice)
                    results.append(
                        output
                    )

                return OwlEncodeImageOutput(
                    image_embeds=torch.cat([r[0] for r in results], dim=0),
                    image_class_embeds=torch.cat([r[1] for r in results], dim=0),
                    logit_shift=torch.cat([r[2] for r in results], dim=0),
                    logit_scale=torch.cat([r[3] for r in results], dim=0),
                    pred_boxes=torch.cat([r[4] for r in results], dim=0)
                )

        image_encoder = Wrapper(base_module, max_batch_size)

        return image_encoder

    def build_image_encoder_engine(self, 
            engine_path: str, 
            max_batch_size: int = 2, 
            fp16_mode = True, 
            onnx_path: Optional[str] = None,
            onnx_opset: int = 17
        ):

        if onnx_path is None:
            onnx_dir = tempfile.mkdtemp()
            onnx_path = os.path.join(onnx_dir, "image_encoder.onnx")
            self.export_image_encoder_onnx(onnx_path,use_dynamic_axes=True , batch_size=max_batch_size, onnx_opset=onnx_opset)

        args = ["/usr/src/tensorrt/bin/trtexec"]
    
        args.append(f"--onnx={onnx_path}")
        args.append(f"--saveEngine={engine_path}")

        if fp16_mode:
            args += ["--fp16"]

        # Set dynamic shapes for TensorRT
        # args += [
        #     f"--minShapes=image:1x3x{self.image_size}x{self.image_size}",
        #     f"--optShapes=image:{max_batch_size//2}x3x{self.image_size}x{self.image_size}",
        #     f"--maxShapes=image:{max_batch_size}x3x{self.image_size}x{self.image_size}"
        # ]
        args += [f"--shapes=image:{max_batch_size}x3x{self.image_size}x{self.image_size}"]

        subprocess.call(args)

        return self.load_image_encoder_engine(engine_path, max_batch_size)


    def predict(self, 
                image: Union[PIL.Image.Image, torch.Tensor], 
                text: List[str], 
                text_encodings: Optional[OwlEncodeTextOutput],
                threshold: Union[int, float, List[Union[int, float]]] = 0.1,
                pad_square: bool = True,
            ) -> OwlDecodeOutput:
        
        if isinstance(image, PIL.Image.Image):
            image_tensor = self.image_preprocessor.preprocess_pil_image(image)

            rois = torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)
           
        elif isinstance(image, torch.Tensor):
            # print(text)
            # print(threshold)
            # print("Check0")
            # print(image[0].shape)
            # print(image.shape)
            # print(image)
            image_tensor = self.image_preprocessor.preprocess_tensor_image(image)
            # print(image_tensor.shape[0])
            # print("Check1")
            # print(image_tensor.shape)
            # For single batch
            # rois = torch.tensor([[0, 0, image.shape[1], image.shape[0]]], dtype=image_tensor.dtype, device=image_tensor.device)
            # For Multi batch
            # rois = torch.tensor([[0, 0, image.shape[2], image.shape[1]]], dtype=image_tensor.dtype, device=image_tensor.device)
            rois = torch.tensor([[0, 0, image.shape[1], image.shape[0]] for image in image], dtype=image_tensor.dtype, device=image_tensor.device)
            # Adjust rois to match the batch size
            # batch_size = image.shape[0]
            # rois = rois.repeat(batch_size, 1)

            # # Add batch indices to rois
            # batch_indices = torch.arange(batch_size, dtype=rois.dtype, device=rois.device).view(-1, 1)
            # rois = torch.cat([batch_indices, rois], dim=1)
            # print(f"rois : {rois.shape} ")  
            # # Define the batch size, height, and width
            # batch_size = image.shape[0]
            # height = image.shape[1]
            # width = image.shape[2]

            # # Create batch indices
            # batch_indices = torch.arange(batch_size, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1)

            # # Create coordinates for RoIs
            # coords = torch.tensor([0, 0, width, height], dtype=image_tensor.dtype, device=image_tensor.device).view(1, -1)

            # # Repeat the coordinates for each batch
            # coords = coords.repeat(batch_size, 1)

            # # Combine batch indices with coordinates
            # rois = torch.cat((batch_indices, coords), dim=1)
        else:
            raise ValueError("Input image must be either a PIL Image or a torch.Tensor")
        
        if text_encodings is None:
            text_encodings = self.encode_text(text)
        
        image_encodings = self.encode_rois(image_tensor, rois, pad_square=pad_square)
        # print(image_encodings)
        return self.decode(image_encodings, text_encodings, threshold)
    
    # def predict(self, 
    #         image: PIL.Image, 
    #         text: List[str], 
    #         text_encodings: Optional[OwlEncodeTextOutput],
    #         threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    #         pad_square: bool = True,
            
    #     ) -> OwlDecodeOutput:

    #     image_tensor = self.image_preprocessor.preprocess_pil_image(image)
    #     # print(image_tensor)
    #     if text_encodings is None:
    #         text_encodings = self.encode_text(text)

    #     rois = torch.tensor([[0, 0, image.width, image.height]], dtype=image_tensor.dtype, device=image_tensor.device)

    #     image_encodings = self.encode_rois(image_tensor, rois, pad_square=pad_square)

    #     return self.decode(image_encodings, text_encodings, threshold)
    
    # def predictTensor(self, 
    #                   image: torch.Tensor,
    #                   text: List[str],
    #                   text_encodings: Optional[OwlEncodeTextOutput],
    #                   threshold: Union[int, float, List[Union[int, float]]] = 0.1,
    #                   pad_square: bool = True,
                      
    #                 ) -> OwlDecodeOutput:
        
    #     image_tensor = self.image_preprocessor.preprocess_tensor_image(image)

    #     if text_encodings is None:
    #         text_encodings = self.encode_text(text)
    #     # print(image_tensor)
    #     ## print(image.shape[1])
    #     rois = torch.tensor([[0, 0, image.shape[1], image.shape[0]]], dtype=image_tensor.dtype, device=image_tensor.device)

    #     image_encodings = self.encode_rois(image_tensor, rois, pad_square=pad_square)
        
    #     return self.decode(image_encodings, text_encodings, threshold)
        