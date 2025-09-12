      
# Copyright 2025 Step1X-Edit Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL, Step1XEditTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import BaseOutput, is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import Step1XEditPipelineOutput
from .pipeline_step1x_edit_thinker import Step1XEditThinker


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import Step1XEditPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = Step1XEditPipeline.from_pretrained("stepfun-ai/Step1X-Edit-v1p1-diffusers", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = "Make Pikachu hold a sign that says 'Step1X-Edit is awesome', yarn art style, detailed, vibrant colors"

        >>> image = pipe(
                image=image,
                prompt=prompt, 
                num_inference_steps=28,
                true_cfg_scale=6.0,
                generator=torch.Generator().manual_seed(42),
            ).images[0]
        >>> image.save("output.png")
        ```
"""

EDIT_PREFIX = """Given a reference image and a user editing instruction.
You need to understand the information of the reference image and the content of the editing instruction,
and then generate a detailed description of the edited target image based on the reference image and the editing instruction.\u0020
The description should include the requirements in the editing instruction and try to closely match the content of the target image.
The editing instruction is:"""  # noqa: E501


def split_string_in_quotation(s: str, prefix_len: int) -> tuple[list[str], list[bool]]:  # noqa: C901
    result = []
    result_in_quotation = []

    state = "OUTSIDE"
    buffer = ""

    quote_content_buffer = []

    for idx, char in enumerate(s):
        if idx < prefix_len:
            buffer += char
            continue

        if state == "OUTSIDE":
            if char == '"':
                result.append(buffer + '"')
                result_in_quotation.append(False)
                buffer = ""
                state = "IN_ENGLISH_QUOTES"
            elif char == "“":
                if buffer:
                    result.append(buffer + "“")
                    result_in_quotation.append(False)
                buffer = ""
                state = "IN_CHINESE_QUOTES"
            else:
                buffer += char

        elif state == "IN_ENGLISH_QUOTES":
            if char == '"':
                result.extend(quote_content_buffer)
                result_in_quotation.extend([True] * len(quote_content_buffer))
                quote_content_buffer = []
                state = "OUTSIDE"
                buffer = '"'
            else:
                quote_content_buffer.append(char)

        elif state == "IN_CHINESE_QUOTES":
            if char == "”":
                result.extend(quote_content_buffer)
                result_in_quotation.extend([True] * len(quote_content_buffer))
                quote_content_buffer = []
                state = "OUTSIDE"
                buffer = "”"
            else:
                quote_content_buffer.append(char)

    if state != "OUTSIDE":
        malformed_text = "".join(quote_content_buffer) + buffer

        if result and not result_in_quotation[-1]:
            result[-1] += malformed_text
        else:
            result.append(malformed_text)
            result_in_quotation.append(False)
    elif buffer:
        result.append(buffer)
        result_in_quotation.append(False)

    return result, result_in_quotation


def split_string_in_quotation_and_special_tokens(
    s: str,
    prefix_len: int,
    special_tokens: list[str],
    should_split_in_quatation=True,
) -> list[str] | tuple[list[str], list[bool]]:
    if not special_tokens:
        if should_split_in_quatation:
            return split_string_in_quotation(s, prefix_len)
        return [s], [False]

    escaped_tokens = sorted(set(special_tokens), key=len, reverse=True)

    pattern = f"({'|'.join(escaped_tokens)})"

    parts = re.split(pattern, s)
    parts = [x for x in parts if len(x) > 0]
    parts_start_index = np.cumsum([0] + [len(part) for part in parts[:-1]]).tolist()

    if not should_split_in_quatation:
        return parts
    else:
        processed_parts = []
        for part, part_start_index in zip(parts, parts_start_index):
            if part not in special_tokens:
                split_result = split_string_in_quotation(part, max(0, prefix_len - part_start_index))
                processed_parts.append(split_result)
            else:
                processed_parts.append(([part], [False]))

        if not processed_parts:
            return [], []

        split_str_processed_parts, in_quatation_flag_processed_parts = zip(*processed_parts)
        split_str_processed_parts = functools.reduce(operator.iadd, split_str_processed_parts, [])
        in_quatation_flag_processed_parts = functools.reduce(operator.iadd, in_quatation_flag_processed_parts, [])
        return split_str_processed_parts, in_quatation_flag_processed_parts

@dataclass
class TextEmbedderOutput(BaseOutput):
    embedding: torch.Tensor
    mask: torch.Tensor
    txt_ids: torch.Tensor
    text_embeds: torch.Tensor
    text_masks: torch.Tensor

 # Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class Step1XEditPipelineV1P2(DiffusionPipeline):
    r"""
    The Step1X-Edit pipeline for image-to-image and text-to-image generation.

    Reference: https://arxiv.org/abs/2504.17761

    Args:
        transformer ([`Step1XEditTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
        processor (`Qwen2_5_VLProcessor`):
            [Qwen2_5_VLProcessor](https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/qwen2_5_vl#transformers.Qwen2_5_VLProcessor).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        processor: Qwen2_5_VLProcessor,
        transformer: Step1XEditTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.image_encoder=None
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Step1X-Edit latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.max_token_length = 2048
        self.default_sample_size = 128
        self.QWEN25VL_PREFIX = EDIT_PREFIX
        self.QWEN25VL_PREFIX_LEN = 90

        self.processor = AutoProcessor.from_pretrained(
            self.processor.tokenizer.name_or_path,
            min_pixels=256 * 28 * 28,
            max_pixels=400 * 28 * 28,
        )


    @torch.no_grad()
    def get_vae_downsample_affine_mat(self, device):
        if not hasattr(self, "vae_affine_mat"):
            self.vae_affine_mat = torch.Tensor(
                [
                    [1 / 16, 0, -11 / 16],
                    [0, 1 / 16, -11 / 16],
                    [0, 0, 1],
                ]
            ).to(device=device, dtype=torch.float32)
        return self.vae_affine_mat

    @staticmethod
    def _special_token_strings(tokenizer):
        tokens = []
        tokens.extend(tokenizer.additional_special_tokens)
        tokens.extend([t.content for t in tokenizer.added_tokens_decoder.values()])
        tokens = [re.escape(x) for x in tokens]
        return tokens

    @staticmethod
    def _get_affine_mat_from_source(resized_width_height, source_width_height, device):
        dtype = torch.float32
        source_width, source_height = source_width_height
        resized_width, resized_height = resized_width_height

        resize_affine_mat = torch.eye(3)
        resize_affine_mat[0, 0] = (resized_width - 1) / (source_width - 1)
        resize_affine_mat[1, 1] = (resized_height - 1) / (source_height - 1)

        # vision_encoder_affine_mat = torch.eye(3)
        vision_encoder_affine_mat = torch.Tensor(
            [
                [1 / 28, 0, -13.5 / 28],
                [0, 1 / 28, -13.5 / 28],
                [0, 0, 1],
            ]
        )

        final_affine_mat = vision_encoder_affine_mat @ resize_affine_mat
        return final_affine_mat.to(device=device, dtype=dtype)

    @torch.no_grad()
    def set_vision_tokens_position_ids(
        self, txt_ids, vision_token_hw, affine_mat, vision_token_mask, device
    ):
        # txt_ids: should be flattened txt_idx, shape: (L, 3)
        # affine_mat: should be the pixel value space to the vision token space, (3, 3)
        # vision_token_mask: should be the bool (L, 3)
        dtype = torch.float32

        v_h = vision_token_hw[0]
        v_w = vision_token_hw[1]
        vision_token_idx = torch.zeros((v_h, v_w, 3), dtype=dtype, device=device)
        vision_token_idx[..., 1] = vision_token_idx[..., 1] + torch.arange(v_h, dtype=dtype, device=device)[:, None]
        vision_token_idx[..., 2] = vision_token_idx[..., 2] + torch.arange(v_w, dtype=dtype, device=device)[None, :]

        vision_token_idx_xy_shape = vision_token_idx[..., 1:].shape
        # notice that the axis 1 is y, axis 2 is x
        vision_token_idx_xy = vision_token_idx[..., [2, 1]].reshape(-1, 2)
        affin_mat = self.get_vae_downsample_affine_mat(device) @ torch.linalg.inv(affine_mat).to(dtype)
        vision_token_idx_xy = vision_token_idx_xy @ affin_mat[:2, :2].T + affin_mat[:2, -1][None]

        # notice that the axis 1 is y, axis 2 is x, set xy to [2, 1]
        vision_token_idx[..., [2, 1]] = vision_token_idx_xy.view(vision_token_idx_xy_shape)
        vision_token_idx = vision_token_idx.reshape(-1, 3)

        # do not change position of time
        txt_ids[..., 1:].masked_scatter_(vision_token_mask[:, 1:], vision_token_idx[:, 1:])


    def _get_qwenvl_embeds(
        self,
        prompt: Union[str, List[str]],
        ref_image: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> tuple[torch.Tensor]:
        text_list = prompt
        embs = torch.zeros(
            len(text_list),
            self.max_token_length,
            self.text_encoder.config.hidden_size,
            dtype=dtype,
            device=device,
        )
        masks = torch.zeros(
            len(text_list),
            self.max_token_length,
            dtype=torch.long,
            device=device,
        )
        text_embeds = torch.zeros(
            len(text_list),
            self.max_token_length,
            self.text_encoder.config.hidden_size,
            dtype=dtype,
            device=device,
        )

        vision_masks = torch.zeros(
            len(text_list), self.max_token_length, dtype=torch.long, device=device
        )
        vision_affine_mats = torch.zeros((len(text_list), 3, 3), dtype=torch.float32, device=device)
        vision_tokens_thw = torch.zeros((len(text_list), 3), dtype=torch.int32)
        text_masks = torch.zeros(len(text_list), self.max_token_length, dtype=torch.long, device=device)
        out_position_ids_list = torch.zeros(len(text_list), self.max_token_length, 3, dtype=torch.float32, device=device)

        for idx, (txt, imgs) in enumerate(zip(text_list, ref_image)):

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{self.QWEN25VL_PREFIX} {txt}"},
                        {"type": "image", "image": imgs},
                    ],
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            imgs = imgs.convert("RGB")
            image_inputs = [imgs]

            text_split_list, is_text_list = split_string_in_quotation_and_special_tokens(
                text,
                prefix_len=len(self.QWEN25VL_PREFIX),
                special_tokens=self._special_token_strings(self.processor.tokenizer),
            )

            token_list = []
            text_mask_list = []
            inputs_pixel_values = None
            inputs_image_grid_thw: torch.Tensor = None
            for text_each, is_text in zip(text_split_list, is_text_list):
                if "<|image_pad|>" in text_each:
                    inputs = self.processor(
                        text=text_each,
                        images=image_inputs,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs_pixel_values = inputs.pixel_values
                    inputs_image_grid_thw = inputs.image_grid_thw
                else:
                    inputs = self.processor(
                        text=text_each,
                        images=None,
                        videos=None,
                        padding=True,
                        return_tensors="pt",
                    )

                token_each = inputs.input_ids

                if is_text:
                    token_list.append(token_each)
                    text_mask_list.append(torch.ones_like(token_each))
                else:
                    token_list.append(token_each)
                    text_mask_list.append(torch.zeros_like(token_each))

            input_ids = torch.cat(token_list, dim=1).to(device)
            text_mask = torch.cat(text_mask_list, dim=1).to(device)
            attention_mask = (input_ids > 0).long().to(device)

            image_grid_thw = inputs_image_grid_thw.to(device)
            pixel_values = inputs_pixel_values.to(device)
            affine_mat_i = self._get_affine_mat_from_source(
                (inputs_image_grid_thw[0, -1] // 2 * 28, inputs_image_grid_thw[0, -2] // 2 * 28),
                imgs.size,  # type: ignore
                device=device,
            )

            inputs_embeds = self.text_encoder.model.language_model.embed_tokens(input_ids)
            if inputs_pixel_values is not None:
                pixel_values = pixel_values.type(self.text_encoder.model.visual.dtype)
                with torch.no_grad():
                    self.text_encoder.model.visual.rotary_pos_emb = self.text_encoder.model.visual.rotary_pos_emb.float()
                    image_embeds = self.text_encoder.model.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.text_encoder.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"  # noqa: E501
                    )

                mask = input_ids == self.text_encoder.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


            position_ids, rope_deltas = self.text_encoder.model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,  # video_grid_thw,
                None,  # second_per_grid_ts
                attention_mask,
            )
            self.text_encoder.model.rope_deltas = rope_deltas

            outputs = self.text_encoder(
                input_ids = input_ids,
                position_ids = position_ids,
                output_hidden_states=True,
                pixel_values = pixel_values,
                image_grid_thw = inputs_image_grid_thw,
            )

            emb = outputs['hidden_states'][-1]

            cut_length = min(self.max_token_length,emb.shape[1]-self.QWEN25VL_PREFIX_LEN)

            embs[idx,:cut_length] = emb[0,self.QWEN25VL_PREFIX_LEN:][:self.max_token_length]
            masks[idx, :cut_length] = torch.ones((cut_length), dtype=torch.long, device=device)

            vision_mask = input_ids == self.text_encoder.config.image_token_id
            vision_masks[idx, :cut_length] = vision_mask[0, self.QWEN25VL_PREFIX_LEN:][: self.max_token_length]
            vision_affine_mats[idx] = affine_mat_i
            vision_tokens_thw[idx : idx + 1] = inputs_image_grid_thw // 2
            text_embeds[idx, :cut_length] = inputs_embeds[0, self.QWEN25VL_PREFIX_LEN:][: self.max_token_length]
            text_masks[idx, :cut_length] = text_mask[0, self.QWEN25VL_PREFIX_LEN:][: self.max_token_length]
            out_position_ids_list[idx, :cut_length] = position_ids.permute(1, 2, 0)[0, self.QWEN25VL_PREFIX_LEN:][:self.max_token_length]
            out_position_ids_list[idx, :cut_length] -= out_position_ids_list[idx, :cut_length, 0].max() + 1

        return (
            embs,
            masks,
            vision_masks,
            vision_affine_mats,
            vision_tokens_thw,
            out_position_ids_list,
            text_embeds,
            text_masks,
        )

    def encode_prompt(
        self,
        ref_image: Optional[torch.Tensor],
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> TextEmbedderOutput:
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self._execution_device

        ref_image = [ref_image] if isinstance(prompt, str) else ref_image # change
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        assert batch_size == 1
        assert prompt_embeds is None

        (
            embedding,
            mask,
            vision_masks,
            vision_affine_mats,
            vision_tokens_thw,
            position_ids,
            text_embeds,
            text_masks
        ) = self._get_qwenvl_embeds(prompt, ref_image, device)


        txt_ids = position_ids.flatten(0, 1)[mask.bool().flatten(0, 1)].contiguous()
        txt_ids[:, 1:] = 0

        vision_token_mask_for_wo_padding = (
            vision_masks.int() * torch.arange(1, 2, dtype=torch.int64, device=device)[:, None]
        ).flatten(0, 1)[mask.bool().flatten(0, 1)]

        vision_tokens_thw = vision_tokens_thw.to(torch.int32).cpu()
        self.set_vision_tokens_position_ids(
            txt_ids,
            vision_token_hw=vision_tokens_thw[0, 1:],
            affine_mat=vision_affine_mats[0],
            vision_token_mask=(vision_token_mask_for_wo_padding == 1)
            .unsqueeze(-1)
            .expand_as(txt_ids),
            device=device,
        )  # type: ignore

        # # BL -> 1xBL -> NxBL -> NBL
        # txt_ids_list = txt_ids.split(mask.sum(dim=1).tolist())
        # txt_ids = torch.cat([x[None].repeat(num_images_per_prompt, 1, 1).flatten(0, 1) for x in txt_ids_list], dim=0)

        valid_len = mask.sum().item()
        embedding = torch.repeat_interleave(embedding, num_images_per_prompt, dim=0)[:, :valid_len]
        mask = torch.repeat_interleave(mask, num_images_per_prompt, dim=0)[:, :valid_len]
        text_embeds = torch.repeat_interleave(text_embeds, num_images_per_prompt, dim=0)[:, :valid_len]
        text_masks = torch.repeat_interleave(text_masks, num_images_per_prompt, dim=0)[:, :valid_len]
        # txt_ids = txt_ids.reshape(num_images_per_prompt, -1, 3)

        return TextEmbedderOutput(
            embedding=embedding,
            mask=mask,
            txt_ids=txt_ids,
            text_embeds=text_embeds,
            text_masks=text_masks,
        )

    def encode_image(
        self,
        image: Optional[torch.Tensor],
        width: Optional[int] = None,
        height: Optional[int] = None,
        size_level: int = 1024,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
    ):

        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img_info = image.size
            width, height = img_info
            aspect_ratio = width / height

            if width > height:
                width_new = math.ceil(math.sqrt(size_level * size_level * aspect_ratio))
                height_new = math.ceil(width_new / aspect_ratio)
            else:
                height_new = math.ceil(math.sqrt(size_level * size_level / aspect_ratio))
                width_new = math.ceil(height_new * aspect_ratio)

            multiple_of = self.vae_scale_factor * 2
            height_new = height_new // multiple_of * multiple_of
            width_new = width_new // multiple_of * multiple_of

            if height != height_new or width != width_new:
                logger.warning(
                    f"Generation `height` and `width` have been adjusted to {height_new} and {width_new} to fit the model requirements."
                )
            height, width = height_new, width_new
            ref_image = self.image_processor.resize(image, height, width)
            image = torch.from_numpy(np.asarray(image.convert(mode="RGB")).copy()).float().permute(2, 0, 1)[None] / 255.0
            image = torch.nn.functional.interpolate(image, (height, width), mode="bilinear", antialias=True)
            image = image * 2 - 1
        else:
            width = width if width is not None else size_level
            height = height if height is not None else size_level
            img_info = (width, height)
            ref_image = torch.zeros(3, size_level, size_level).unsqueeze(0).to(device)
            ref_image = self.image_processor.pt_to_numpy(ref_image)
            ref_image = self.image_processor.numpy_to_pil(ref_image)[0]
            image = None

        return image, ref_image, img_info, width, height

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @staticmethod
    def _output_process_image(image, image_size):
        res_image = [img.resize(image_size) for img in image]
        return res_image

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        image_latents = None
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="sample")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="sample")

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, torch.float32  # change
                # batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1
        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, torch.float32)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)   # change
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 6.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        size_level: int = 1024,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 6.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor | TextEmbedderOutput] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        timesteps_truncate: float = 0.93,
        process_norm_power: float = 0.4,
        enable_thinking_mode: bool = True,
        enable_reformat_prompt: bool = False,
        max_try_cnt: int = 3,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 6.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] or `tuple`:
            [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        device = self._execution_device

        try_cnt = 0
        success = False

        if enable_thinking_mode:
            thinker = Step1XEditThinker(self.text_encoder, self.processor)
            if enable_reformat_prompt:
                reformat_prompt = thinker.prompt_reformat(image, prompt)
            else:
                reformat_prompt = prompt
            prompt = reformat_prompt
            out_images = []
            out_think_info = []
        else:
            max_try_cnt = 1
            out_images = None

        original_ref_image = image
        original_prompt = prompt

        while not success and try_cnt < max_try_cnt:
            # 1. Preprocess image
            image, ref_image, img_info, width, height = self.encode_image(
                image,
                width,
                height,
                size_level,
                device,
                num_images_per_prompt
            )

            # 2. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
            )

            self._guidance_scale = guidance_scale
            self._joint_attention_kwargs = joint_attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            # 3. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            lora_scale = (
                self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
            )
            has_neg_prompt = negative_prompt is not None or (
                negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
            )
            if not has_neg_prompt:
                negative_prompt = "" if image is not None else "worst quality, wrong limbs, unreasonable limbs, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
            do_true_cfg = true_cfg_scale > 1
            prompt_embeds = self.encode_prompt(
                ref_image=ref_image,
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

            if do_true_cfg:
                negative_prompt_embeds = self.encode_prompt(
                    ref_image=ref_image,
                    prompt=negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                )

            # 4. Prepare latent variables
            num_channels_latents = self.transformer.config.in_channels // 4
            latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.embedding.dtype,
                device,
                generator,
            )
            if image_ids is not None:
                latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

            # 5. Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
                ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}

            image_embeds = None
            negative_image_embeds = None
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )
            if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
                negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )

            # 6. Denoising loop
            # We set the index here to remove DtoH sync, helpful especially during compilation.
            # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
            self.scheduler.set_begin_index(0)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                    latent_model_input = latents
                    if image_latents is not None:
                        latent_model_input = torch.cat([latents, image_latents], dim=1)
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds.embedding,
                        prompt_embeds_mask=prompt_embeds.mask,
                        txt_ids=prompt_embeds.txt_ids,
                        img_ids=latent_ids,
                        text_embeddings=prompt_embeds.text_embeds,
                        text_mask=prompt_embeds.text_masks,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=negative_prompt_embeds.embedding,
                            prompt_embeds_mask=negative_prompt_embeds.mask,
                            txt_ids=negative_prompt_embeds.txt_ids,
                            img_ids=latent_ids,
                            text_embeddings=negative_prompt_embeds.text_embeds,
                            text_mask=negative_prompt_embeds.text_masks,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        if t.item() > timesteps_truncate:
                            diff = noise_pred - neg_noise_pred
                            diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                            noise_pred = neg_noise_pred + true_cfg_scale * (
                                noise_pred - neg_noise_pred
                            ) / self.process_diff_norm(diff_norm, k=process_norm_power)
                        else:
                            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

            self._current_timestep = None

            if output_type == "latent":
                image = latents
            else:
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                image = self._output_process_image(image, img_info)

            if enable_thinking_mode:
                thinking_info = thinker(original_ref_image, image[0], original_prompt)
                success, refine_prompt = thinker.format_text(thinking_info)
                out_images.append(image[0])
                out_think_info.append(thinking_info)
                if not success:
                    if refine_prompt is not None:  # type: ignore
                        prompt = refine_prompt
                        image = image[0]
                    else:
                        image = original_ref_image
                        prompt = reformat_prompt
                    try_cnt += 1
            else:
                out_images = image
                break
        # Offload all models
        self.maybe_free_model_hooks()
        if enable_thinking_mode:
            if enable_reformat_prompt:
                return Step1XEditPipelineOutput(images=out_images, reformat_prompt=reformat_prompt, think_info=out_think_info)
            else:
                return Step1XEditPipelineOutput(images=out_images, think_info=out_think_info)
        else:
            if not return_dict:
                return (image,)

            return Step1XEditPipelineOutput(images=out_images)

    