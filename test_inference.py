import torch
from diffusers import Step1XEditPipelineV1P2
from diffusers.utils import load_image
# pipe = Step1XEditPipelineV1P2.from_pretrained("/mnt/sirui/model_zoo/Step1X-Edit-diffusers", torch_dtype=torch.bfloat16)
pipe = Step1XEditPipelineV1P2.from_pretrained("/mnt/sirui/model_zoo/Step1X-Edit-diffusers_v1p2", torch_dtype=torch.bfloat16)
pipe.to("cuda")
print("=== processing image ===")
image = load_image("0000.jpg").convert("RGB")
prompt = "给这个女生的脖子上戴一个带有红宝石的吊坠。"
enable_thinking_mode=True
enable_reformat_prompt=True
pipe_output = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=50,
    true_cfg_scale=4,
    generator=torch.Generator().manual_seed(42),
    size_level=512,
    enable_thinking_mode=True,
    enable_reformat_prompt=True,
)
if enable_reformat_prompt:
    print("Reformat Prompt:", pipe_output.reformat_prompt)
for image_idx in range(len(pipe_output.images)):
    pipe_output.images[image_idx].save(f"0000-{image_idx}.jpg", lossless=True)
    print(pipe_output.think_info[image_idx])