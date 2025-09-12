#!/usr/bin/env python3

import base64
import json
import re
from io import BytesIO
from typing import Any

import megfile
import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def validate_string(input_str: str) -> bool:
    if not input_str.startswith("<#Think>") or input_str.count("<#Think>") != 1:
        return False

    tag_failed_count = input_str.count("<#Failed>")
    tag_success_count = input_str.count("<#Success>")
    tag_reflection_count = input_str.count("<#Reflection>")
    tag_others_count = input_str.count("<#Others>")

    total_tags_count = tag_failed_count + tag_success_count + tag_reflection_count + tag_others_count

    return total_tags_count == 1


def analyze_text(prompts):
    if isinstance(prompts, str):
        prompts = [prompts]

    chinese_char_count = 0
    english_char_count = 0

    chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
    english_pattern = re.compile(r"[a-zA-Z]")

    for text in prompts:
        if not isinstance(text, str):
            continue

        chinese_chars = chinese_pattern.findall(text)
        chinese_char_count += len(chinese_chars)

        english_chars = english_pattern.findall(text)
        english_char_count += len(english_chars)

    if chinese_char_count > 1:
        return "中文"
    else:
        return "英文"



def image_to_data_url(image_pil, max_side: int = 1024) -> str:

    if isinstance(image_pil, str):
        newf = megfile.smart_open(image_pil, "rb")
        image_pil = Image.open(newf)
    im = image_pil.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def replace_module(prefix: str, ref_model_parameters: dict[str, nn.Parameter], module: nn.Module):
    for name, _param in module.named_parameters(recurse=False):
        ref_name = prefix + "." + name if len(prefix) > 0 else name
        ref_name = ref_name.replace("language_model.", "model.")

        _param.data = ref_model_parameters[ref_name].data
        ref_model_parameters.pop(ref_name)

    for name, children in module.named_children():
        children_name = prefix + "." + name if len(prefix) > 0 else name
        replace_module(children_name, ref_model_parameters, children)


class QwenGenerativeAIClient:
    def __init__(self, model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor):
        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            processor.tokenizer.name_or_path,
            min_pixels=262144,
            max_pixels=802816,
        )

    @torch.no_grad()
    def call_model(self, messages: list[dict], max_tokens: int, retries: int = 3, timeout: int = 60):
        for each_message in messages:
            if each_message["role"] == "user":
                if isinstance(each_message["content"], list):
                    for content in each_message["content"]:
                        if content["type"] == "image_url":
                            content["type"] == "image"
                            content_image_url = content.pop("image_url")["url"]
                            content_image_url = content_image_url.replace(
                                "data:image/png;base64,", "data:image;base64,"
                            )
                            content["image"] = content_image_url

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()


class ImageEditEvaluator:

    def __init__(self, model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor):

        self.client = QwenGenerativeAIClient(model, processor)

    def prompt_reformat(self, source_image: Image.Image, original_instruction: str):
        src_data_url = image_to_data_url(source_image)
        messages = self.build_reformt_prompt(src_data_url, original_instruction)
        resp_text = self.client.call_model(messages, max_tokens=512)

        if resp_text.startswith("错误:"):
            raise RuntimeError(f"API call failed: {resp_text}")

        return resp_text.strip()

    def run_full_pipeline(self, source_image, edited_image, original_instruction: str) -> dict[str, Any]:
        while True:
            try:
                generation_result = self.generate_descriptions(
                    source_image=source_image, instruction=original_instruction
                )
                break
            except Exception as e:
                print(f"❌ 描述生成步骤出错: {e}")

        while True:
            try:
                evaluation_result = self.evaluate_consistency(
                    target_image=edited_image,
                    instruction=original_instruction,
                    description=generation_result,
                )
                break
            except Exception as e:
                print(f"❌ 评估步骤出错: {e}")

        while True:
            try:
                reflection_result = self.generate_reflection(
                    source_image=source_image,
                    edited_image=edited_image,
                    instruction=original_instruction,
                    evaluation_result=evaluation_result,
                )
                break
            except Exception as e:
                print(f"❌ 反思步骤出错: {e}")

        return {"generation": generation_result, "evaluation": evaluation_result, "reflection": reflection_result}

    def evaluate_consistency(self, target_image, instruction: str, description: str) -> dict[str, Any]:
        img_data_url = image_to_data_url(target_image)
        messages = self._build_scoring_messages(img_data_url, description=description, instruction=instruction)
        resp_text = self.client.call_model(messages, max_tokens=512)

        if resp_text.startswith("错误:"):
            raise RuntimeError(f"API call failed: {resp_text}")

        obj = self._parse_json_block(resp_text)
        if obj is None:
            raise ValueError(f"无法从API响应中解析JSON: {resp_text[:200]}...")

        final_result = self._apply_score_caps_and_finalize(obj)
        if final_result is None:
            raise ValueError(f"结构化字段缺失或异常，无法后处理: {obj}")

        return final_result

    def generate_descriptions(self, source_image, instruction: str, language_type=None):
        src_data_url = image_to_data_url(source_image)
        messages = self.build_descriptions_messages(src_data_url, instruction, language_type)
        resp_text = self.client.call_model(messages, max_tokens=768)

        if resp_text.startswith("错误:"):
            raise RuntimeError(f"API call failed: {resp_text}")

        return resp_text.strip()

    def generate_reflection(
        self, source_image, edited_image, instruction: str, evaluation_result: dict[str, Any]
    ) -> str:
        src_data_url = image_to_data_url(source_image)
        tgt_data_url = image_to_data_url(edited_image)

        prior_score = evaluation_result.get("score", evaluation_result.get("raw_score", "N/A"))
        prior_reason = evaluation_result.get("reason", "无")

        messages = self._build_reflection_messages(
            src_img_data_url=src_data_url,
            tgt_img_data_url=tgt_data_url,
            instruction_cn=instruction,
            prior_score=prior_score,
            prior_reason=prior_reason,
        )

        resp_text = self.client.call_model(messages, max_tokens=768)

        is_valid, cleaned_text = self._validate_reflection_response(resp_text)
        if not is_valid:
            raise ValueError(f"反思结果格式无效: {cleaned_text}")

        return cleaned_text


    def _parse_json_block(self, content: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def build_reformt_prompt(self, source_image_data_url: str, original_instruction: str) -> list[dict]:
        language_type = analyze_text(original_instruction)
        system_prompt = (
            "You are a precise visual-to-text describer. "
            "Your goal is to produce a concise, factual description of the input image "
            "that would be obtained after applying the edit instruction to the input image. "
            "Only describe what is explicitly visible in the input image."
        )
        user_text = f"""
        你是一个“编辑任务指令改写器”。你的任务是：
        在保证含义完全不变的情况下，请将用户输入的模糊、口语化、不正式的中文编辑指令转换为一个清晰、标准、可操作的编辑指令。
        用户输入一些复杂的含义时，请将他拆解成两到三个简单明确的编辑指令。确保这些编辑指令彼此不冲突，并且保留下来有意义的编辑指令。请只输出改写后的内容。
        最终得到的编辑指令尽可能只包含一个编辑动作。如果含义过于复杂不能用一个编辑动作来描述，最终的编辑动作数量也不能超过三个。
        如果输入中存在明确的文字修改，请将要修改的文字用双引号标注出来。
        编辑的类型主要分为：物体增加，物体替换，物体删除，颜色修改，背景替换，风格变化，材质变化，文字替换等。
        最终的编辑指令尽可能包含编辑的位置、动作（参考编辑类型）、被编辑的物体。请尽可能仿照自然语言的习惯，不允许出现编号，可以用自然语言描述的先后关系来代替编号。
        如果用户的输入需要理解和推理，请思考根据用户的描述，图片最后会在哪里发生变化，而不是只返回一个理解或推理之前的结果。
        请模仿下面例子风格来进行改写。
        例子1：用户输入：叶子缺钾的症状 改写后：叶子变黄，叶尖干枯。
        例子2：用户输入：面包变成煎过头的样子 改写后：将面包的颜色变成深色焦黄色。
        例子2：用户输入：帮我p一个男朋友，出一张双人照 改写后：在女生旁边添加一个男生。
        例子3：用户输入：帮我优化这张图片，去掉远处的电线，保留写实风格 改写后：删除电线。
        例子4：用户输入：帮我把图片中的TOFEL修改为IELTS 改写后：将文本 “TOFEL” 替换为 “IELTS”。
        例子5：用户输入：帮我调成证件照。 改写后：将背景替换成白色。
        例子6：用户输入：帮我把领带换成黑色的。 改写后：将领带的颜色变成黑色。
        例子7：用户输入：p掉右下角的水印。 改写后：移除右下角水印。
        例子8：用户输入：增加胸部的发量。将发型改为长发。使人物看起来更温柔。改写后：增加头发数量，使人物看起来更温柔。
        用户输入指令为：{original_instruction}
        
        你的返回指令必须以{language_type}的语言类型返回。你的返回指令必须以{language_type}的语言类型返回。你的返回指令必须以{language_type}的语言类型返回。
        返回的指令为："""  # noqa: W293

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": source_image_data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def build_descriptions_messages(self, source_image_data_url: str, instruction_cn: str, language_type=None):
        system_prompt = (
            "You are a precise visual-to-text describer. "
            "Your goal is to produce a concise, factual Chinese description of the **final target image** "
            "that would be obtained after applying the edit instruction to the input image. "
            "Only describe what is explicitly visible in the final target image."
        )
        language_type = analyze_text(instruction_cn)

        user_text = (
            "你将看到一张输入图像和一条编辑指令，请根据它们生成**目标图**的中文描述，必须遵循以下原则：\n"
            "1. **严格执行指令修改**：凡是指令明确要求修改的内容，必须在描述中准确体现，不能忽略或替换为与指令不符的内容。\n"  # noqa: E501
            "2. **未被指令修改的部分**：应与输入图保持一致，但若保留会导致与指令冲突，则必须按指令的预期替换为合理、简洁的形式（例如指令要求生成证件照，则背景必须符合证件照规范，不得保留原图场景）。\n"  # noqa: E501
            "3. **禁止幻觉**：不得添加、推测、替换或润色任何指令和输入图中未明确出现的元素（包括地点、环境、时间、光照、人物身份、服装细节、物体材质等）。\n"  # noqa: E501
            "4. **保持简洁**：用一段简明的中文描述目标图，准确传达画面中可见的关键人物、物体、姿态、背景和颜色；不要出现“原图”、“保持不变”、“指令中要求”等措辞。\n"  # noqa: E501
            "5. **优先忠实**：若输入图信息与指令冲突，以指令为准；不要混合保留会违背指令的元素。\n\n"
            f"编辑指令：{instruction_cn}\n"
            f"请用{language_type}描述目标图。请直接给出目标图的描述："
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": source_image_data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _build_scoring_messages(self, img_data_url: str, description: str, instruction: str) -> list[dict]:
        sys_prompt = "你是严格的图文一致性评估器，只输出严格 JSON。"
        language_type = analyze_text(instruction)
        user_prompt = (
            "在满足【编辑指令】前提下，评估【图片】与【文字描述】相符度。\n"
            "请只输出一个 JSON 对象，包含以下键：\n"
            '  - meets_instruction: "yes" | "partial" | "no"\n'
            "  - major_conflicts: 非负整数\n"
            "  - minor_conflicts: 非负整数\n"
            "  - hallucinations:  非负整数\n"
            "  - omissions:       非负整数\n"
            "  - raw_score:       0 到 10 的浮点数\n"
            f"  - reason:          简短{language_type}理由，先说明是否满足指令，再指出主要一致/不一致点\n"
            "判定要点：\n"
            "1) 指令满足度优先；\n"
            "2) 明显矛盾计入 major_conflicts；轻微出入计入 minor_conflicts；\n"
            "3) 描述加入图片中不存在的细节/推测/修饰，计入 hallucinations；\n"
            "4) 描述遗漏图片中的关键要素，计入 omissions；\n"
            "5) raw_score 参考 0..10，但不要包含任何多余文本。\n\n"
            f"【编辑指令】\n{instruction}\n\n"
            f"【文字描述】\n{description}\n"
        )
        return [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": img_data_url}},
                ],
            },
        ]

    def _apply_score_caps_and_finalize(self, obj: dict[str, Any]) -> dict[str, Any] | None:  # noqa: C901
        try:
            mi = str(obj.get("meets_instruction", "")).strip().lower()
            major = int(obj.get("major_conflicts", 0) or 0)
            minor = int(obj.get("minor_conflicts", 0) or 0)
            hallu = int(obj.get("hallucinations", 0) or 0)
            omit = int(obj.get("omissions", 0) or 0)
            raw = float(obj.get("raw_score", 0))
            reason = str(obj.get("reason", "")).strip()
            # 原始分裁剪到 0..10
            if 0 <= raw <= 1.0000001:
                raw *= 10.0
            raw = max(0.0, min(10.0, raw))
            cap = 10.0
            cap_trace: list[str] = []
            # 指令满足度封顶
            if mi == "no":
                cap = min(cap, 2.0)
                cap_trace.append("cap<=2: 未满足指令关键要求")
            elif mi == "partial":
                cap = min(cap, 5.0)
                cap_trace.append("cap<=5: 仅部分满足指令")
            # 冲突封顶
            if major >= 2:
                cap = min(cap, 5.0)
                cap_trace.append("cap<=5: 多个明显矛盾")
            elif major == 1:
                cap = min(cap, 8.0)
                cap_trace.append("cap<=8: 存在1个明显矛盾")
            # 幻觉/遗漏封顶
            if hallu > 0 or omit > 0:
                cap = min(cap, 7.0)
                why = []
                if hallu > 0:
                    why.append(f"幻觉x{hallu}")
                if omit > 0:
                    why.append(f"遗漏x{omit}")
                cap_trace.append(f"cap<=7: {', '.join(why)}")
            # 存在明显差异不允许≥9（当 major>0 或 mi!='yes'）
            if mi != "yes" or major > 0 or hallu > 0 or omit > 0:
                cap = min(cap, 8.99)
                cap_trace.append("cap<9: 存在明显差异/问题")
            final_score = min(raw, cap)
            return {
                "score": float(f"{final_score:.4f}"),
                "raw_score": raw,
                "cap_trace": cap_trace,
                "meets_instruction": mi,
                "major_conflicts": major,
                "minor_conflicts": minor,
                "hallucinations": hallu,
                "omissions": omit,
                "reason": reason,
            }
        except Exception:
            return None

    def _build_reflection_messages(
        self, src_img_data_url: str, tgt_img_data_url: str, instruction_cn: str, prior_score: Any, prior_reason: str
    ) -> list[dict[str, Any]]:
        new_query_template = """第一张图是输入图，第二张图是根据图像“编辑指令”编辑后得到的结果。
    当前的编辑指令为：
    {}
    希望能根据第两张图的内容和编辑指令，判断编辑后的结果是否符合要求，要求输出格式为：
    ```
    <#Think> ...
    <#Reflection> ...
    ```
    或者
    ```
    <#Think> ...
    <#Failed>
    ```
    或者
    ```
    <#Think> ...
    <#Success>
    ```
    其中，<#Think>后面要添加思考过程；如果有<#Reflection>，则表明图像没有完全按照编辑指令生成，需要做进一步修改，此时<#Reflection>后面接上“二次编辑指令”；如果是<#Failed>，那么说明无法基于当前的编辑结果进一步修改，得到符合编辑指令的生成结果；如果是<#Success>说明编辑成功。
    现在，我希望你能够帮助按照上述的格式对图像和编辑指令进行检查，检查标准如下：
    1. 严格检查图像中是否存在明显的崩坏、伪影，不存在的话需要将上述描述中的崩坏去掉；尤其注意，手部的崩坏和伪影如果不是非常严重，请直接将其去掉。
    2. 如果背景变化的语义基本正确、没有出现明显的扭曲，修正后中不能出现对应的描述
    3. 需要尽可能保证图像中未被编辑指令提及的区域不发生变化，如果第二张图发生了一些预期外的改动，导致有一些第一张图象中本来不希望发生改动的区域发生了巨大的变化，在这种情况下已经不可能基于第二张图进行修改，那么需要给一个<#Failed>标签，说明无法生成符合标准的编辑指令。
    4. 请确保<#Reflection>得到的反思指令和原始的编辑指令存在显著差异。如果reflection和原始的编辑指令基本相同，说明模型根本不具备编辑的能力，此类例子需要被剔除。
    5. 你需要确保返回的结果里，反思内容和可能存在的新编辑指令都是正确的。
    6. <#Reflection>和<#Failed>标签只能出现一次，且必须在<#Think>标签之后。
    7. <#Reflection>后面的新编辑指令中，需要确保去掉了当前编辑结果中已经成功编辑的内容，减少和原始指令的重复内容。
    这里再次强调，<#Reflection>后面的编辑指令是基于第二张图进行编辑的指令，你必须假定这个编辑指令无法利用第一张图象里的信息。
    接下来
    1. 如果你认为可以基于当前结果编辑得到符合“编辑指令”的数据，请你返回且仅返回预期的反思内容和“二次编辑指令”，符合格式要求，
    格式形如：<#Think>...<#Reflection>...
    不要返回格式之外的内容。
    2. 如何你认为无法生成符合标准的数据，请你返回且仅返回预期的反思内容以及<#Failed>标签，
    格式形如：<#Think>...<#Failed>
    不要返回格式之外的内容。
    3. 如何你认为已经编辑成功，
    格式形如：<#Think>...<#Success>
    不要返回格式之外的内容。
    """  # noqa: E501
        sys = "你是严格的图像编辑结果审查器，必须严格遵守用户给定的输出格式要求。"
        prior_text = (
            new_query_template.format(instruction_cn) + "\n【历史先验（来自上一轮独立评分供参考，非强制结论）】\n"
            f"- 评分: {prior_score}\n"
            f"- 理由: {prior_reason}\n"
            "请充分参考先验，但仍以两张图与编辑指令的真实内容为准，给出严格且可执行的结论与编辑建议。"
        )
        return [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": src_img_data_url}},
                    {"type": "image_url", "image_url": {"url": tgt_img_data_url}},
                    {"type": "text", "text": prior_text},
                ],
            },
        ]

    def _validate_reflection_response(self, text: str) -> tuple[bool, str]:
        if not isinstance(text, str):
            return False, "错误: 返回非字符串"

        think_idx = text.find("<#Think>")
        if think_idx < 0:
            return False, "错误: 缺少 <#Think>"

        tags = ["<#Reflection>", "<#Failed>", "<#Success>", "<#Others>"]
        present_tags = [tag for tag in tags if tag in text]

        if len(present_tags) != 1:
            return False, f"错误: 需要且仅需要 {tags} 之一, 但找到了 {len(present_tags)} 个。"

        tag = present_tags[0]
        if text.count(tag) > 1:
            return False, f"错误: 标签 {tag} 出现次数 > 1"

        second_idx = text.find(tag)
        if second_idx < think_idx:
            return False, "错误: 反思/失败/成功标签必须出现在 <#Think> 之后"

        cleaned = text[think_idx:].strip()
        return True, cleaned


class Step1XEditThinker:
    def __init__(self, model, processor):
        self.reflector = ImageEditEvaluator(model=model, processor=processor)

    def reflect(
        self,
        source_image: Image.Image,
        result_image: Image.Image | None,
        instruction: str,
    ):
        res = self.reflector.run_full_pipeline(
            source_image=source_image, edited_image=result_image, original_instruction=instruction
        )
        return res["reflection"]

    def think(self, source_image: Image.Image, instruction: str):
        res = self.reflector.prompt_reformat(source_image=source_image, original_instruction=instruction)
        return res

    def format_text(self, text):
        reflection_prompt = None
        success = False
        if not validate_string(text):
            return success, reflection_prompt
        if "<#Reflection>" in text:
            reflection_prompt = text.split("<#Reflection>")[1]
        elif "<#Success>" in text or "<#Others>" in text:
            success = True
        return success, reflection_prompt

__all__ = ["Step1XEditThinker"]
