# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import gc
import random
import warnings
from typing import List

import numpy as np
import PIL
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from transformers import AutoConfig, LogitsProcessor, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration

import rv_train.constants as C
from rv_train.utils.train_utils import ForkedPdb as debug  # noqa: F401


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """
    Logits processor that constrains generation to only numbers (0-9), spaces, and end-of-text tokens.
    强制 Qwen 输出 “空格分隔的整数序列”，方便后面解码为动作
    典型的「把 LLM 输出当成 structured action 的技巧」
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Get token IDs for allowed tokens
        self.allowed_tokens = set()

        # Add numbers 0-9
        for i in range(10):
            # 开头、结尾等特殊符号不要；返回的是列表，取第一个
            token_id = tokenizer.encode(str(i), add_special_tokens=False)[0]
            self.allowed_tokens.add(token_id)
        # Add space token
        space_token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
        self.allowed_tokens.add(space_token_id)

        # Add end of text token
        # 没有 EOS token，模型会一直生成直到达到最大长度
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        # Set logits to negative infinity for all tokens except allowed ones
        # input_ids: 当前已生成的token序列 [batch_size, sequence_length]
        # scores: 模型对下一个token的预测分数 [batch_size, vocab_size]
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


def format_data(system_message, image, instr, action_txt):
    """
    Convert the data into the format required by the model
    :param system_message: str, the system message
    :param image: list of PIL images, the image to be processed
    :param instr: str, the instruction
    :param action_txt: str, the action text
    :return: list of dicts, the format required by the model
    system: 系统角色设定
    user: 用户输入，包含：
        图像（一张或多张）
        文本指令
    assistant: 助手回复（通常是动作指令）
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": _image} for _image in image]
            + [{"type": "text", "text": instr}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": action_txt}],
        },
    ]


class QwenActor(nn.Module):
    def __init__(
        self,
        qwen_model_id,
        action_type,
        original_action_dim,
        horizon,
        history,
        use_lora=True,
        use_qlora=True,
        num_cam=1,
        lora_config="",
        lora_rank=8,
        rgb_input=False,
        rgb_img_size=(84, 84),
        add_vision_id=False,
        tiled_rgb_imgs=False,
        num_bins_actions=1000,
        use_flash_attention_2=False,
        system_message_version=1,
        action_mask_aug=0,
        action_mask_aug_per=0.1,
        attention_dropout=0.0,
    ):
        """
        :param qwen_model_id: str, the id of the qwen model to use
        :param action_type: str, the type of action to use, either ORIGINAL or EE
        :param original_action_dim: int, the dimension of the original action
        :param horizon: int, the horizon of the action
        :param history: int, the history of the action
        :param use_qlora: bool, whether to use qlora for parameter efficient fine-tuning
        :param num_cam: int, the number of cameras for rgb input
        :param lora_config: str, the lora configuration to use, either empty string or "default"
        :param lora_rank: int, the rank of the lora to use, only used if lora_config is "default"
        :param rgb_input: bool, whether to use rgb image input
        :param rgb_img_size: tuple, the size of the rgb image input (height, width)
        :param add_vision_id: bool, whether to add vision id to the input for qwen2.5
        :param tiled_rgb_imgs: bool, whether to tile the rgb images into a single image instead feeding them separately
        :param num_bins_actions: int, the number of bins in which each action dimension is discretized
        :param use_flash_attention_2: bool, whether to use flash attention 2 for faster training and inference
        :param attention_dropout: float, the dropout rate for the attention layer in the qwen model. Only tested when use_lora is False.
        """
        super(QwenActor, self).__init__()

        # current assumptions
        if use_qlora:
            assert use_lora, "use_lora must be True if use_qlora is True"
        if attention_dropout > 0.0:
            assert (
                not use_lora
            ), "attention_dropout is only supported when use_lora is False"
        assert lora_config in ["", "default"]
        if history > 1 or num_cam > 1:
            assert (
                add_vision_id
            ), "add_vision_id must be True if history > 1 or num_cam > 1"

        # assert not use_flash_attention_2, "use_flash_attention_2 is not supported yet, it requires tokenizer.pad=left which we have not fully implemented/understood"

        # for Qwen model, we need to load the parameters before DDP
        # in case we want to load the model from a checkpoint
        self.load_param_before_ddp = True

        self.qwen_model_id = qwen_model_id
        self.action_type = action_type
        self.original_action_dim = original_action_dim
        self.horizon = horizon
        self.history = history
        self.use_lora = use_lora
        self.use_qlora = use_qlora
        self.num_cam = num_cam
        self.lora_config = lora_config
        self.lora_rank = lora_rank
        self.rgb_input = rgb_input
        self.rgb_img_size = rgb_img_size
        self.add_vision_id = add_vision_id
        self.tiled_rgb_imgs = tiled_rgb_imgs
        self.num_bins_actions = num_bins_actions
        self.use_flash_attention_2 = use_flash_attention_2
        self.action_mask_aug_per = action_mask_aug_per
        self.attention_dropout = attention_dropout

        self.model = self.load_qwen_model(
            qwen_model_id=self.qwen_model_id,
            use_lora=self.use_lora,
            use_qlora=self.use_qlora,
            lora_config=self.lora_config,
            lora_rank=self.lora_rank,
            use_flash_attention_2=self.use_flash_attention_2,
            attention_dropout=self.attention_dropout,
        )

        # Enable gradient checkpointing if requested
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.min_pixel = self.max_pixel = self.rgb_img_size[0] * self.rgb_img_size[1]
        if self.rgb_input and self.tiled_rgb_imgs:
            self.min_pixel *= self.history * self.num_cam
            self.max_pixel *= self.history * self.num_cam

        self.processor = self.load_qwen_model_processor(
            qwen_model_id=qwen_model_id,
            min_pixel=self.min_pixel,
            max_pixel=self.max_pixel,
            padding_side="left" if use_flash_attention_2 else None,
        )
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        print(
            "WARNING: Using hardcoded dataset stats for DP3. This should be replaced with loading from a file."
        )

        if action_type == C.ORIGINAL:
            self.act_dim = original_action_dim
        elif action_type == C.EE:
            self.act_dim = 7
        else:
            assert False

        self.system_message = f"Analyze the input image and predict robot actions for the next {self.horizon} timesteps. Each action has {self.act_dim} dimensions. Output a single sequence of {self.horizon * self.act_dim} integers (0-{self.num_bins_actions} each), representing the {self.horizon} timesteps sequentially. Provide only space separated numbers. Nothing else."

        # todo: need better way to determine it
        self.num_tokens = 1024
        self.original_dataset_stats = None  # original dataset stats has the same format as the one provided by the dataset
        self.dataset_stats = (
            None  # dataset stats is in the format specific to the model
        )

        self._sysuser_len = None

        self.cache_sysuser_len = False

    def set_dataset_stats(self, dataset_stats):
        """
        Set the dataset stats for the model
        :param dataset_stats: dict, the dataset stats
        """
        if dataset_stats == {}:
            warnings.warn(
                "Dataset stats is empty likely because the system does not have the data used to compute the stats. Ignore this is you are loading a pretrained model."
            )
            return

        self.original_dataset_stats = dataset_stats
        if self.action_type == C.ORIGINAL:
            self.dataset_stats = dataset_stats["out_ori_act"]
        else:
            raise NotImplementedError(f"Action type {self.action_type} not implemented")

    @staticmethod
    def load_qwen_model(
        qwen_model_id,  # 模型ID或路径，如 "Qwen/Qwen2.5-VL-7B-Instruct"
        use_lora,
        use_qlora,
        lora_config,
        lora_rank,
        use_flash_attention_2,
        attention_dropout,  # # 注意力dropout率
    ):
        if lora_config == "":
            lora_config = None
        elif lora_config == "default":
            if use_lora:
                """
                原始前向传播: h = Wx
                LoRA前向传播: h = Wx + BAx

                Rank
                # 控制适配器的表达能力
                r=8   # 较小秩，参数少，可能欠拟合
                r=16  # 中等秩，平衡效果和效率
                r=64  # 较大秩，参数多，可能过拟合

                lora_alpha
                # 控制LoRA输出的缩放程度
                实际输出 = (BAx) * (alpha / r)
                类似于学习率，控制适配器对原始输出的影响程度;通常设置为 r 的 1-2 倍;alpha/r 比值越大，LoRA影响越强

                target
                # 指定哪些层应用LoRA
                target_modules=["q_proj", "v_proj"]  # 只对query和value投影应用
                # 或者
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 所有注意力投影
                # 或者
                target_modules="all-linear"  # 所有线性层

                type
                掩码和损失计算不同
                task_type="CAUSAL_LM"        # 因果语言建模 (文本生成)
                task_type="SEQ_2_SEQ_LM"     # 序列到序列任务
                task_type="TOKEN_CLS"        # 令牌分类
                task_type="QUESTION_ANS"     # 问答任务


                """
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    lora_alpha=16,  # LoRA缩放因子
                    lora_dropout=0.05,  # LoRA层的dropout
                    r=lora_rank,  # LoRA秩，控制参数量
                    bias="none",  # 不训练任何偏置 (最常用);最节省参数，效果通常足够好
                    target_modules=["q_proj", "v_proj"],  # 目标模块
                    task_type="CAUSAL_LM",  # 因果语言建模任务
                )
        else:
            raise ValueError(f"Invalid lora_config: {lora_config}")

        # Qwen Init BitsAndBytes Config（比特与字节配置）
        bnb_config = None
        if use_lora and use_qlora:
            # QLoRA = 量化（Quantization） + LoRA
            from transformers import BitsAndBytesConfig

            """
                # NF4 (Normal Float 4)
                # - 专门为正态分布数据优化
                # - 在重要区域（接近0）有更高精度
                # - 适合神经网络权重
                # FP4 (Float Point 4)
                # - 均匀精度分布
                # - 适合通用数据
                # 实际效果：NF4在相同bit数下精度损失更小

                # 第一次量化：原始权重 → 4bit值 + 缩放因子
                original_weights = [0.123, 0.456, 0.789, 0.234, ...]
                quantized_values = [1, 3, 5, 2, ...]        # 4bit值
                scale_factors = [0.1, 0.2, ...]             # 缩放因子
                # 第二次量化：对缩放因子本身再次量化
                # scale_factors → 进一步压缩
                # 效果：额外节省约0.4-0.5bit/参数
                缩放因子 = (35 - (-25)) / 15 = 60 / 15 = 4
                量化过程：
                北京：-10°C → (-10 - (-25)) / 4 = 15/4 ≈ 4（刻度值）
                广州：25°C  → (25 - (-25)) / 4 = 50/4 ≈ 13（刻度值）
                哈尔滨：-25°C → (-25 - (-25)) / 4 = 0（刻度值）
                三亚：35°C  → (35 - (-25)) / 4 = 60/4 = 15（刻度值）
                存储：
                刻度值：[4, 13, 0, 15]（4bit）
                缩放因子：4（保存这个就能还原）
                最小值：-25（保存这个确定起点）

                # bfloat16优势：
                # - 保持与float32相同的指数范围（8bit指数）
                # - 减少尾数精度（7bit尾数）
                # - 训练稳定性更好，减少梯度消失/爆炸
            """
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 核心：4bit量化 将模型权重从16bit/32bit压缩到4bit 内存减少: 75.0%（16bit）
                bnb_4bit_use_double_quant=True,  # 双量化，进一步压缩
                bnb_4bit_quant_type="nf4",  # 量化数据类型
                bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度 存储与计算是不同的
            )

        extra_kwargs = {}
        if use_flash_attention_2:
            """
            # 传统注意力 vs Flash Attention 2
            传统注意力: O(N²) 复杂度，速度慢
            Flash Attention 2: 优化到接近 O(N)，速度快2-4倍
            # 内存占用对比（序列长度=4096）
            传统注意力: ~16GB
            Flash Attention 2: ~4GB  # 减少75%内存

            核心：分块计算将大矩阵分成小块处理
            """
            extra_kwargs["attn_implementation"] = "flash_attention_2"

        if attention_dropout > 0.0:
            """
            # 预训练时通常用很小的dropout
            attention_dropout = 0.0  # 或 0.1
            # 原因：预训练数据量大，过拟合风险小
            # 微调时数据量小，需要更强正则化
            attention_dropout = 0.1  # 或 0.2
            # 原因：防止在小数据集上过拟合
            """
            config = AutoConfig.from_pretrained(qwen_model_id)
            config.attention_dropout = attention_dropout
            extra_kwargs["config"] = config

        """
        # 这三种写法效果相同 对于大多数情况 不指定 device_map，让库自动处理 需要精确控制时再写
        device_map={"": "cuda:0"}
        device_map="cuda:0"  # 简化写法
        device_map=0         # 更简化的写法
        """
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_id,
            # device_map={"": "cuda:0"},  # Use the explicit map
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            **extra_kwargs,
        )

        if use_lora and (lora_config is not None):
            model = get_peft_model(model, lora_config)

        return model

    @staticmethod
    def load_qwen_model_processor(
        qwen_model_id,  # 模型ID，如 "Qwen/Qwen2.5-VL-7B-Instruct"
        min_pixel,  # 图像最小像素值（预处理用） 原始像素值 → 缩放到 [min_pixel, max_pixel] 范围
        max_pixel,  # 图像最大像素值（预处理用） 原始像素值 → 缩放到 [min_pixel, max_pixel] 范围
        padding_side,  # 文本填充方向
    ):
        """
        Tokenizer（分词器）
            仅处理文本：文本 → tokens
            功能单一：只负责文本分词
        Processor（处理器）:内部复杂组成
            多模态处理：文本 + 图像 → 统一格式
            功能全面：协调分词器和图像处理器

        填充PAD
        文本生成	"right"	保持上下文完整性，便于自回归生成
        文本分类	"left"	序列结尾对齐，便于分类特征提取
        问答系统	"right"	问题信息在前，模型基于此生成答案
        翻译任务	编码器"right"
        解码器"left"	分别优化编码和解码过程
        """
        if padding_side is not None:
            processor = Qwen2_5_VLProcessor.from_pretrained(
                qwen_model_id,
                min_pixels=min_pixel,
                max_pixels=max_pixel,
                padding_side=padding_side,
            )
        else:
            processor = Qwen2_5_VLProcessor.from_pretrained(
                qwen_model_id,
                min_pixels=min_pixel,
                max_pixels=max_pixel,
            )

        return processor

    def get_min_max_act(self, instruction):
        """
        Get the min and max action for the instruction.
        :param instruction: str, the instruction for the current episode. This is needed for libero bounds 99% as the action space is different for different instructions.
        :return: torch.Tensor, the min and max action.
        """
        assert instruction is not None, "instruction is needed for libero bounds 99%"
        min_act = []
        max_act = []
        for _instruction in instruction:
            _suite = self.instruction_to_suite[_instruction]
            min_act.append(torch.tensor(self.dataset_stats[_suite]["min"]))
            max_act.append(torch.tensor(self.dataset_stats[_suite]["max"]))
        min_act = torch.stack(min_act, dim=0)
        max_act = torch.stack(max_act, dim=0)
        return min_act, max_act

    def get_text_action(self, actions, instruction=None):
        """
        Get the text action from the actions.
        :param actions: torch.Tensor, the actions to convert to text.
        :param instruction: str, the instruction for the current episode. This is needed for libero bounds 99% as the action space is different for different instructions.
        :return: List[str], the text action.
        """
        # TODO: implement this for ee action space
        if (
            (not hasattr(self, "_min_act"))
            or (not hasattr(self, "_max_act"))
            or (self._min_act.device != actions.device)
            or (self._max_act.device != actions.device)
        ):
            # safer to recompute the min and max action for each device
            min_act = torch.tensor(self.dataset_stats["min"], device=actions.device)
            max_act = torch.tensor(self.dataset_stats["max"], device=actions.device)
            self._min_act = min_act
            self._max_act = max_act
        else:
            # use the cached min and max action
            min_act = self._min_act
            max_act = self._max_act

        assert torch.all(min_act <= actions) and torch.all(
            actions <= max_act
        ), f"Action is out of range: {actions}"

        actions = (actions - min_act) / (max_act - min_act)
        actions *= self.num_bins_actions
        actions = torch.round(actions).long()
        actions = actions.reshape(actions.shape[0], -1)
        action_txt = [" ".join(map(str, x.tolist())) for x in actions]

        return action_txt

    def get_action_from_text_action(self, action_txt, instruction=None):
        """
        Get the action from the text action.
        :param action_txt: List[str], the text action.
        :param instruction: str, the instruction for the current episode. This is needed for libero bounds 99% as the action space is different for different instructions.
        """
        # TODO: implement this for ee action space
        bs = len(action_txt)
        min_act = torch.tensor(self.dataset_stats["min"])
        max_act = torch.tensor(self.dataset_stats["max"])

        try:
            """ "
            Note: The action_txt is a list of strings. action_txt[i] is the action text for the i-th sample.
            action_txt[i] is a string that contains horizon * act_dim numbers in space separated format.
            We have built in some flexbility for handling minor mistakes in the action_txt.
            """
            # remove space from the front and back of the action_txt if they exist
            action_txt = [x.strip() for x in action_txt]
            action = [[x for x in _action_txt.split(" ")] for _action_txt in action_txt]
            action = torch.tensor(
                [[int(x) for x in _action_txt] for _action_txt in action],
                dtype=torch.float32,
            )
            # This handles tha case when bs == 1 and the action_txt is not divisible by act_dim
            # We remove some elements so that it is divisible by act_dim
            if bs == 1 and len(action[0]) % self.act_dim != 0:
                action = action[0][: len(action[0]) - len(action[0]) % self.act_dim][
                    None, :
                ]

            # reshape to (bs, -1, act_dim)
            # takes care of case when the action_txt has less than horizon * act_dim numbers
            action = action.reshape(bs, -1, self.act_dim)
            # if action.shape[1] < self.horizon, pad the action with the last action
            if action.shape[1] < self.horizon:
                action = torch.cat(
                    [
                        action,
                        action[:, -1:].repeat(1, self.horizon - action.shape[1], 1),
                    ],
                    dim=1,
                )
            if action.shape[1] > self.horizon:
                action = action[:, : self.horizon]
            action = ((action / self.num_bins_actions) * (max_act - min_act)) + min_act
        except Exception as e:
            print(f"Error in parsing action text: {e}")
            print(action_txt)
            action = ((min_act + max_act) / 2).repeat(bs, self.horizon, 1)

        return action

    def check_inputs(
        self,
        pc,
        rgb_pc,
        instr,
        rgb,
        ori_act,
        ee_pos,
        ee_rot,
        ee_gri,
        out_ee_pos,
        out_ee_rot,
        out_ee_gri,
        out_ori_act,
        get_loss,
        get_action,
        get_one_step_action,
        last_action_txt,
    ):
        assert (
            self.dataset_stats is not None
        ), "dataset_stats must be set before calling forward"
        assert isinstance(instr, list)

        assert not (get_loss and get_action)
        assert get_loss or get_action
        if get_one_step_action:
            assert get_action
            assert isinstance(last_action_txt, str)
            assert len(instr) == 1, "one_step_action is only supported for batch size 1"
        if self.rgb_input:
            assert rgb is not None
            assert rgb.shape[1:] == (
                self.history,
                self.num_cam,
                *self.rgb_img_size,
                3,
            ), f"rgb.shape: {rgb.shape}, self.history: {self.history}, self.num_cam: {self.num_cam}, self.rgb_img_size: {self.rgb_img_size}"
            # some room for numerical errors
            # 0, 2, 255 are valid values
            assert (rgb.min() >= -1e-2) and (
                1.99 <= rgb.max() <= 255.01
            ), f"rgb.min(): {rgb.min()}, rgb.max(): {rgb.max()}"
        else:
            assert rgb is None

        assert pc is None
        assert rgb_pc is None

    def get_imgs(
        self,
        bs,
        pc,
        rgb_pc,
        rgb,
    ):
        """
        Get the images for the given inputs
        :param bs: int, the batch size
        :param pc: torch.Tensor, the point cloud
        :param rgb_pc: torch.Tensor, the rgb point cloud
        :param rgb: torch.Tensor, the rgb image
        :return: list of list of PIL images
        """
        imgs = [[] for _ in range(bs)]

        if self.rgb_input:
            for i, _rgb in enumerate(rgb):
                _imgs = []
                for j in range(self.history):
                    for k in range(self.num_cam):
                        _imgs.append(_rgb[j][k])
                if self.tiled_rgb_imgs:
                    _imgs = [self.tile_images(_imgs)]
                _imgs = [
                    Image.fromarray(x.cpu().numpy().astype(np.uint8)) for x in _imgs
                ]
                imgs[i].extend(_imgs)

        return imgs

    def get_qwen_inputs(
        self,
        bs: int,
        imgs: List[List[PIL.Image.Image]],
        instr: List[str],
        action_txt: List[str],
        drop_assistant: bool = False,
        add_generation_prompt: bool = False,
    ):
        """
        Get the Qwen inputs for the given inputs
        :param bs: int, the batch size
        :param imgs: list of list of PIL images
        :param instr: list of strings
        :param action_txt: list of strings
        :param drop_assistant: bool, whether to drop the assistant portion.
        :param add_generation_prompt: bool, whether to add the generation prompt assistant\n.
        """

        examples = [
            format_data(
                system_message=self.system_message,
                image=imgs[i],
                instr=instr[i],
                action_txt=action_txt[i],
            )
            for i in range(bs)
        ]
        if drop_assistant:
            # drop the assistant portion so the model must generate it
            examples = [e[:2] for e in examples]

        texts = [
            self.processor.apply_chat_template(
                example,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                add_vision_id=self.add_vision_id,
            )
            # when add_generation_prompt is True, it will add the prompt
            # `assistant\n` to the end of the input text
            for example in examples
        ]
        # [0] in process_vision_info is for image input, [1] is for video input
        image_inputs = [process_vision_info(example)[0] for example in examples]

        model_inputs = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )
        for key in model_inputs:
            model_inputs[key] = model_inputs[key].to(
                next(self.model.parameters()).device
            )
        return model_inputs, examples

    def forward(
        self,
        pc=None,
        rgb_pc=None,
        instr=None,
        rgb=None,
        ori_act=None,
        ee_pos=None,
        ee_rot=None,
        ee_gri=None,
        out_ee_pos=None,
        out_ee_rot=None,
        out_ee_gri=None,
        out_ori_act=None,
        get_loss=True,
        get_action=False,
        generate_temperature=0.1,
        get_one_step_action=False,
        last_action_txt="",
    ):
        """
        Forward pass for the Qwen model

        Parameters
        ----------
        pc : optional
            Point cloud data
        rgb_pc : optional
            RGB point cloud data
        instr : optional
            Instructions
        rgb : optional
            RGB images
        ori_act : optional
            Original actions
        ee_pos : optional
            End effector position
        ee_rot : optional
            End effector rotation
        ee_gri : optional
            End effector gripper state
        out_ee_pos : optional
            Output end effector position
        out_ee_rot : optional
            Output end effector rotation
        out_ee_gri : optional
            Output end effector gripper state
        out_ori_act : optional
            Output original actions
        get_loss : bool, default=True
            Whether to calculate and return loss
        get_action : bool, default=False
            Whether to get actions
        generate_temperature : float, default=0.1
            Temperature for generation
        get_one_step_action : bool, default=False
            Whether to run the model forward for only one step of action at a time. If True,
            we complete the last_action_txt to next action string sufficient to decode the
            action for one next step. If False, we complete the last_action_txt to next
            action string sufficient to decode the action for all the next steps.
        last_action_txt : str, default=""
            The last action text to complete to get the next action text. This is only
            used when get_one_step_action is True.

        Returns
        -------
        dict or tuple
            Model outputs (should specify exact return structure)

        Raises
        ------
        Exception
            Any exceptions that can be raised (if applicable)
        """
        self.check_inputs(
            pc=pc,
            rgb_pc=rgb_pc,
            instr=instr,
            rgb=rgb,
            ori_act=ori_act,
            ee_pos=ee_pos,
            ee_rot=ee_rot,
            ee_gri=ee_gri,
            out_ee_pos=out_ee_pos,
            out_ee_rot=out_ee_rot,
            out_ee_gri=out_ee_gri,
            out_ori_act=out_ori_act,
            get_loss=get_loss,
            get_action=get_action,
            get_one_step_action=get_one_step_action,
            last_action_txt=last_action_txt,
        )

        bs = len(instr)

        # imgs is list of list of PIL images
        imgs = self.get_imgs(
            bs=bs,
            pc=pc,
            rgb_pc=rgb_pc,
            rgb=rgb,
        )

        # TODO: implement this for ee action space
        if out_ori_act is None:
            assert not get_loss
            action_txt = [[]] * bs
        else:
            action_txt = self.get_text_action(out_ori_act, instruction=instr)

        model_inputs, examples = self.get_qwen_inputs(
            bs=bs,
            imgs=imgs,
            instr=instr,
            action_txt=action_txt,
            drop_assistant=get_action,  # when getting action, we drop the assistant portion
            add_generation_prompt=get_action,  # when getting action, we add the generation prompt assistant\n so that the model need not generate it
        )

        if get_loss:
            labels = model_inputs["input_ids"].clone()
            # mask system message and image token IDs in the labels
            for i, example in enumerate(examples):
                if (self._sysuser_len is None) or (not self.cache_sysuser_len):
                    sysuser_conv = example[:-1]
                    sysuser_text = self.processor.apply_chat_template(
                        sysuser_conv, tokenize=False, add_vision_id=self.add_vision_id
                    )
                    sysuser_img, _ = process_vision_info(sysuser_conv)

                    sysuser_inputs = self.processor(
                        text=[sysuser_text],
                        images=[sysuser_img],
                        return_tensors="pt",
                        padding=True,
                    )

                    sysuser_len = sysuser_inputs["input_ids"].shape[1]
                    sysuser_len += 3  # to mask out `assistant\n`
                    self._sysuser_len = sysuser_len
                else:
                    sysuser_len = self._sysuser_len
                # TIP: to decode the input use:
                # when padding is right: self.processor.decode(model_inputs["input_ids"][0][0:sysuser_len])
                # when padding is left: self.processor.decode(model_inputs["input_ids"][0][num_pad_tokens: num_pad_tokens + sysuser_len])
                if self.processor.tokenizer.padding_side == "right":
                    labels[i, :sysuser_len] = -100
                elif self.processor.tokenizer.padding_side == "left":
                    num_pad_tokens = sum(labels[i] == 151643).item()
                    labels[i, num_pad_tokens : num_pad_tokens + sysuser_len] = -100
                else:
                    raise ValueError(
                        f"Unknown padding side: {self.processor.tokenizer.padding_side}"
                    )

                assert (
                    not self.processor.tokenizer.padding_side == "left"
                ), "current implementation only supports right padding"
                # for debugging, compare
                # self.processor.decode(model_inputs["input_ids"][i][model_inputs["attention_mask"][i] == 1])
                # with self.processor.decode(model_inputs["input_ids"][i])
                _action_txt = action_txt[i]
                # 10% of sample has no augmentation
                if random.random() < 0.1:
                    _action_mask_aug_per = 0.0
                else:
                    _action_mask_aug_per = random.uniform(0.0, self.action_mask_aug_per)
                mask_len = int(len(_action_txt) * _action_mask_aug_per)
                mask_indices = random.sample(range(len(_action_txt)), mask_len)
                mask_indices = [
                    x + sysuser_len for x in mask_indices
                ]  # add sysuser_len to the mask indices to get the correct indices of these tokens
                labels[
                    i, mask_indices
                ] = -100  # these elements will not be used for loss calculation
                model_inputs["input_ids"][
                    i, mask_indices
                ] = 30  # replace the input ids with '?' token id

            labels[labels == 151643] = -100

            outputs = self.model(**model_inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            # copied from modeling_qwen2_5_vl.py to compute the loss
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)

            return {"loss": loss}

        if get_action:
            sample_args = {}
            if generate_temperature > 0:
                sample_args["temperature"] = generate_temperature
            else:
                sample_args[
                    "do_sample"
                ] = False  # greedy search, this makes the generation deterministic

            if get_one_step_action:
                # we calculate the max number of tokens to generate for one step of action
                # +1 is for the space token
                max_new_tokens = self.act_dim * (len(str(self.num_bins_actions)) + 1)
                if last_action_txt != "":
                    last_action_txt_ids = self.processor.tokenizer(
                        last_action_txt, return_tensors="pt"
                    )["input_ids"].to(model_inputs["input_ids"].device)
                    model_inputs["input_ids"] = torch.cat(
                        [model_inputs["input_ids"], last_action_txt_ids], dim=1
                    )
                    model_inputs["attention_mask"] = torch.cat(
                        [
                            model_inputs["attention_mask"],
                            torch.ones_like(last_action_txt_ids),
                        ],
                        dim=1,
                    )

            # token id to text mapping
            # 220 is space
            # 15 to 24 are 0 to 9
            # 151645 is the end of text token
            generated_ids = self.model.generate(
                **model_inputs,
                logits_processor=[self.logits_processor],
                max_new_tokens=(
                    max_new_tokens if get_one_step_action else self.num_tokens
                ),
                **sample_args,
            )

            input_ids = model_inputs["input_ids"]
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            generated_action_txt = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if get_one_step_action:
                # TODO: Only supports batch size 1
                generated_action_txt = [last_action_txt + generated_action_txt[0]]

            out_ori_act = self.get_action_from_text_action(
                generated_action_txt, instruction=instr
            )

            if get_one_step_action:
                out_ori_act = out_ori_act[:, -1:]

            return {
                "gt_action_text": action_txt,
                "pred_action_txt": generated_action_txt,
                "gt_out_ori_act": out_ori_act,
                "out_ori_act": out_ori_act,
            }

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def from_pretrained(self, path, is_trainable=True):
        # 常见的技巧 自动获取模型所在的设备
        # self.parameters() 返回一个生成器（generator），包含模型的所有可学习参数
        # next(self.parameters()) 获取第一个参数张量 可能是第一个卷积层的权重或第一个全连接层的权重
        _device = next(self.parameters()).device

        # 防止在加载新模型时出现内存不足的问题
        del self.model
        # 清空GPU缓存，避免内存碎片
        torch.cuda.empty_cache()
        # 强制垃圾回收，彻底释放内存
        gc.collect()

        # LoRA模型加载路径
        if self.use_lora:
            # PEFT (Parameter-Efficient Fine-Tuning) 库
            # 参数高效微调的 Hugging Face 库，主要目的是用极少的参数量来微调大语言模型
            from peft import PeftModel

            base_model = self.load_qwen_model(
                qwen_model_id=self.qwen_model_id,
                use_lora=self.use_lora,
                use_qlora=self.use_qlora,
                lora_config="",  # None regarless of lora config as lora is added later using PeftModel
                lora_rank=self.lora_rank,  # Doesn't matter here as lora_config is None
                use_flash_attention_2=self.use_flash_attention_2,
                attention_dropout=self.attention_dropout,
            )

            self.model = PeftModel.from_pretrained(
                base_model,
                path,
                is_trainable=is_trainable,
            )
            print("Loading Qwen2.5 PEFT model from", path)
        else:
            extra_kwargs = {}
            if self.use_flash_attention_2:
                extra_kwargs["attn_implementation"] = "flash_attention_2"
            if self.attention_dropout > 0.0:
                config = AutoConfig.from_pretrained(path)
                config.attention_dropout = self.attention_dropout
                extra_kwargs["config"] = config
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                # device_map={"": "cuda:0"},
                torch_dtype=torch.bfloat16,
                **extra_kwargs,
            )
            print("Loading Qwen2.5 full model from", path)

        if self.use_flash_attention_2:
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                path,
                min_pixels=self.min_pixel,
                max_pixels=self.max_pixel,
                padding_side="left",
            )
        else:
            self.processor = Qwen2_5_VLProcessor.from_pretrained(
                path,
                min_pixels=self.min_pixel,
                max_pixels=self.max_pixel,
            )

        print("Loading Qwen2.5 processor from", path)

        QwenActor.to(self, _device)

    def to(self, device):
        super().to(device)
        # if device is interger like 0 or "0", convert to cuda:0
        if isinstance(device, int) or (isinstance(device, str) and device.isnumeric()):
            device = f"cuda:{device}"
        if hasattr(self, "renderer"):
            self.renderer.renderer.device = device
            self.renderer.cameras.to(device)

    def tile_images(self, images):
        """
        Tile a images into a single image
        :param images: Tensor of shape (bs, H, W, 3) or list of tensors of shape (H, W, 3)
        :return: Tensor of shape (bs, H, W, 3)
        """
        for img in images:
            assert len(img.shape) == 3, f"img.shape: {img.shape}"
            assert img.shape[2] == 3, f"img.shape: {img.shape}"

        widths, heights = zip(*(im.shape[:-1] for im in images))
        total_width = sum(widths)
        max_height = max(heights)
        dst = torch.zeros((max_height, total_width, 3), device=images[0].device)
        current_x = 0
        for i, img in enumerate(images):
            dst[: img.shape[0], current_x : current_x + img.shape[1], :] = img
            current_x += img.shape[1]
        return dst
