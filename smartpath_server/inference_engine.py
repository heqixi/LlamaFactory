"""
Inference Engine
Handles model loading and inference
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .types import (
    SmartPathConfig,
    SmartNode,
    ActionPlan,
    InferenceRequest,
    InferenceResponse,
)
from .training_controller import TrainingController

logger = logging.getLogger("SmartPath.Inference")


class InferenceEngine:
    """Inference Engine"""

    def __init__(
        self,
        config: SmartPathConfig,
        training_controller: Optional[TrainingController] = None,
    ):
        self.config = config
        self.training_controller = training_controller
        
        self._model = None
        self._tokenizer = None
        self._current_model_id: Optional[str] = None
        self._adapter_path: Optional[str] = None

    async def initialize(self, model_id: Optional[str] = None) -> bool:
        """初始化推理引擎"""
        try:
            # 延迟导入，因为可能不需要推理功能
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # 确定要加载的模型
            base_model = self.config.default_base_model
            adapter_path = None

            if model_id and model_id != "base":
                # 加载特定的 LoRA adapter
                if self.training_controller:
                    models = self.training_controller.get_available_models()
                    for model in models:
                        if model["id"] == model_id and model.get("path"):
                            adapter_path = model["path"]
                            break

            elif self.config.default_inference_model:
                adapter_path = self.config.default_inference_model

            elif self.training_controller:
                # 使用最新的 adapter
                adapter_path = self.training_controller.get_latest_adapter_path()

            # 加载 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )

            # 加载模型
            if adapter_path:
                # 加载带 LoRA 的模型
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._model = PeftModel.from_pretrained(base, adapter_path)
                self._adapter_path = adapter_path
            else:
                # 加载基础模型
                self._model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

            self._current_model_id = model_id or "base"
            return True

        except ImportError as e:
            logger.warning(f"Inference requires transformers and torch: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            return False

    async def inference(self, request: InferenceRequest) -> InferenceResponse:
        """执行推理"""
        import time
        start_time = time.time()

        # 如果模型未加载，尝试初始化
        if self._model is None:
            success = await self.initialize(request.model_id)
            if not success:
                return InferenceResponse(
                    success=False,
                    actions=[],
                    raw_output="模型未加载且无法初始化",
                )

        try:
            # 构建提示词
            prompt = self._build_prompt(request.instruction, request.vdom)

            # 生成
            output = await self._generate(prompt)

            # 解析输出
            actions = self._parse_output(output)

            latency = (time.time() - start_time) * 1000  # ms

            return InferenceResponse(
                success=len(actions) > 0,
                actions=actions,
                raw_output=output,
                latency=latency,
                model_used=self._current_model_id,
            )

        except Exception as e:
            return InferenceResponse(
                success=False,
                actions=[],
                raw_output=str(e),
                latency=(time.time() - start_time) * 1000,
            )

    def _build_prompt(self, instruction: str, vdom: SmartNode) -> str:
        """构建推理提示词"""
        # 剥离 UID，生成紧凑的 VDOM 表示
        vdom_str = self._serialize_vdom(vdom)

        system_prompt = """你是一个路径寻址专家。你的任务是根据用户的自然语言指令，从给定的文档结构(VDOM)中找到最准确的目标路径，并生成相应的操作计划。

输出格式必须是一个 JSON 数组，每个元素包含:
- action: 操作类型 (setStyle, setText, remove, insert)
- targetPath: 目标节点的路径 (如 "root.children[0]")  
- params: 操作参数
- reason: 操作原因

请只输出 JSON，不要有其他内容。"""

        user_prompt = f"""当前文档结构:
{vdom_str}

用户指令: {instruction}

请生成操作计划:"""

        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _serialize_vdom(self, node: SmartNode, include_uid: bool = False) -> str:
        """序列化 VDOM 为紧凑的 JSON"""
        def to_dict(n: SmartNode) -> Dict[str, Any]:
            d: Dict[str, Any] = {
                "tag": n.tag,
                "path": n.path,
            }
            if n.content:
                d["content"] = n.content[:100]  # 截断长内容
            if n.semantic_features:
                d["semantic_features"] = n.semantic_features
            if n.children:
                d["children"] = [to_dict(c) for c in n.children]
            return d

        return json.dumps(to_dict(node), ensure_ascii=False, indent=2)

    async def _generate(self, prompt: str) -> str:
        """生成文本"""
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 解码输出
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        output = self._tokenizer.decode(generated, skip_special_tokens=True)

        return output.strip()

    def _parse_output(self, output: str) -> List[ActionPlan]:
        """解析模型输出"""
        actions = []

        try:
            # 尝试提取 JSON 数组
            json_match = re.search(r'\[[\s\S]*\]', output)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                for item in data:
                    action = ActionPlan(
                        action=item.get("action", ""),
                        targetPath=item.get("targetPath", ""),
                        params=item.get("params", {}),
                        reason=item.get("reason", ""),
                    )
                    actions.append(action)

        except json.JSONDecodeError:
            # 尝试其他解析方式
            pass

        return actions

    async def switch_model(self, model_id: str) -> bool:
        """切换模型"""
        if model_id == self._current_model_id:
            return True

        # 清理现有模型
        if self._model is not None:
            del self._model
            self._model = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 加载新模型
        return await self.initialize(model_id)

    def get_current_model(self) -> Optional[str]:
        """获取当前模型 ID"""
        return self._current_model_id

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

