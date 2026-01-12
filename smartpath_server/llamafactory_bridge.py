"""
Llama-Factory Deep Integration Bridge
Handles sample sync, training log monitoring, status push
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

import aiofiles

# Configure logger
logger = logging.getLogger("SmartPath.Bridge")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from .types import (
    SmartPathConfig,
    TrainingSample,
    TrainStatus,
    TrainingProgress,
    CloudState,
)


# Llama-Factory 常量 (与 llamafactory/extras/constants.py 保持一致)
DATA_CONFIG = "dataset_info.json"
TRAINER_LOG = "trainer_log.jsonl"
RUNNING_LOG = "running_log.txt"


@dataclass
class TrainingLogEntry:
    """训练日志条目"""
    current_steps: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    percentage: float = 0.0
    elapsed_time: str = ""
    remaining_time: str = ""
    samples_per_second: float = 0.0
    
    # 额外字段
    current_sample_instruction: str = ""
    current_sample_path: str = ""


@dataclass
class LlamaFactoryBridgeConfig:
    """桥接器配置"""
    # Llama-Factory 根目录
    llamafactory_root: str = "./LLaMA-Factory"
    # 数据目录
    data_dir: str = "data"
    # 输出目录前缀 (用于查找训练日志)
    output_prefix: str = "saves"
    # SmartPath 数据集名称前缀
    dataset_prefix: str = "smartpath"
    # 日志监控间隔 (秒)
    log_poll_interval: float = 1.0
    # 自动注册数据集
    auto_register: bool = True


class LlamaFactoryBridge:
    """
    Llama-Factory 桥接器
    
    功能:
    1. 样本同步到 data/ 目录
    2. 自动更新 dataset_info.json
    3. 监控训练日志获取实时进度
    4. 推送状态到 WebSocket
    """

    def __init__(
        self,
        config: LlamaFactoryBridgeConfig,
        on_progress: Optional[Callable[[TrainingProgress], None]] = None,
        on_log_entry: Optional[Callable[[TrainingLogEntry], None]] = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self.on_log_entry = on_log_entry
        
        # 路径
        self.lf_root = Path(config.llamafactory_root).resolve()
        self.data_dir = self.lf_root / config.data_dir
        self.dataset_info_path = self.data_dir / DATA_CONFIG
        
        # 状态
        self._samples: Dict[str, List[TrainingSample]] = {}  # dataset_name -> samples
        self._current_output_dir: Optional[Path] = None
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_log_position = 0
        self._last_log_mtime = 0.0
        
        # 当前训练进度
        self._progress = TrainingProgress()

    async def initialize(self) -> bool:
        """初始化桥接器"""
        # 检查 Llama-Factory 目录
        if not self.lf_root.exists():
            logger.warning(f"LLaMA-Factory directory not found: {self.lf_root}")
            return False
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return False
        
        # 加载现有数据集信息
        await self._load_existing_datasets()
        
        logger.info(f"Initialized, LLaMA-Factory path: {self.lf_root}")
        return True

    async def _load_existing_datasets(self) -> None:
        """加载现有的 SmartPath 数据集"""
        if not self.dataset_info_path.exists():
            return
        
        try:
            async with aiofiles.open(self.dataset_info_path, "r", encoding="utf-8") as f:
                content = await f.read()
                dataset_info = json.loads(content)
            
            # 查找 SmartPath 数据集
            for name, info in dataset_info.items():
                if name.startswith(self.config.dataset_prefix):
                    file_name = info.get("file_name", "")
                    file_path = self.data_dir / file_name
                    if file_path.exists():
                        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                            samples_data = json.loads(await f.read())
                            self._samples[name] = [
                                TrainingSample(**s) for s in samples_data
                            ]
                        logger.info(f"Loaded dataset '{name}': {len(self._samples[name])} samples")
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")

    # ============== 样本管理 ==============

    async def sync_samples(
        self,
        samples: List[TrainingSample],
        dataset_name: Optional[str] = None,
        append: bool = True,
    ) -> tuple[bool, str]:
        """
        同步样本到 Llama-Factory
        
        Args:
            samples: 训练样本列表
            dataset_name: 数据集名称 (默认自动生成)
            append: 是否追加到现有数据集
            
        Returns:
            (success, dataset_name)
        """
        if not dataset_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"{self.config.dataset_prefix}_{timestamp}"
        
        # 准备样本数据
        if append and dataset_name in self._samples:
            all_samples = self._samples[dataset_name] + samples
        else:
            all_samples = samples
        
        # 转换为 Alpaca 格式
        alpaca_data = []
        for sample in all_samples:
            alpaca_data.append({
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
                "history": sample.history,
            })
        
        # 写入数据文件
        data_file = self.data_dir / f"{dataset_name}.json"
        try:
            async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(alpaca_data, ensure_ascii=False, indent=2))
            
            # 更新内存缓存
            self._samples[dataset_name] = all_samples
            
            # 注册数据集
            if self.config.auto_register:
                await self._register_dataset(dataset_name, f"{dataset_name}.json")
            
            logger.info(f"Synced {len(samples)} samples to '{dataset_name}', total {len(all_samples)}")
            return True, dataset_name
            
        except Exception as e:
            logger.error(f"Failed to sync samples: {e}")
            return False, ""

    async def add_sample(
        self,
        sample: TrainingSample,
        dataset_name: str,
    ) -> bool:
        """添加单个样本到指定数据集"""
        return (await self.sync_samples([sample], dataset_name, append=True))[0]

    async def remove_sample(
        self,
        sample_id: str,
        dataset_name: str,
    ) -> bool:
        """从数据集中删除样本"""
        if dataset_name not in self._samples:
            return False
        
        original_count = len(self._samples[dataset_name])
        self._samples[dataset_name] = [
            s for s in self._samples[dataset_name] if s.id != sample_id
        ]
        
        if len(self._samples[dataset_name]) < original_count:
            # 重新写入文件
            success, _ = await self.sync_samples(
                self._samples[dataset_name],
                dataset_name,
                append=False
            )
            return success
        
        return False

    async def _register_dataset(self, dataset_name: str, file_name: str) -> bool:
        """在 dataset_info.json 中注册数据集"""
        try:
            # 读取现有配置
            if self.dataset_info_path.exists():
                async with aiofiles.open(self.dataset_info_path, "r", encoding="utf-8") as f:
                    dataset_info = json.loads(await f.read())
            else:
                dataset_info = {}
            
            # 添加/更新数据集
            dataset_info[dataset_name] = {
                "file_name": file_name,
            }
            
            # 写回
            async with aiofiles.open(self.dataset_info_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(dataset_info, ensure_ascii=False, indent=2))
            
            logger.info(f"Registered dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register dataset: {e}")
            return False

    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有 SmartPath 数据集"""
        datasets = []
        for name, samples in self._samples.items():
            datasets.append({
                "name": name,
                "sample_count": len(samples),
                "file_path": str(self.data_dir / f"{name}.json"),
            })
        return datasets

    def get_dataset_samples(self, dataset_name: str) -> List[TrainingSample]:
        """获取指定数据集的样本"""
        return self._samples.get(dataset_name, [])

    # ============== 训练监控 ==============

    async def start_monitoring(self, output_dir: Optional[str] = None) -> None:
        """开始监控训练日志"""
        if self._monitoring:
            return
        
        # 确定输出目录
        if output_dir:
            self._current_output_dir = Path(output_dir)
        else:
            # 自动查找最新的输出目录
            self._current_output_dir = await self._find_latest_output_dir()
        
        if not self._current_output_dir:
            logger.warning("Training output directory not found")
            return
        
        self._monitoring = True
        self._last_log_position = 0
        self._last_log_mtime = 0.0
        
        # 启动监控任务
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Started monitoring training logs: {self._current_output_dir}")

    async def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Stopped monitoring")

    async def _find_latest_output_dir(self) -> Optional[Path]:
        """查找最新的训练输出目录"""
        saves_dir = self.lf_root / self.config.output_prefix
        if not saves_dir.exists():
            return None
        
        latest_dir = None
        latest_mtime = 0.0
        
        # 递归查找包含 trainer_log.jsonl 或 running_log.txt 的目录
        for root, dirs, files in os.walk(saves_dir):
            # 优先找 trainer_log.jsonl
            if TRAINER_LOG in files:
                log_path = Path(root) / TRAINER_LOG
                mtime = log_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_dir = Path(root)
            # 如果没有 trainer_log，也接受 running_log
            elif RUNNING_LOG in files and latest_dir is None:
                log_path = Path(root) / RUNNING_LOG
                mtime = log_path.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_dir = Path(root)
        
        return latest_dir

    async def _monitor_loop(self) -> None:
        """监控循环"""
        while self._monitoring:
            try:
                await self._check_trainer_log()
                await asyncio.sleep(self.config.log_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.log_poll_interval * 2)

    async def _check_trainer_log(self) -> None:
        """检查训练日志更新"""
        if not self._current_output_dir:
            # 尝试重新查找输出目录
            self._current_output_dir = await self._find_latest_output_dir()
            if not self._current_output_dir:
                return
            logger.info(f"Found training directory: {self._current_output_dir}")
        
        # 先检查 trainer_log.jsonl
        trainer_log_path = self._current_output_dir / TRAINER_LOG
        if trainer_log_path.exists():
            await self._read_trainer_log(trainer_log_path)
        else:
            # 如果没有 trainer_log，检查 running_log 来确定状态
            running_log_path = self._current_output_dir / RUNNING_LOG
            if running_log_path.exists():
                await self._check_running_log(running_log_path)

    async def _read_trainer_log(self, log_path: Path) -> None:
        """读取 trainer_log.jsonl"""
        # 检查文件是否更新
        mtime = log_path.stat().st_mtime
        if mtime <= self._last_log_mtime:
            return
        
        self._last_log_mtime = mtime
        
        # 读取新日志
        try:
            async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
                await f.seek(self._last_log_position)
                new_content = await f.read()
                self._last_log_position = await f.tell()
            
            # 解析新日志行
            for line in new_content.strip().split('\n'):
                if line:
                    await self._parse_log_entry(line)
                    
        except Exception as e:
            logger.error(f"Failed to read log: {e}")

    async def _check_running_log(self, log_path: Path) -> None:
        """检查 running_log.txt 来确定训练状态"""
        try:
            async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
                content = await f.read()
            
            # 从 running_log 解析状态
            if "Running training" in content:
                # 训练正在进行
                self._progress.status = TrainStatus.TRAINING
                
                # 尝试解析样本数等信息
                import re
                num_examples = re.search(r"Num examples\s*=\s*(\d+)", content)
                num_epochs = re.search(r"Num Epochs\s*=\s*(\d+)", content)
                total_steps = re.search(r"Total optimization steps\s*=\s*(\d+)", content)
                
                if num_examples:
                    self._progress.message = f"训练中... 样本数: {num_examples.group(1)}"
                if total_steps:
                    self._progress.total_steps = int(total_steps.group(1))
                if num_epochs:
                    self._progress.total_epochs = int(num_epochs.group(1))
                    
                if self.on_progress:
                    self.on_progress(self._progress)
                    
            elif "Loading" in content and "Training" not in content:
                # 正在加载
                self._progress.status = TrainStatus.PREPARING
                self._progress.message = "正在加载模型和数据..."
                if self.on_progress:
                    self.on_progress(self._progress)
                    
        except Exception as e:
            logger.error(f"Failed to read running_log: {e}")

    async def _parse_log_entry(self, line: str) -> None:
        """解析日志条目"""
        try:
            data = json.loads(line)
            
            entry = TrainingLogEntry(
                current_steps=data.get("current_steps", 0),
                total_steps=data.get("total_steps", 0),
                loss=data.get("loss", 0.0),
                learning_rate=data.get("learning_rate", 0.0),
                epoch=data.get("epoch", 0.0),
                percentage=data.get("percentage", 0.0),
                elapsed_time=data.get("elapsed_time", ""),
                remaining_time=data.get("remaining_time", ""),
                samples_per_second=data.get("samples_per_second", 0.0),
            )
            
            # 更新进度
            self._progress = TrainingProgress(
                status=TrainStatus.TRAINING,
                progress=entry.percentage,
                current_step=entry.current_steps,
                total_steps=entry.total_steps,
                current_epoch=int(entry.epoch) if entry.epoch else None,
                loss=entry.loss,
                learning_rate=entry.learning_rate,
                eta=entry.remaining_time,
            )
            
            # 回调
            if self.on_progress:
                self.on_progress(self._progress)
            
            if self.on_log_entry:
                self.on_log_entry(entry)
                
        except json.JSONDecodeError:
            pass  # 忽略非 JSON 行

    def get_progress(self) -> TrainingProgress:
        """获取当前进度"""
        return self._progress

    def get_state(self) -> CloudState:
        """获取当前状态"""
        return CloudState(
            status=self._progress.status,
            progress=self._progress.progress,
            current_epoch=self._progress.current_epoch,
            remote_url="http://localhost:7860",
            last_update=int(time.time() * 1000),
        )

    # ============== 运行日志 ==============

    async def get_running_log(self, max_lines: int = 100) -> str:
        """获取运行日志"""
        if not self._current_output_dir:
            return ""
        
        log_path = self._current_output_dir / RUNNING_LOG
        if not log_path.exists():
            return ""
        
        try:
            async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
                content = await f.read()
            
            lines = content.split('\n')
            return '\n'.join(lines[-max_lines:])
        except Exception:
            return ""

    # ============== 工具方法 ==============

    async def get_sample_at_step(self, step: int, dataset_name: str) -> Optional[TrainingSample]:
        """
        根据训练步数获取对应的样本
        (需要配合数据加载器的采样逻辑)
        """
        samples = self._samples.get(dataset_name, [])
        if not samples:
            return None
        
        # 简化实现：根据步数取模获取样本
        # 实际情况需要与 Trainer 的采样逻辑对齐
        index = step % len(samples)
        return samples[index]

    # ============== LoRA 模型管理 ==============

    def list_lora_models(self) -> List[Dict[str, str]]:
        """
        列出可用的 LoRA 模型
        扫描 saves 目录下的所有训练输出
        """
        models = []
        saves_dir = self.lf_root / self.config.output_prefix
        
        if not saves_dir.exists():
            return models
        
        # 遍历 saves 目录
        for model_dir in saves_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            lora_dir = model_dir / "lora"
            if not lora_dir.exists():
                continue
            
            # 遍历 lora 目录下的训练输出
            for train_dir in lora_dir.iterdir():
                if not train_dir.is_dir():
                    continue
                
                # 检查是否包含 adapter_model.safetensors 或 adapter_model.bin
                has_adapter = (
                    (train_dir / "adapter_model.safetensors").exists() or
                    (train_dir / "adapter_model.bin").exists() or
                    (train_dir / "adapter_config.json").exists()
                )
                
                if has_adapter:
                    # 获取修改时间
                    try:
                        mtime = train_dir.stat().st_mtime
                        time_str = datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
                    except:
                        time_str = "unknown"
                    
                    models.append({
                        "name": f"{model_dir.name}/{train_dir.name}",
                        "path": str(train_dir.relative_to(self.lf_root)),
                        "time": time_str,
                    })
        
        # 按时间倒序排列
        models.sort(key=lambda x: x["time"], reverse=True)
        return models

    # 使用 Llama-Factory 的 ChatModel 进行推理
    _chat_model: Optional[Any] = None
    _loaded_lora_path: Optional[str] = None
    _loaded_base_model: Optional[str] = None
    _is_loading: bool = False
    
    def get_loaded_lora(self) -> Optional[str]:
        """获取当前加载的 LoRA 模型路径"""
        return self._loaded_lora_path
    
    def is_loading_model(self) -> bool:
        """是否正在加载模型"""
        return self._is_loading
    
    async def load_lora_model(self, lora_path: str) -> Dict[str, Any]:
        """
        使用 Llama-Factory 的 ChatModel 加载 LoRA 模型
        """
        if self._is_loading:
            return {"success": False, "error": "正在加载其他模型，请稍后"}
        
        full_path = self.lf_root / lora_path
        
        if not full_path.exists():
            return {"success": False, "error": f"LoRA 模型不存在: {lora_path}"}
        
        # 检查是否已经加载相同的模型
        if self._loaded_lora_path == lora_path and self._chat_model is not None:
            return {"success": True, "message": "模型已加载", "lora_path": lora_path}
        
        self._is_loading = True
        
        try:
            # 先卸载旧模型
            self.unload_lora_model()
            
            # 读取 adapter_config.json 获取基础模型
            adapter_config_path = full_path / "adapter_config.json"
            base_model = "Qwen/Qwen2-0.5B-Instruct"
            
            if adapter_config_path.exists():
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path", base_model)
            
            logger.info("Loading model using Llama-Factory ChatModel")
            logger.info(f"Base model: {base_model}")
            logger.info(f"LoRA path: {full_path}")
            
            # 使用 Llama-Factory 的 ChatModel
            import sys
            sys.path.insert(0, str(self.lf_root / "src"))
            
            from llamafactory.chat import ChatModel
            
            # 构建参数
            args = {
                "model_name_or_path": base_model,
                "adapter_name_or_path": str(full_path),
                "template": "qwen3_nothink",  # 使用 Qwen3 模板
                "finetuning_type": "lora",
                "infer_backend": "huggingface",
                "trust_remote_code": True,
            }
            
            logger.info(f"Initializing ChatModel with args: {args}")
            
            self._chat_model = ChatModel(args)
            self._loaded_lora_path = lora_path
            self._loaded_base_model = base_model
            
            logger.info("Model loaded successfully!")
            
            return {
                "success": True, 
                "message": "模型加载成功 (via Llama-Factory ChatModel)",
                "lora_path": lora_path,
                "base_model": base_model,
                "template": "qwen3_nothink",
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
        finally:
            self._is_loading = False
    
    def unload_lora_model(self) -> Dict[str, Any]:
        """
        卸载已加载的 LoRA 模型，释放显存
        """
        if self._chat_model is None:
            return {"success": True, "message": "没有已加载的模型"}
        
        old_path = self._loaded_lora_path
        
        try:
            from llamafactory.extras.misc import torch_gc
            
            # 释放模型
            self._chat_model.engine = None
            self._chat_model = None
            self._loaded_lora_path = None
            self._loaded_base_model = None
            
            # 清理显存
            torch_gc()
            
            logger.info(f"Model unloaded: {old_path}")
            
            return {"success": True, "message": f"模型已卸载: {old_path}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    async def run_lora_inference(
        self, 
        lora_path: str, 
        instruction: str,
        input_text: str,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,  # 增大默认值，不限制模型输出
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        """
        Use Llama-Factory ChatModel for inference
        
        Args:
            lora_path: Path to LoRA adapter
            instruction: System instruction (role description, capabilities, rules)
            input_text: User input (should include Focus Window)
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            enable_thinking: Enable thinking mode
        """
        # 打印完整的模型输入（调试用）
        logger.info("=" * 60)
        logger.info("[SmartPath] 模型完整输入:")
        logger.info("-" * 40)
        logger.info(f"[SYSTEM INSTRUCTION]\n{instruction}")
        logger.info("-" * 40)
        logger.info(f"[USER INPUT]\n{input_text}")
        logger.info("=" * 60)
        
        # Check if model is loaded
        if self._chat_model is None or self._loaded_lora_path != lora_path:
            # Model not loaded, use mock output
            mock_output = self._generate_mock_output(input_text, None)
            return {
                "success": True,
                "lora_path": lora_path,
                "instruction": instruction,
                "input": input_text,
                "output": mock_output,
                "note": "Model not loaded. Please click 'Load Model' button first.",
                "model_loaded": False,
                "full_prompt": f"[SYSTEM]\n{instruction}\n\n[USER]\n{input_text}",
            }
        
        # 构建消息（注意：只传 input_text 作为 user message，不要包含 output！）
        messages = [{"role": "user", "content": input_text}]
        
        try:
            logger.info(f"Starting inference with temperature={temperature}, max_tokens={max_new_tokens}")
            
            # Use ChatModel.chat() for inference
            responses = self._chat_model.chat(
                messages=messages,
                system=instruction,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            
            if responses and len(responses) > 0:
                output = responses[0].response_text
            else:
                output = ""
            
            logger.info("-" * 40)
            logger.info(f"[MODEL OUTPUT]\n{output}")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "lora_path": lora_path,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "model_loaded": True,
                "full_prompt": f"[SYSTEM]\n{instruction}\n\n[USER]\n{input_text}",
            }
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "lora_path": lora_path,
                "full_prompt": f"[SYSTEM]\n{instruction}\n\n[USER]\n{input_text}",
            }
    
    async def _do_inference(self, lora_path: str, system_prompt: str, user_input: str) -> str:
        """
        执行真正的推理
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # 检查是否需要重新加载模型
        if self._loaded_lora_path != lora_path or self._loaded_model is None:
            print(f"[SmartPath] 加载 LoRA 模型: {lora_path}")
            
            # 从 adapter_config.json 读取基础模型
            # 注意：配置中的路径可能是训练服务器的本地路径，需要转换为 HuggingFace 模型名
            adapter_config_path = Path(lora_path) / "adapter_config.json"
            base_model = "Qwen/Qwen2-0.5B-Instruct"  # 默认值
            
            if adapter_config_path.exists():
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                config_model = adapter_config.get("base_model_name_or_path", "")
                
                # 尝试从路径中提取模型名称
                if "Qwen2-0" in config_model or "Qwen2-0.5B" in config_model:
                    base_model = "Qwen/Qwen2-0.5B-Instruct"
                elif "Qwen2-1.5B" in config_model:
                    base_model = "Qwen/Qwen2-1.5B-Instruct"
                elif "Qwen2-7B" in config_model:
                    base_model = "Qwen/Qwen2-7B-Instruct"
                elif "/" in config_model and not config_model.startswith("/"):
                    # 已经是 HuggingFace 格式
                    base_model = config_model
            
            print(f"[SmartPath] 基础模型: {base_model}")
            
            # 加载 tokenizer
            self._loaded_tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
            )
            
            # 加载基础模型
            base_model_instance = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            
            # 加载 LoRA 权重
            self._loaded_model = PeftModel.from_pretrained(
                base_model_instance,
                lora_path,
            )
            self._loaded_model.eval()
            self._loaded_lora_path = lora_path
            
            print("[SmartPath] 模型加载完成")
        
        # 构建 ChatML 消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        
        # 应用 chat template
        text = self._loaded_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self._loaded_tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 推理
        import torch
        with torch.no_grad():
            outputs = self._loaded_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._loaded_tokenizer.pad_token_id,
                eos_token_id=self._loaded_tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = self._loaded_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def _generate_mock_output(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """
        生成模拟输出（当无法加载模型时）
        """
        pos = context.get('pos', '1.0') if context else '1.0'
        
        # 简单的规则匹配
        action = "setText"
        params = ["新内容"]
        
        if "红" in prompt or "color" in prompt.lower():
            action = "setColor"
            params = ["red"]
        elif "蓝" in prompt:
            action = "setColor" 
            params = ["blue"]
        elif "加粗" in prompt or "bold" in prompt.lower():
            action = "toggleBold"
            params = [True]
        elif "删" in prompt or "移除" in prompt:
            action = "remove"
            params = []
        elif "标题" in prompt or "h1" in prompt.lower() or "h2" in prompt.lower():
            action = "setHeader"
            params = ["H2"]
        
        thought = f"1. 当前位于 {context.get('type', '?')}({pos})。\n2. 用户要求执行: {prompt}\n3. 执行 {action} 操作。"
        
        action_graph = [
            {
                "target": f"pos('{pos}')",
                "ops": [[action] + params]
            }
        ]
        
        return f"<thought>\n{thought}\n</thought>\n```json\n{json.dumps(action_graph, ensure_ascii=False, indent=2)}\n```"

