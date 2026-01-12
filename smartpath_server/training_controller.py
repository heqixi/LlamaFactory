"""
Training Controller
Integrates with Llama-Factory training system
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .types import (
    SmartPathConfig,
    TrainStatus,
    TrainingProgress,
    CloudState,
    TrainStartRequest,
    ModelInfo,
)
from .sample_store import SampleStore

logger = logging.getLogger("SmartPath.Training")


class TrainingController:
    """Training Controller"""

    def __init__(self, config: SmartPathConfig, sample_store: SampleStore):
        self.config = config
        self.sample_store = sample_store
        
        self._status = TrainStatus.IDLE
        self._progress = TrainingProgress()
        self._current_job_id: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._output_dir: Optional[str] = None
        self._on_status_change: Optional[Callable[[CloudState], None]] = None
        self._monitor_task: Optional[asyncio.Task] = None

    def set_status_callback(self, callback: Callable[[CloudState], None]) -> None:
        """设置状态变更回调"""
        self._on_status_change = callback

    def get_state(self) -> CloudState:
        """获取当前状态"""
        return CloudState(
            status=self._status,
            progress=self._progress.progress,
            current_epoch=self._progress.current_epoch,
            total_epochs=self._progress.total_epochs,
            remote_url=self.config.llamafactory_webui_url,
            last_update=int(datetime.now().timestamp() * 1000),
        )

    def get_progress(self) -> TrainingProgress:
        """获取训练进度"""
        return self._progress

    async def start_training(
        self,
        request: TrainStartRequest,
        dataset_name: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        启动训练任务

        Returns:
            (success, job_id, error_message)
        """
        if self._status == TrainStatus.TRAINING:
            return False, None, "训练任务已在运行中"

        try:
            # 1. 如果没有指定数据集，创建并注册新数据集
            if not dataset_name:
                dataset_name = await self.sample_store.create_and_register_dataset()
                if not dataset_name:
                    return False, None, "创建数据集失败"

            # 2. 生成任务 ID 和输出目录
            job_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_base_dir) / f"smartpath_{timestamp}_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = str(output_dir)

            # 3. 构建训练参数
            train_args = self._build_train_args(request, dataset_name, str(output_dir))

            # 4. 启动训练进程
            success = await self._launch_training(train_args)
            
            if success:
                self._current_job_id = job_id
                self._status = TrainStatus.TRAINING
                self._progress = TrainingProgress(
                    status=TrainStatus.TRAINING,
                    progress=0.0,
                    total_epochs=request.epochs,
                    current_epoch=0,
                )
                self._notify_status_change()
                
                # 启动监控任务
                self._monitor_task = asyncio.create_task(self._monitor_training())
                
                return True, job_id, None
            else:
                return False, None, "启动训练进程失败"

        except Exception as e:
            return False, None, str(e)

    def _build_train_args(
        self,
        request: TrainStartRequest,
        dataset_name: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """构建训练参数"""
        return {
            "stage": "sft",
            "do_train": True,
            "model_name_or_path": self.config.default_base_model,
            "dataset": dataset_name,
            "template": "qwen",  # 可配置
            "finetuning_type": "lora",
            "lora_rank": request.lora_rank,
            "lora_alpha": request.lora_alpha,
            "lora_target": "all",
            "output_dir": output_dir,
            "per_device_train_batch_size": request.batch_size,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "learning_rate": request.learning_rate,
            "num_train_epochs": request.epochs,
            "max_samples": 1000,
            "cutoff_len": 1024,
            "fp16": True,
            "plot_loss": True,
        }

    async def _launch_training(self, args: Dict[str, Any]) -> bool:
        """启动训练进程"""
        try:
            # 将参数写入临时文件
            args_file = Path(self._output_dir) / "train_args.json"
            with open(args_file, "w", encoding="utf-8") as f:
                json.dump(args, f, ensure_ascii=False, indent=2)

            # 构建命令
            cmd = ["llamafactory-cli", "train", str(args_file)]

            # 设置环境变量
            env = os.environ.copy()
            env["LLAMABOARD_ENABLED"] = "1"
            env["LLAMABOARD_WORKDIR"] = self._output_dir

            # 启动进程
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(Path(__file__).parent.parent),  # LLaMA-Factory 目录
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False

    async def _monitor_training(self) -> None:
        """监控训练进度"""
        if not self._process or not self._output_dir:
            return

        trainer_log_path = Path(self._output_dir) / "trainer_log.jsonl"
        
        while self._process.poll() is None:
            try:
                # 解析训练日志
                if trainer_log_path.exists():
                    await self._parse_trainer_log(trainer_log_path)
                
                # 读取进程输出
                if self._process.stdout:
                    line = self._process.stdout.readline()
                    if line:
                        await self._parse_output_line(line)

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Training monitoring error: {e}")
                await asyncio.sleep(2)

        # 训练结束
        return_code = self._process.returncode
        
        if return_code == 0:
            self._status = TrainStatus.COMPLETED
            self._progress.status = TrainStatus.COMPLETED
            self._progress.progress = 100.0
        else:
            self._status = TrainStatus.FAILED
            self._progress.status = TrainStatus.FAILED

        self._notify_status_change()
        self._process = None

    async def _parse_trainer_log(self, log_path: Path) -> None:
        """解析 Llama-Factory 训练日志"""
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            if not lines:
                return

            # 获取最新的日志条目
            last_line = lines[-1].strip()
            if not last_line:
                return

            log_entry = json.loads(last_line)
            
            # 更新进度
            if "current_steps" in log_entry and "total_steps" in log_entry:
                current = log_entry["current_steps"]
                total = log_entry["total_steps"]
                self._progress.current_step = current
                self._progress.total_steps = total
                self._progress.progress = (current / total) * 100 if total > 0 else 0

            if "loss" in log_entry:
                self._progress.loss = log_entry["loss"]

            if "learning_rate" in log_entry:
                self._progress.learning_rate = log_entry["learning_rate"]

            if "epoch" in log_entry:
                self._progress.current_epoch = int(log_entry["epoch"])

            self._notify_status_change()

        except Exception as e:
            pass  # 日志可能还在写入

    async def _parse_output_line(self, line: str) -> None:
        """解析训练输出行"""
        # 匹配常见的进度模式
        # 例如: "Epoch 1/3: 50%|████      | 100/200 [00:30<00:30, 3.33it/s, loss=2.345]"
        
        epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", line)
        if epoch_match:
            self._progress.current_epoch = int(epoch_match.group(1))
            self._progress.total_epochs = int(epoch_match.group(2))

        progress_match = re.search(r"(\d+)%\|", line)
        if progress_match:
            self._progress.progress = float(progress_match.group(1))

        loss_match = re.search(r"loss[=:]\s*([\d.]+)", line, re.IGNORECASE)
        if loss_match:
            self._progress.loss = float(loss_match.group(1))

    async def stop_training(self) -> bool:
        """停止训练"""
        if self._process is None:
            return False

        try:
            self._process.terminate()
            
            # 等待进程结束
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

            self._status = TrainStatus.IDLE
            self._progress = TrainingProgress()
            self._process = None
            self._current_job_id = None
            
            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None

            self._notify_status_change()
            return True

        except Exception as e:
            logger.error(f"Failed to stop training: {e}")
            return False

    def _notify_status_change(self) -> None:
        """通知状态变更"""
        if self._on_status_change:
            state = self.get_state()
            self._on_status_change(state)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用的模型列表"""
        models = []
        
        # 添加基础模型
        models.append({
            "id": "base",
            "name": self.config.default_base_model,
            "type": "base",
        })

        # 扫描输出目录中的 LoRA 模型
        output_base = Path(self.config.output_base_dir)
        if output_base.exists():
            for dir_path in output_base.iterdir():
                if dir_path.is_dir() and dir_path.name.startswith("smartpath_"):
                    adapter_config = dir_path / "adapter_config.json"
                    if adapter_config.exists():
                        models.append({
                            "id": dir_path.name,
                            "name": f"LoRA: {dir_path.name}",
                            "type": "lora",
                            "path": str(dir_path),
                        })

        return models

    def get_latest_adapter_path(self) -> Optional[str]:
        """获取最新的 LoRA adapter 路径"""
        output_base = Path(self.config.output_base_dir)
        if not output_base.exists():
            return None

        adapters = []
        for dir_path in output_base.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith("smartpath_"):
                adapter_config = dir_path / "adapter_config.json"
                if adapter_config.exists():
                    adapters.append((dir_path, dir_path.stat().st_mtime))

        if not adapters:
            return None

        # 返回最新的
        adapters.sort(key=lambda x: x[1], reverse=True)
        return str(adapters[0][0])

