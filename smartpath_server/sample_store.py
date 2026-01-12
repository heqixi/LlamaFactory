"""
Sample Storage Manager
Handles sample persistence and dataset generation
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from .types import (
    TrainingSample,
    SampleMetadata,
    SmartPathConfig,
)

logger = logging.getLogger("SmartPath.SampleStore")


class SampleStore:
    """样本存储管理器"""

    def __init__(self, config: SmartPathConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.samples_file = self.data_dir / config.samples_file
        self.dataset_dir = self.data_dir / config.dataset_dir
        self._samples: Dict[str, TrainingSample] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """初始化存储"""
        if self._initialized:
            return

        # 创建目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        # 加载现有样本
        await self._load_samples()
        self._initialized = True

    async def _load_samples(self) -> None:
        """从文件加载样本"""
        if not self.samples_file.exists():
            return

        try:
            async with aiofiles.open(self.samples_file, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
                
                for item in data:
                    sample = TrainingSample(**item)
                    if sample.id:
                        self._samples[sample.id] = sample
        except Exception as e:
            logger.error(f"Failed to load samples: {e}")

    async def _save_samples(self) -> None:
        """保存样本到文件"""
        try:
            data = [s.model_dump() for s in self._samples.values()]
            async with aiofiles.open(self.samples_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"Failed to save samples: {e}")

    async def add_sample(self, sample: TrainingSample) -> str:
        """添加样本"""
        if not sample.id:
            sample.id = str(uuid.uuid4())
        
        if not sample.timestamp:
            sample.timestamp = int(datetime.now().timestamp() * 1000)

        self._samples[sample.id] = sample
        await self._save_samples()
        return sample.id

    async def add_samples(self, samples: List[TrainingSample]) -> List[str]:
        """批量添加样本"""
        ids = []
        for sample in samples:
            if not sample.id:
                sample.id = str(uuid.uuid4())
            if not sample.timestamp:
                sample.timestamp = int(datetime.now().timestamp() * 1000)
            self._samples[sample.id] = sample
            ids.append(sample.id)
        
        await self._save_samples()
        return ids

    async def get_sample(self, sample_id: str) -> Optional[TrainingSample]:
        """获取样本"""
        return self._samples.get(sample_id)

    async def get_all_samples(self) -> List[TrainingSample]:
        """获取所有样本"""
        return list(self._samples.values())

    async def get_hard_samples(self) -> List[TrainingSample]:
        """获取难例样本"""
        return [s for s in self._samples.values() if s.is_hard]

    async def delete_sample(self, sample_id: str) -> bool:
        """删除样本"""
        if sample_id in self._samples:
            del self._samples[sample_id]
            await self._save_samples()
            return True
        return False

    async def clear_all(self) -> None:
        """清空所有样本"""
        self._samples.clear()
        await self._save_samples()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        samples = list(self._samples.values())
        return {
            "total": len(samples),
            "hard_samples": sum(1 for s in samples if s.is_hard),
            "synthetic_samples": sum(
                1 for s in samples 
                if s.metadata and s.metadata.source == "synthetic"
            ),
            "feedback_samples": sum(
                1 for s in samples 
                if s.metadata and s.metadata.source == "feedback"
            ),
        }

    async def export_to_alpaca(
        self,
        filename: Optional[str] = None,
        include_hard_only: bool = False,
    ) -> str:
        """
        导出为 Alpaca 格式的训练数据集
        
        Returns:
            导出的文件路径
        """
        samples = self._samples.values()
        
        if include_hard_only:
            samples = [s for s in samples if s.is_hard]

        # 转换为 Alpaca 格式
        alpaca_data = []
        for sample in samples:
            alpaca_data.append({
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
                "history": sample.history,
            })

        # 确定文件名
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"smartpath_dataset_{timestamp}.json"

        filepath = self.dataset_dir / filename

        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(json.dumps(alpaca_data, ensure_ascii=False, indent=2))

        return str(filepath)

    async def register_dataset_info(self, dataset_name: str, filepath: str) -> bool:
        """
        在 Llama-Factory 的 dataset_info.json 中注册数据集
        """
        try:
            # 查找 Llama-Factory 的 data 目录
            # 假设相对于当前工作目录
            llamafactory_data_dir = Path("./LLaMA-Factory/data")
            if not llamafactory_data_dir.exists():
                llamafactory_data_dir = Path("../LLaMA-Factory/data")
            
            if not llamafactory_data_dir.exists():
                logger.warning("LLaMA-Factory data directory not found")
                return False

            dataset_info_path = llamafactory_data_dir / "dataset_info.json"
            
            # 读取现有配置
            if dataset_info_path.exists():
                async with aiofiles.open(dataset_info_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    dataset_info = json.loads(content)
            else:
                dataset_info = {}

            # 添加新数据集
            # 计算相对路径
            rel_path = os.path.relpath(filepath, llamafactory_data_dir)
            
            dataset_info[dataset_name] = {
                "file_name": rel_path,
            }

            # 写回配置
            async with aiofiles.open(dataset_info_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(dataset_info, ensure_ascii=False, indent=2))

            logger.info(f"Registered dataset '{dataset_name}' -> {rel_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to register dataset: {e}")
            return False

    async def create_and_register_dataset(
        self,
        dataset_name: Optional[str] = None,
        include_hard_only: bool = False,
    ) -> Optional[str]:
        """
        创建数据集并注册到 Llama-Factory
        
        Returns:
            数据集名称 (如果成功)
        """
        if not dataset_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"smartpath_{timestamp}"

        filename = f"{dataset_name}.json"
        filepath = await self.export_to_alpaca(filename, include_hard_only)

        if self.config.auto_register_dataset:
            success = await self.register_dataset_info(dataset_name, filepath)
            if success:
                return dataset_name

        return dataset_name

