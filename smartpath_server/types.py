"""
SmartPath Server SDK - 类型定义
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============== 核心数据结构 ==============

class SmartNode(BaseModel):
    """Smart-JSON 节点结构 (VDOM)"""
    uid: str = ""
    tag: str
    path: str
    content: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    children: List["SmartNode"] = Field(default_factory=list)
    semantic_features: Optional[str] = None
    level: Optional[int] = None


class ActionPlan(BaseModel):
    """推理行动计划"""
    action: str  # setStyle, setText, remove, insert
    targetPath: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class SampleMetadata(BaseModel):
    """样本元数据"""
    is_hard: bool = False
    vdom_depth: Optional[int] = None
    source: Optional[str] = None  # synthetic, manual, feedback
    model_version: Optional[str] = None
    correction_count: int = 0


class TrainingSample(BaseModel):
    """训练样本 (Alpaca 格式)"""
    id: Optional[str] = None
    instruction: str
    input: str
    output: str
    timestamp: Optional[int] = None
    is_hard: bool = False
    metadata: Optional[SampleMetadata] = None
    history: List[Any] = Field(default_factory=list)


# ============== 训练状态 ==============

class TrainStatus(str, Enum):
    """训练状态枚举"""
    IDLE = "idle"
    PENDING = "pending"
    PREPARING = "preparing"  # 正在加载模型/数据
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingProgress(BaseModel):
    """训练进度"""
    status: TrainStatus = TrainStatus.IDLE
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta: Optional[str] = None
    gpu_utilization: Optional[float] = None
    memory_usage: Optional[float] = None
    message: Optional[str] = None  # 状态消息


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    version: str
    base_model: str
    adapter_path: Optional[str] = None


class CloudState(BaseModel):
    """云端状态"""
    status: TrainStatus = TrainStatus.IDLE
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    remote_url: str = ""
    model_info: Optional[ModelInfo] = None
    last_update: Optional[int] = None


# ============== API 请求/响应 ==============

class SyncRequest(BaseModel):
    """样本同步请求"""
    samples: List[TrainingSample]


class SyncResponse(BaseModel):
    """同步响应"""
    success: bool
    synced_count: int = 0
    failed_count: int = 0
    failed_ids: Optional[List[str]] = None
    message: Optional[str] = None


class InferenceRequest(BaseModel):
    """推理请求"""
    instruction: str
    vdom: SmartNode
    model_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """推理响应"""
    success: bool
    actions: List[ActionPlan] = Field(default_factory=list)
    raw_output: Optional[str] = None
    latency: Optional[float] = None
    model_used: Optional[str] = None


class ValidationResult(BaseModel):
    """验证结果"""
    sample_id: str
    expected_path: str
    actual_path: str
    is_match: bool
    path_deviation: Optional[int] = None
    suggestion: Optional[str] = None


class TrainStartRequest(BaseModel):
    """启动训练请求"""
    dataset_name: Optional[str] = None
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    lora_rank: int = 8
    lora_alpha: int = 16
    output_dir: Optional[str] = None


class TrainStartResponse(BaseModel):
    """启动训练响应"""
    success: bool
    job_id: Optional[str] = None
    message: Optional[str] = None


class ModelSwitchRequest(BaseModel):
    """模型切换请求"""
    model_id: str


class ModelListItem(BaseModel):
    """模型列表项"""
    id: str
    name: str
    type: str  # base, lora, merged
    path: Optional[str] = None
    created_at: Optional[datetime] = None


# ============== 配置 ==============

class SmartPathConfig(BaseModel):
    """SmartPath 服务配置"""
    # 数据存储
    data_dir: str = "./smartpath_data"
    samples_file: str = "samples.json"
    dataset_dir: str = "datasets"
    
    # 训练配置
    default_base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_base_dir: str = "./outputs"
    
    # 推理配置
    default_inference_model: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.1
    
    # 服务配置
    api_prefix: str = "/v1/smartpath"
    enable_websocket: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    # Llama-Factory 集成
    llamafactory_webui_url: str = "http://localhost:7860"
    auto_register_dataset: bool = True


# 允许递归模型
SmartNode.model_rebuild()

