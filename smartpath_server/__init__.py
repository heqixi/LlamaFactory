"""
SmartPath Server SDK
用于 Llama-Factory 训练系统集成

提供样本接收、训练控制、状态监控等功能
"""

from .types import (
    SmartNode,
    TrainingSample,
    SampleMetadata,
    ActionPlan,
    TrainStatus,
    TrainingProgress,
    CloudState,
    SyncRequest,
    SyncResponse,
    InferenceRequest,
    InferenceResponse,
    ValidationResult,
    SmartPathConfig,
)
from .sample_store import SampleStore
from .training_controller import TrainingController
from .inference_engine import InferenceEngine
from .api_router import create_smartpath_router, mount_smartpath_to_app
from .websocket_manager import WebSocketManager
from .llamafactory_bridge import (
    LlamaFactoryBridge,
    LlamaFactoryBridgeConfig,
    TrainingLogEntry,
)

__version__ = "1.0.0"

__all__ = [
    # 类型
    "SmartNode",
    "TrainingSample",
    "SampleMetadata",
    "ActionPlan",
    "TrainStatus",
    "TrainingProgress",
    "CloudState",
    "SyncRequest",
    "SyncResponse",
    "InferenceRequest",
    "InferenceResponse",
    "ValidationResult",
    "SmartPathConfig",
    # 核心组件
    "SampleStore",
    "TrainingController",
    "InferenceEngine",
    "WebSocketManager",
    # Llama-Factory 桥接
    "LlamaFactoryBridge",
    "LlamaFactoryBridgeConfig",
    "TrainingLogEntry",
    # API
    "create_smartpath_router",
    "mount_smartpath_to_app",
]

