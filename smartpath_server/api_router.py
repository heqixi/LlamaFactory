"""
FastAPI 路由器
提供 RESTful API 和 WebSocket 端点
深度集成 Llama-Factory
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .types import (
    SmartPathConfig,
    TrainingSample,
    SyncRequest,
    SyncResponse,
    InferenceRequest,
    InferenceResponse,
    TrainStartRequest,
    TrainStartResponse,
    CloudState,
    TrainingProgress,
    ModelSwitchRequest,
    ModelListItem,
)
from .sample_store import SampleStore
from .training_controller import TrainingController
from .inference_engine import InferenceEngine
from .websocket_manager import WebSocketManager
from .llamafactory_bridge import (
    LlamaFactoryBridge,
    LlamaFactoryBridgeConfig,
    TrainingLogEntry,
)


class DatasetInfo(BaseModel):
    """数据集信息"""
    name: str
    sample_count: int
    file_path: str


class SamplePreview(BaseModel):
    """样本预览"""
    instruction: str
    input_preview: str
    output_preview: str


def create_smartpath_router(
    config: Optional[SmartPathConfig] = None,
    llamafactory_root: str = "./LLaMA-Factory",
    sample_store: Optional[SampleStore] = None,
    training_controller: Optional[TrainingController] = None,
    inference_engine: Optional[InferenceEngine] = None,
    ws_manager: Optional[WebSocketManager] = None,
) -> APIRouter:
    """
    创建 SmartPath API 路由器
    深度集成 Llama-Factory
    """
    # 使用默认配置
    if config is None:
        config = SmartPathConfig()

    # 创建组件
    if sample_store is None:
        sample_store = SampleStore(config)

    if training_controller is None:
        training_controller = TrainingController(config, sample_store)

    if inference_engine is None:
        inference_engine = InferenceEngine(config, training_controller)

    if ws_manager is None:
        ws_manager = WebSocketManager()

    # 创建 Llama-Factory 桥接器
    bridge_config = LlamaFactoryBridgeConfig(
        llamafactory_root=llamafactory_root,
        auto_register=True,
    )
    
    bridge = LlamaFactoryBridge(
        config=bridge_config,
        on_progress=lambda p: asyncio.create_task(ws_manager.broadcast_progress(p)),
        on_log_entry=lambda e: asyncio.create_task(
            ws_manager.broadcast({
                "type": "log_entry",
                "data": {
                    "current_steps": e.current_steps,
                    "total_steps": e.total_steps,
                    "loss": e.loss,
                    "epoch": e.epoch,
                    "percentage": e.percentage,
                    "elapsed_time": e.elapsed_time,
                    "remaining_time": e.remaining_time,
                }
            })
        ),
    )

    router = APIRouter(prefix=config.api_prefix, tags=["SmartPath"])

    # ============== 生命周期 ==============

    @router.on_event("startup")
    async def startup():
        await sample_store.initialize()
        await bridge.initialize()
        print("[SmartPath] 服务已启动")

    # ============== 样本同步 (直接写入 LLaMA-Factory/data) ==============

    @router.post("/samples/sync", response_model=SyncResponse)
    async def sync_samples(request: SyncRequest, dataset_name: Optional[str] = None, append: bool = True):
        """
        同步训练样本到 Llama-Factory
        
        样本将直接写入 LLaMA-Factory/data 目录，并自动更新 dataset_info.json
        在 Llama-Factory WebUI 中可以直接选择该数据集进行训练
        """
        try:
            success, ds_name = await bridge.sync_samples(
                request.samples,
                dataset_name=dataset_name,
                append=append,
            )
            
            if success:
                # 同时保存到本地存储
                await sample_store.add_samples(request.samples)
                
                return SyncResponse(
                    success=True,
                    synced_count=len(request.samples),
                    failed_count=0,
                    message=f"已同步到数据集 '{ds_name}'，可在 WebUI 中选择训练",
                )
            else:
                return SyncResponse(
                    success=False,
                    synced_count=0,
                    failed_count=len(request.samples),
                    message="同步失败",
                )
        except Exception as e:
            return SyncResponse(
                success=False,
                synced_count=0,
                failed_count=len(request.samples),
                message=str(e),
            )

    @router.post("/samples/add")
    async def add_sample(sample: TrainingSample, dataset_name: str):
        """添加单个样本到指定数据集"""
        success = await bridge.add_sample(sample, dataset_name)
        return {"success": success}

    @router.delete("/samples/{sample_id}")
    async def delete_sample(sample_id: str, dataset_name: str):
        """从数据集中删除样本"""
        success = await bridge.remove_sample(sample_id, dataset_name)
        if not success:
            raise HTTPException(status_code=404, detail="样本不存在")
        return {"success": True}

    @router.get("/samples", response_model=List[TrainingSample])
    async def get_samples(dataset_name: Optional[str] = None):
        """获取样本列表"""
        if dataset_name:
            return bridge.get_dataset_samples(dataset_name)
        return await sample_store.get_all_samples()

    @router.get("/samples/stats")
    async def get_sample_stats():
        """获取样本统计"""
        datasets = bridge.list_datasets()
        return {
            "datasets": datasets,
            "total_samples": sum(d["sample_count"] for d in datasets),
            "local_samples": sample_store.get_stats(),
        }

    # ============== 数据集管理 ==============

    @router.get("/datasets", response_model=List[DatasetInfo])
    async def list_datasets():
        """
        列出所有 SmartPath 数据集
        这些数据集可以在 Llama-Factory WebUI 中直接选择
        """
        return [DatasetInfo(**d) for d in bridge.list_datasets()]

    @router.post("/datasets/create")
    async def create_dataset(dataset_name: str):
        """创建空数据集"""
        success, name = await bridge.sync_samples([], dataset_name=dataset_name)
        return {"success": success, "dataset_name": name}

    @router.get("/datasets/{dataset_name}/preview")
    async def preview_dataset(dataset_name: str, limit: int = 10):
        """预览数据集内容"""
        samples = bridge.get_dataset_samples(dataset_name)
        previews = []
        for s in samples[:limit]:
            previews.append(SamplePreview(
                instruction=s.instruction[:100] + "..." if len(s.instruction) > 100 else s.instruction,
                input_preview=s.input[:200] + "..." if len(s.input) > 200 else s.input,
                output_preview=s.output[:200] + "..." if len(s.output) > 200 else s.output,
            ))
        return {
            "dataset_name": dataset_name,
            "total_count": len(samples),
            "previews": previews,
        }

    # ============== 训练监控 ==============

    @router.get("/train/status", response_model=CloudState)
    async def get_train_status():
        """获取训练状态"""
        return bridge.get_state()

    @router.get("/train/progress", response_model=TrainingProgress)
    async def get_train_progress():
        """获取详细训练进度"""
        return bridge.get_progress()

    @router.post("/train/monitor/start")
    async def start_monitoring(output_dir: Optional[str] = None):
        """
        开始监控训练进度
        
        如果不指定 output_dir，会自动查找最新的训练输出目录
        监控开始后，进度更新会通过 WebSocket 实时推送
        """
        await bridge.start_monitoring(output_dir)
        return {"success": True, "message": "已开始监控训练日志"}

    @router.post("/train/monitor/stop")
    async def stop_monitoring():
        """停止监控"""
        await bridge.stop_monitoring()
        return {"success": True}

    @router.get("/train/log")
    async def get_training_log(max_lines: int = 100):
        """获取训练日志"""
        log = await bridge.get_running_log(max_lines)
        return {"log": log}

    # ============== 训练控制 (通过 Llama-Factory WebUI) ==============

    @router.post("/train/start", response_model=TrainStartResponse)
    async def start_training(request: TrainStartRequest):
        """
        启动训练
        
        注意: 实际训练需要在 Llama-Factory WebUI (http://localhost:7860) 中操作
        此 API 仅用于准备数据集和开始监控
        """
        # 如果指定了数据集，确保它已同步
        if request.dataset_name:
            datasets = bridge.list_datasets()
            if not any(d["name"] == request.dataset_name for d in datasets):
                return TrainStartResponse(
                    success=False,
                    message=f"数据集 '{request.dataset_name}' 不存在，请先同步样本"
                )
        
        # 开始监控
        await bridge.start_monitoring()
        
        return TrainStartResponse(
            success=True,
            message=(
                f"请在 Llama-Factory WebUI (http://localhost:7860) 中:\n"
                f"1. 选择数据集: {request.dataset_name or '任意 smartpath_* 数据集'}\n"
                f"2. 配置训练参数\n"
                f"3. 点击 'Start' 开始训练\n"
                f"训练进度将实时推送到此服务"
            )
        )

    @router.post("/train/stop")
    async def stop_training():
        """停止训练监控"""
        await bridge.stop_monitoring()
        return {"success": True, "message": "已停止监控"}

    # ============== 推理验证 ==============

    @router.post("/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        """执行推理验证"""
        return await inference_engine.inference(request)

    # ============== LoRA 模型管理 ==============

    @router.get("/lora/list")
    async def list_lora_models():
        """
        获取可用的 LoRA 模型列表
        扫描 saves 目录下的所有训练输出
        """
        models = bridge.list_lora_models()
        loaded = bridge.get_loaded_lora()
        return {
            "models": models,
            "loaded_model": loaded,
        }

    @router.post("/lora/load")
    async def load_lora_model(request: dict):
        """
        加载 LoRA 模型（预热）
        加载模型是耗时操作，建议先调用此接口预加载
        """
        lora_path = request.get("lora_path")
        if not lora_path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        
        result = await bridge.load_lora_model(lora_path)
        return result

    @router.post("/lora/unload")
    async def unload_lora_model():
        """
        卸载已加载的 LoRA 模型，释放显存
        """
        result = bridge.unload_lora_model()
        return result

    @router.get("/lora/status")
    async def get_lora_status():
        """
        获取当前模型加载状态
        """
        return {
            "loaded_model": bridge.get_loaded_lora(),
            "is_loading": bridge.is_loading_model(),
        }

    @router.post("/lora/inference")
    async def lora_inference(request: dict):
        """
        使用指定的 LoRA 模型进行推理
        
        请求体:
        {
            "lora_path": "saves/Qwen2-0.5B-Instruct/lora/train_xxx",
            "prompt": "用户指令",
            "context": {"pos": "1.2", "type": "h1", "label": "标题内容"}
        }
        """
        lora_path = request.get("lora_path")
        prompt = request.get("prompt", "")
        context = request.get("context")
        
        if not lora_path:
            raise HTTPException(status_code=400, detail="lora_path is required")
        
        result = await bridge.run_lora_inference(lora_path, prompt, context)
        return result

    # ============== 模型管理 ==============

    @router.get("/models", response_model=List[ModelListItem])
    async def list_models():
        """获取可用模型列表"""
        models = training_controller.get_available_models()
        return [ModelListItem(**m) for m in models]

    @router.post("/model/switch")
    async def switch_model(request: ModelSwitchRequest):
        """切换推理模型"""
        success = await inference_engine.switch_model(request.model_id)
        return {"success": success}

    # ============== 实时样本查看 ==============

    @router.get("/train/current-sample")
    async def get_current_sample(dataset_name: str):
        """
        获取当前训练步对应的样本
        (用于在推理终端实时展示正在训练的样本)
        """
        progress = bridge.get_progress()
        if progress.current_step:
            sample = await bridge.get_sample_at_step(progress.current_step, dataset_name)
            if sample:
                return {
                    "step": progress.current_step,
                    "sample": {
                        "instruction": sample.instruction,
                        "input": sample.input[:500],
                        "output": sample.output[:500],
                    }
                }
        return {"step": 0, "sample": None}

    # ============== WebSocket ==============

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket 端点
        
        消息类型:
        - status: 训练状态更新
        - progress: 详细进度更新
        - log_entry: 日志条目
        """
        await ws_manager.connect(websocket)
        
        # 自动开始监控训练日志
        await bridge.start_monitoring()
        
        # 发送当前状态
        await ws_manager.send_to(websocket, {
            "type": "status",
            "data": bridge.get_state().model_dump(),
        })

        try:
            while True:
                data = await websocket.receive_text()
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    await ws_manager.send_to(websocket, {
                        "type": "status",
                        "data": bridge.get_state().model_dump(),
                    })
                elif data == "progress":
                    await ws_manager.send_to(websocket, {
                        "type": "progress",
                        "data": bridge.get_progress().model_dump(),
                    })
                elif data.startswith("monitor:"):
                    # 开始监控指定目录
                    output_dir = data.split(":", 1)[1] if ":" in data else None
                    await bridge.start_monitoring(output_dir or None)
                    await ws_manager.send_to(websocket, {
                        "type": "info",
                        "data": "监控已启动",
                    })

        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)

    # ============== 健康检查 ==============

    @router.get("/health")
    async def health_check():
        """健康检查"""
        datasets = bridge.list_datasets()
        return {
            "status": "ok",
            "llamafactory_connected": True,
            "datasets_count": len(datasets),
            "total_samples": sum(d["sample_count"] for d in datasets),
            "training_status": bridge.get_state().status.value,
            "websocket_connections": ws_manager.connection_count,
            "webui_url": "http://localhost:7860",
        }

    @router.get("/info")
    async def get_info():
        """获取服务信息"""
        return {
            "service": "SmartPath Training Bridge",
            "version": "1.0.0",
            "llamafactory_webui": "http://localhost:7860",
            "api_docs": "/docs",
            "websocket": "/v1/smartpath/ws",
            "usage": {
                "1_sync_samples": "POST /v1/smartpath/samples/sync - 同步样本到 Llama-Factory",
                "2_list_datasets": "GET /v1/smartpath/datasets - 查看可用数据集",
                "3_train_in_webui": "在 http://localhost:7860 选择数据集并开始训练",
                "4_monitor_progress": "WS /v1/smartpath/ws 或 GET /v1/smartpath/train/progress",
            }
        }

    return router


def mount_smartpath_to_app(app, config: Optional[SmartPathConfig] = None, llamafactory_root: str = "./LLaMA-Factory"):
    """
    便捷函数：将 SmartPath 路由挂载到现有的 FastAPI 应用
    """
    router = create_smartpath_router(config, llamafactory_root=llamafactory_root)
    app.include_router(router)
    return app

