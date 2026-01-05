#!/usr/bin/env python3
"""
SmartPath 与 Llama-Factory 深度集成服务

功能:
1. 样本同步 - 将前端样本直接写入 data/ 目录，自动注册到 dataset_info.json
2. 训练监控 - 实时监控 trainer_log.jsonl，通过 WebSocket 推送进度
3. 状态同步 - 与前端推理验证终端保持实时同步

启动方式:
    python smartpath_integration.py

环境变量:
    SMARTPATH_HOST: 服务监听地址 (默认 0.0.0.0)
    SMARTPATH_PORT: 服务监听端口 (默认 8000)
"""

import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 直接从本地模块导入 SmartPath Server SDK
from smartpath_server import (
    create_smartpath_router,
    SmartPathConfig,
)


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    
    app = FastAPI(
        title="SmartPath Training Bridge",
        description="""
## SmartPath 训练桥接服务

将前端编辑器与 Llama-Factory 训练系统深度集成。

### 工作流程

1. **同步样本**: 前端调用 `POST /v1/smartpath/samples/sync` 将样本推送到服务
2. **查看数据集**: 在 [Llama-Factory WebUI](http://localhost:7860) 中选择 `smartpath_*` 数据集
3. **开始训练**: 在 WebUI 中配置参数并启动训练
4. **实时监控**: 连接 WebSocket `/v1/smartpath/ws` 接收训练进度

### 关键接口

- `POST /v1/smartpath/samples/sync` - 同步样本
- `GET /v1/smartpath/datasets` - 查看数据集
- `GET /v1/smartpath/train/progress` - 获取训练进度
- `WS /v1/smartpath/ws` - WebSocket 实时推送
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # SmartPath 配置
    config = SmartPathConfig(
        data_dir="./smartpath_data",
        default_base_model=os.getenv("SMARTPATH_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        output_base_dir="./saves",
        llamafactory_webui_url="http://localhost:7860",
        api_prefix="/v1/smartpath",
        enable_websocket=True,
    )
    
    # 挂载 SmartPath 路由
    llamafactory_root = Path(__file__).parent
    router = create_smartpath_router(
        config=config,
        llamafactory_root=str(llamafactory_root),
    )
    app.include_router(router)
    
    # 根路由
    @app.get("/")
    async def root():
        return {
            "service": "SmartPath Training Bridge",
            "version": "1.0.0",
            "llamafactory_webui": "http://localhost:7860",
            "api_docs": "/docs",
            "endpoints": {
                "sync_samples": "POST /v1/smartpath/samples/sync",
                "list_datasets": "GET /v1/smartpath/datasets",
                "train_progress": "GET /v1/smartpath/train/progress",
                "websocket": "WS /v1/smartpath/ws",
                "health": "GET /v1/smartpath/health",
            },
            "usage": [
                "1. 调用 POST /v1/smartpath/samples/sync 同步训练样本",
                "2. 在 http://localhost:7860 选择 smartpath_* 数据集",
                "3. 配置训练参数并点击 Start",
                "4. 连接 WebSocket 接收实时进度",
            ]
        }
    
    return app


def main():
    """主入口"""
    host = os.getenv("SMARTPATH_HOST", "0.0.0.0")
    port = int(os.getenv("SMARTPATH_PORT", "6006"))  # 远程训练后端代理端口
    
    print()
    print("=" * 70)
    print("  SmartPath Training Bridge")
    print("=" * 70)
    print()
    print(f"  [*] Server: http://localhost:{port}")
    print(f"  [*] API Docs: http://localhost:{port}/docs")
    print(f"  [*] WebSocket: ws://localhost:{port}/v1/smartpath/ws")
    print()
    print("  [*] Llama-Factory WebUI: http://localhost:7860")
    print()
    print("=" * 70)
    print()
    print("  Usage:")
    print("  1. Generate samples (semantic/physical path)")
    print("  2. Sync to training backend")
    print("  3. Select dataset in WebUI and start training")
    print("  4. Progress pushed to frontend in realtime")
    print()
    print("=" * 70)
    print()
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
