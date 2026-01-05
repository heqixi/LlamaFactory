"""
WebSocket 连接管理器
负责实时状态推送
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Set

from fastapi import WebSocket

from .types import CloudState, TrainingProgress


class WebSocketManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """添加新连接"""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        print(f"[SmartPath] WebSocket 客户端已连接，当前连接数: {len(self._connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """移除连接"""
        async with self._lock:
            self._connections.discard(websocket)
        print(f"[SmartPath] WebSocket 客户端已断开，当前连接数: {len(self._connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """广播消息到所有连接"""
        if not self._connections:
            return

        message_str = json.dumps(message, ensure_ascii=False)
        
        # 收集需要移除的失效连接
        disconnected: List[WebSocket] = []

        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message_str)
                except Exception:
                    disconnected.append(ws)

            # 移除失效连接
            for ws in disconnected:
                self._connections.discard(ws)

    async def broadcast_status(self, state: CloudState) -> None:
        """广播状态更新"""
        await self.broadcast({
            "type": "status",
            "data": state.model_dump(),
        })

    async def broadcast_progress(self, progress: TrainingProgress) -> None:
        """广播进度更新"""
        await self.broadcast({
            "type": "progress",
            "data": progress.model_dump(),
        })

    async def broadcast_error(self, error: str) -> None:
        """广播错误消息"""
        await self.broadcast({
            "type": "error",
            "data": error,
        })

    async def send_to(self, websocket: WebSocket, message: Dict[str, Any]) -> bool:
        """发送消息到特定连接"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            return True
        except Exception:
            return False

    @property
    def connection_count(self) -> int:
        """获取当前连接数"""
        return len(self._connections)

    def get_connections(self) -> Set[WebSocket]:
        """获取所有连接"""
        return self._connections.copy()

