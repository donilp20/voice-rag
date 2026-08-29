"""
WebSocket endpoint for the Voice RAG pipeline.

This is a STUB for Step 1 (scaffolding). It establishes the connection
contract only. The actual audio-in -> FSM orchestration -> streamed
answer-out logic gets wired in once STT (Step 5), retrieval (Step 3),
guardrails (Step 4), generation (Step 6), and the orchestrator
(Step 7) all exist.
"""

import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    query_id = str(uuid.uuid4())

    try:
        # Placeholder handshake so the endpoint is testable end-to-end
        # before the real pipeline is wired in.
        await websocket.send_json(
            {
                "query_id": query_id,
                "status": "CONNECTED",
                "message": "Voice RAG WebSocket stub — pipeline not yet wired.",
            }
        )

        while True:
            # In the real implementation this receives binary audio frames
            # (or base64 chunks) and feeds them into the STT layer /
            # orchestrator. For now we just echo receipt.
            data = await websocket.receive()

            if data.get("type") == "websocket.disconnect":
                break

            await websocket.send_json(
                {
                    "query_id": query_id,
                    "status": "RECEIVED_AUDIO",
                    "message": "Audio frame received (stub — no processing yet).",
                }
            )

    except WebSocketDisconnect:
        pass