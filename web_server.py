import json
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from law_agent import chat_with_agent, chat_with_agent_stream, storage


app = FastAPI(title="Law Assistant Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).parent
FRONTEND_FILE = ROOT / "web_frontend.html"


class ChatRequest(BaseModel):
    message: str | None = None
    question: str | None = None
    session_id: str | None = "default_session"


@app.get("/")
def index():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="web_frontend.html 不存在")
    return FileResponse(FRONTEND_FILE)


@app.post("/chat")
def chat(req: ChatRequest):
    message = (req.message or req.question or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    try:
        return chat_with_agent(message, user_id="default_user", session_id=req.session_id or "default_session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    message = (req.message or req.question or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    async def event_generator():
        try:
            async for chunk in chat_with_agent_stream(
                message,
                user_id="default_user",
                session_id=req.session_id or "default_session",
            ):
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/sessions")
def list_sessions():
    return {"sessions": storage.list_session_infos("default_user")}


@app.get("/sessions/{session_id}")
def get_session_messages(session_id: str):
    return {"messages": storage.get_session_messages("default_user", session_id)}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = storage.delete_session("default_user", session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "message": "成功删除会话"}


@app.get("/health")
def health():
    return {"ok": True, "service": "law-assistant-web"}


def open_browser_later(url: str, delay_seconds: float = 1.2):
    def _open():
        time.sleep(delay_seconds)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    target_url = f"http://{host}:{port}"

    print(f"正在启动服务: {target_url}")
    open_browser_later(target_url)
    uvicorn.run("web_server:app", host=host, port=port, reload=False)
