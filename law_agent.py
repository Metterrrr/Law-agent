import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, List

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from search import rag_search


class ConversationStorage:
    """轻量会话存储：对齐 backend 接口，便于 web_server 直接复用。"""

    def __init__(self):
        self._messages: Dict[str, List[dict]] = {}

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        return f"{user_id}:{session_id}"

    def load(self, user_id: str, session_id: str) -> List[dict]:
        return list(self._messages.get(self._key(user_id, session_id), []))

    def save(self, user_id: str, session_id: str, messages: List[dict]) -> None:
        self._messages[self._key(user_id, session_id)] = list(messages)

    def get_session_messages(self, user_id: str, session_id: str) -> List[dict]:
        return self.load(user_id, session_id)

    def list_session_infos(self, user_id: str) -> List[dict]:
        prefix = f"{user_id}:"
        session_ids = [k.split(":", 1)[1] for k in self._messages.keys() if k.startswith(prefix)]
        out = []
        for sid in session_ids:
            records = self.load(user_id, sid)
            updated_at = records[-1]["timestamp"] if records else datetime.utcnow().isoformat()
            out.append(
                {
                    "session_id": sid,
                    "updated_at": updated_at,
                    "message_count": len(records),
                }
            )
        return sorted(out, key=lambda x: x["updated_at"], reverse=True)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        key = self._key(user_id, session_id)
        if key not in self._messages:
            return False
        del self._messages[key]
        return True


class LawAgent:
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek API。")

        self.model = DEEPSEEK_MODEL
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.legal_system_prompt = (
            "你是专业的中国法律助手，严格依据提供的法条回答问题，严禁编造任何法律内容。"
            "回答法律问题必须分4步，每步单独一行："
            "1.核心分析: 简要分析用户问题涉及的法律核心点；"
            "2.检索过程：说明检索到的相关法律及条款；"
            "3.推理过程：结合法条分析用户问题的法律逻辑；"
            "4.最终结论：给出明确、合规的法律结论。"
        )
        self.general_system_prompt = "你是友好的聊天助手，请自然、简洁地回答用户。"

    def _chat_once(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def _stream_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = ""
            if chunk.choices:
                delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    def _build_legal_prompt(self, user_input: str, retrieve_summary: str, context: str) -> str:
        return (
            f"用户问题：{user_input}\n"
            f"相关法条摘要：\n{retrieve_summary}\n"
            f"完整法律条文：\n{context}\n"
            "请严格按照以下4步格式回答：\n"
            "1.核心分析:\n"
            "2.检索过程：\n"
            "3.推理过程：\n"
            "4.最终结论："
        )

    def chat_once(self, user_input: str) -> dict:
        rag_result = rag_search(query_text=user_input, limit=8, catalog_top_n=12)
        rag_trace = rag_result.get("rag_trace")

        if not rag_result.get("is_legal", False):
            prompt = f"用户说：{user_input}\n请友好、简洁地回答。"
            return {
                "response": self._chat_once(self.general_system_prompt, prompt, temperature=0.7),
                "rag_trace": rag_trace,
            }

        citations = rag_result.get("citations", [])
        context = rag_result.get("context", "")
        retrieve_summary = rag_result.get("retrieve_summary", "")
        if not context.strip():
            return {
                "response": "未检索到相关法律条文，请确认问题描述或检查本地向量库。",
                "rag_trace": rag_trace,
                "citations": citations,
            }

        prompt = self._build_legal_prompt(user_input, retrieve_summary, context)
        return {
            "response": self._chat_once(self.legal_system_prompt, prompt, temperature=0.1),
            "rag_trace": rag_trace,
            "citations": citations,
        }

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        rag_result = rag_search(query_text=user_input, limit=8, catalog_top_n=12)
        rag_trace = rag_result.get("rag_trace")

        if not rag_result.get("is_legal", False):
            prompt = f"用户说：{user_input}\n请友好、简洁地回答。"
            for chunk in self._stream_chat(self.general_system_prompt, prompt, temperature=0.7):
                yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
            if rag_trace:
                yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        citations = rag_result.get("citations", [])
        context = rag_result.get("context", "")
        retrieve_summary = rag_result.get("retrieve_summary", "")
        if not context.strip():
            payload = {
                "type": "content",
                "content": "未检索到相关法律条文，请确认问题描述或检查本地向量库。",
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if rag_trace:
                yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace}, ensure_ascii=False)}\n\n"
            if citations:
                yield f"data: {json.dumps({'type': 'citations', 'citations': citations}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        prompt = self._build_legal_prompt(user_input, retrieve_summary, context)
        for chunk in self._stream_chat(self.legal_system_prompt, prompt, temperature=0.1):
            yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

        if rag_trace:
            yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace}, ensure_ascii=False)}\n\n"
        if citations:
            yield f"data: {json.dumps({'type': 'citations', 'citations': citations}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


agent = LawAgent()
storage = ConversationStorage()


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    messages = storage.load(user_id, session_id)
    messages.append({"type": "human", "content": user_text, "timestamp": datetime.utcnow().isoformat()})

    result = agent.chat_once(user_text)
    messages.append(
        {
            "type": "ai",
            "content": result.get("response", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "rag_trace": result.get("rag_trace"),
        }
    )
    storage.save(user_id, session_id, messages)
    return {"response": result.get("response", ""), "rag_trace": result.get("rag_trace")}


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
):
    messages = storage.load(user_id, session_id)
    messages.append({"type": "human", "content": user_text, "timestamp": datetime.utcnow().isoformat()})

    full_response = ""
    rag_trace = None
    async for event in agent.chat_stream(user_text):
        if event.startswith("data: ") and not event.startswith("data: [DONE]"):
            payload = event[6:].strip()
            try:
                parsed = json.loads(payload)
                if parsed.get("type") == "content":
                    full_response += parsed.get("content", "")
                if parsed.get("type") == "trace":
                    rag_trace = parsed.get("rag_trace")
            except json.JSONDecodeError:
                pass
        yield event

    messages.append(
        {
            "type": "ai",
            "content": full_response,
            "timestamp": datetime.utcnow().isoformat(),
            "rag_trace": rag_trace,
        }
    )
    storage.save(user_id, session_id, messages)
