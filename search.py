
"""RAG search engine for Chinese law QA (backend-like pipeline)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json

import jieba
from openai import OpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
)


@dataclass
class LawSearchResult:
    chunk_id: str
    doc_id: str
    law_title: str
    tiao: str
    content: str
    semantic_score: float
    publish_date: Optional[str] = ""
    effective_date: Optional[str] = ""
    rrf_rank: Optional[int] = None


class _LLMResponse:
    def __init__(self, content: str):
        self.content = content


class DeepSeekLLM:
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek API。")
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.model = DEEPSEEK_MODEL

    def invoke(self, prompt: str) -> _LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=False,
        )
        return _LLMResponse(resp.choices[0].message.content or "")


def _extract_json_obj(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return {}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}


def _tokenize_text(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in jieba.lcut(text.lower()) if len(token.strip()) >= 2]


def _keyword_match_score(query_text: str, candidate_text: str) -> float:
    query_tokens = set(_tokenize_text(query_text))
    candidate_tokens = set(_tokenize_text(candidate_text))
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / max(1, len(query_tokens))


def get_embedding() -> OllamaEmbeddings:
    return OllamaEmbeddings(base_url=EMBEDDING_BASE_URL, model=EMBEDDING_MODEL)


def get_llm() -> DeepSeekLLM:
    return DeepSeekLLM()


def get_collection_name(doc_id: str) -> str:
    return f"law_{hashlib.md5(doc_id.encode('utf-8')).hexdigest()}"


def is_legal_question_by_ai(query_text: str) -> bool:
    llm = get_llm()
    prompt = f"""
你是分类器。请判断用户问题是否属于中国法律咨询。
只输出JSON，禁止额外文本：
{{"is_legal": true/false}}

用户问题：{query_text}
"""
    try:
        resp = llm.invoke(prompt)
        payload = _extract_json_obj(getattr(resp, "content", str(resp)))
        return bool(payload.get("is_legal", False))
    except Exception:
        return True


def search_law_catalog(query_text: str, top_n: int = 12) -> List[Dict[str, Any]]:
    catalog_db = Chroma(
        collection_name="law_catalog",
        embedding_function=get_embedding(),
        persist_directory=VECTOR_DB_DIR,
    )
    docs = catalog_db.similarity_search_with_score(query=query_text, k=top_n)
    results = []
    for doc, score in docs:
        results.append(
            {
                "doc_id": doc.metadata.get("doc_id", ""),
                "law_title": doc.metadata.get("law_title", ""),
                "law_type": doc.metadata.get("law_type", ""),
                "score": float(score),
            }
        )
    return results


def select_most_relevant_law_by_ai(query_text: str, catalog_matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not catalog_matches:
        return None
    if len(catalog_matches) == 1:
        return catalog_matches[0]

    llm = get_llm()
    options = []
    for idx, item in enumerate(catalog_matches, 1):
        options.append(
            {
                "index": idx,
                "law_title": item.get("law_title", ""),
                "law_type": item.get("law_type", ""),
            }
        )

    prompt = f"""
你是中国法律检索路由器。请根据用户问题，从候选法律中选择最相关的一部。
只输出JSON，禁止额外文本：
{{"index": 数字}}

用户问题：{query_text}
候选法律：
{json.dumps(options, ensure_ascii=False, indent=2)}
"""
    try:
        resp = llm.invoke(prompt)
        payload = _extract_json_obj(getattr(resp, "content", str(resp)))
        pick = int(payload.get("index", 1))
        if 1 <= pick <= len(catalog_matches):
            return catalog_matches[pick - 1]
    except Exception:
        pass
    return catalog_matches[0]


def _rerank_documents(query_text: str, docs: List[LawSearchResult], top_k: int) -> Tuple[List[LawSearchResult], Dict[str, Any]]:
    rescored = []
    for idx, doc in enumerate(docs, 1):
        keyword_sim = _keyword_match_score(query_text, f"{doc.law_title} {doc.tiao} {doc.content}")
        semantic_sim = max(0.0, 1.0 - doc.semantic_score)
        final_sim = 0.78 * semantic_sim + 0.22 * keyword_sim
        rescored.append(
            LawSearchResult(
                chunk_id=doc.chunk_id,
                doc_id=doc.doc_id,
                law_title=doc.law_title,
                tiao=doc.tiao,
                content=doc.content,
                semantic_score=1.0 - final_sim,
                publish_date=doc.publish_date,
                effective_date=doc.effective_date,
                rrf_rank=idx,
            )
        )
    reranked = sorted(rescored, key=lambda x: x.semantic_score)[:top_k]
    return reranked, {
        "rerank_enabled": True,
        "rerank_applied": True,
        "candidate_count": len(docs),
    }


def _auto_merge_documents(docs: List[LawSearchResult], top_k: int) -> Tuple[List[LawSearchResult], Dict[str, Any]]:
    merged: Dict[str, LawSearchResult] = {}
    replaced = 0
    for doc in docs:
        key = f"{doc.law_title}_{doc.tiao}".strip("_")
        if key in merged:
            replaced += 1
        if key not in merged or doc.semantic_score < merged[key].semantic_score:
            merged[key] = doc
    out = sorted(merged.values(), key=lambda x: x.semantic_score)[:top_k]
    return out, {
        "auto_merge_enabled": True,
        "auto_merge_applied": replaced > 0,
        "auto_merge_threshold": 2,
        "auto_merge_replaced_chunks": replaced,
        "auto_merge_steps": 1 if replaced > 0 else 0,
    }


def retrieve_documents_by_law(query_text: str, law_match: Dict[str, Any], limit: int = 10) -> Tuple[List[LawSearchResult], Dict[str, Any]]:
    if not law_match or not law_match.get("doc_id"):
        return [], {}

    doc_id = law_match["doc_id"]
    law_title = law_match.get("law_title", "")
    law_db = Chroma(
        collection_name=get_collection_name(doc_id),
        embedding_function=get_embedding(),
        persist_directory=VECTOR_DB_DIR,
    )

    candidate_k = max(limit * 3, 24)
    raw_docs = law_db.similarity_search_with_score(query=query_text, k=candidate_k)
    candidates = []
    for doc, distance in raw_docs:
        md = doc.metadata
        semantic_sim = 1.0 / (1.0 + max(0.0, float(distance)))
        candidates.append(
            LawSearchResult(
                chunk_id=doc.id,
                doc_id=md.get("doc_id", doc_id),
                law_title=md.get("law_title", law_title),
                tiao=md.get("tiao", ""),
                content=doc.page_content,
                semantic_score=1.0 - semantic_sim,
                publish_date=md.get("publish_date", ""),
                effective_date=md.get("effective_date", ""),
            )
        )

    reranked, rerank_meta = _rerank_documents(query_text=query_text, docs=candidates, top_k=limit)
    merged, merge_meta = _auto_merge_documents(docs=reranked, top_k=limit)
    meta = {
        "retrieval_mode": "dense",
        "candidate_k": candidate_k,
        "leaf_retrieve_level": 3,
        **rerank_meta,
        **merge_meta,
    }
    return merged, meta


def grade_documents(question: str, docs: List[LawSearchResult]) -> str:
    if not docs:
        return "no"
    llm = get_llm()
    context = "\n\n".join(item.content[:300] for item in docs[:3])
    prompt = f"""
你是相关性评估器。判断检索到的文档是否能回答用户问题。
只输出 JSON:
{{"binary_score":"yes|no"}}

用户问题：{question}
检索上下文：
{context}
"""
    try:
        resp = llm.invoke(prompt)
        payload = _extract_json_obj(getattr(resp, "content", str(resp)))
        score = str(payload.get("binary_score", "no")).strip().lower()
        return "yes" if score == "yes" else "no"
    except Exception:
        return "no"


def choose_rewrite_strategy(query_text: str) -> str:
    llm = get_llm()
    prompt = f"""
请根据用户问题选择最合适的查询扩展策略，只输出JSON：
{{"strategy":"step_back|hyde|complex"}}

用户问题：{query_text}
"""
    try:
        resp = llm.invoke(prompt)
        payload = _extract_json_obj(getattr(resp, "content", str(resp)))
        strategy = str(payload.get("strategy", "step_back")).strip().lower()
        if strategy in {"step_back", "hyde", "complex"}:
            return strategy
    except Exception:
        pass
    return "step_back"


def step_back_expand(query_text: str) -> Dict[str, str]:
    llm = get_llm()
    try:
        q_resp = llm.invoke(
            f"请将用户问题抽象成更通用的退步问题，只输出一句话。\n用户问题：{query_text}"
        )
        step_back_question = (getattr(q_resp, "content", str(q_resp)) or "").strip()
    except Exception:
        step_back_question = ""

    try:
        a_resp = llm.invoke(
            f"请简要回答以下退步问题，控制在120字以内，仅输出答案。\n退步问题：{step_back_question or query_text}"
        )
        step_back_answer = (getattr(a_resp, "content", str(a_resp)) or "").strip()
    except Exception:
        step_back_answer = ""

    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query_text}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query_text
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def generate_hypothetical_document(query_text: str) -> str:
    llm = get_llm()
    prompt = f"""
请基于用户问题生成一段“假设性文档”，用于检索增强。
要求紧扣问题语义，只输出正文，不要解释。
用户问题：{query_text}
"""
    try:
        resp = llm.invoke(prompt)
        return (getattr(resp, "content", str(resp)) or "").strip()
    except Exception:
        return ""


def _build_rag_payload(results: List[LawSearchResult]) -> Dict[str, Any]:
    context = "\n\n".join([item.content for item in results])
    summary_lines = []
    for i, item in enumerate(results, 1):
        short_content = item.content[:150] + "..." if len(item.content) > 150 else item.content
        summary_lines.append(f"{i}. {item.law_title} {item.tiao}：{short_content}")
    citations = sorted({f"{item.law_title} {item.tiao}".strip() for item in results if item.law_title or item.tiao})
    return {
        "context": context,
        "retrieve_summary": "\n".join(summary_lines),
        "citations": citations,
    }


def rag_search(query_text: str, limit: int = 10, catalog_top_n: int = 12) -> Dict[str, Any]:
    is_legal = is_legal_question_by_ai(query_text)
    if not is_legal:
        return {
            "is_legal": False,
            "law_match": None,
            "results": [],
            "context": "",
            "retrieve_summary": "",
            "citations": [],
            "rag_trace": {
                "retrieval_stage": "skipped",
                "reason": "non_legal_question",
            },
        }

    catalog_matches = search_law_catalog(query_text, top_n=catalog_top_n)
    selected_law = select_most_relevant_law_by_ai(query_text, catalog_matches)
    if not selected_law:
        return {
            "is_legal": True,
            "law_match": None,
            "results": [],
            "context": "",
            "retrieve_summary": "",
            "citations": [],
            "rag_trace": {
                "retrieval_stage": "initial",
                "reason": "catalog_empty",
            },
        }

    initial_results, initial_meta = retrieve_documents_by_law(query_text, selected_law, limit=limit)
    grade_score = grade_documents(query_text, initial_results)
    route = "generate_answer" if grade_score == "yes" else "rewrite_question"

    final_results = initial_results
    final_meta = dict(initial_meta)
    expansion_type = None
    expanded_query = query_text
    step_back_question = ""
    step_back_answer = ""
    hypothetical_doc = ""

    if route == "rewrite_question":
        expansion_type = choose_rewrite_strategy(query_text)
        expanded_candidates: List[LawSearchResult] = []
        expanded_meta_list: List[Dict[str, Any]] = []

        if expansion_type in ("step_back", "complex"):
            step_back_payload = step_back_expand(query_text)
            step_back_question = step_back_payload.get("step_back_question", "")
            step_back_answer = step_back_payload.get("step_back_answer", "")
            expanded_query = step_back_payload.get("expanded_query", query_text) or query_text
            step_results, step_meta = retrieve_documents_by_law(expanded_query, selected_law, limit=limit)
            expanded_candidates.extend(step_results)
            expanded_meta_list.append(step_meta)

        if expansion_type in ("hyde", "complex"):
            hypothetical_doc = generate_hypothetical_document(query_text)
            hyde_query = hypothetical_doc or query_text
            hyde_results, hyde_meta = retrieve_documents_by_law(hyde_query, selected_law, limit=limit)
            expanded_candidates.extend(hyde_results)
            expanded_meta_list.append(hyde_meta)

        combined = initial_results + expanded_candidates
        deduped, merge_meta = _auto_merge_documents(combined, top_k=limit)
        for idx, item in enumerate(deduped, 1):
            item.rrf_rank = idx
        final_results = deduped
        final_meta = dict(initial_meta)
        if expanded_meta_list:
            final_meta["rerank_applied"] = any(bool(m.get("rerank_applied")) for m in expanded_meta_list)
            final_meta["candidate_k"] = max([int(initial_meta.get("candidate_k", 0))] + [int(m.get("candidate_k", 0)) for m in expanded_meta_list])
        final_meta.update(merge_meta)

    payload = _build_rag_payload(final_results)
    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query_text,
        "expanded_query": expanded_query,
        "retrieval_stage": "expanded" if expansion_type else "initial",
        "expansion_type": expansion_type,
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "hypothetical_doc": hypothetical_doc,
        "grade_score": grade_score,
        "grade_route": route,
        "rewrite_needed": route == "rewrite_question",
        "initial_hit_count": len(initial_results),
        "final_hit_count": len(final_results),
        "retrieved_chunks": [
            {"law_title": r.law_title, "tiao": r.tiao, "text": r.content, "score": r.semantic_score, "rrf_rank": r.rrf_rank}
            for r in final_results
        ],
        "rerank_enabled": final_meta.get("rerank_enabled"),
        "rerank_applied": final_meta.get("rerank_applied"),
        "retrieval_mode": final_meta.get("retrieval_mode"),
        "candidate_k": final_meta.get("candidate_k"),
        "leaf_retrieve_level": final_meta.get("leaf_retrieve_level"),
        "auto_merge_enabled": final_meta.get("auto_merge_enabled"),
        "auto_merge_applied": final_meta.get("auto_merge_applied"),
        "auto_merge_threshold": final_meta.get("auto_merge_threshold"),
        "auto_merge_replaced_chunks": final_meta.get("auto_merge_replaced_chunks"),
        "auto_merge_steps": final_meta.get("auto_merge_steps"),
    }

    return {
        "is_legal": True,
        "law_match": selected_law,
        "results": final_results,
        "context": payload["context"],
        "retrieve_summary": payload["retrieve_summary"],
        "citations": payload["citations"],
        "rag_trace": rag_trace,
    }


def semantic_search(
    query_text: str,
    domains: Optional[List[str]] = None,
    limit: int = 10,
    catalog_top_n: int = 12,
) -> List[LawSearchResult]:
    _ = domains
    out = rag_search(query_text=query_text, limit=limit, catalog_top_n=catalog_top_n)
    return out["results"] if out.get("is_legal") else []


def search_by_tiao(doc_id: Optional[str] = None, law_title: Optional[str] = None, tiao: str = "") -> Optional[LawSearchResult]:
    if not tiao or (not doc_id and not law_title):
        return None

    embeddings = get_embedding()
    if law_title and not doc_id:
        catalog_db = Chroma(
            collection_name="law_catalog",
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_DIR,
        )
        catalog_docs = catalog_db.similarity_search(query=law_title, k=1)
        if not catalog_docs:
            return None
        doc_id = catalog_docs[0].metadata["doc_id"]

    law_db = Chroma(
        collection_name=get_collection_name(doc_id),
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    tiao_docs = law_db.get(where={"tiao": tiao}, include=["metadatas", "documents", "ids"])
    if not tiao_docs["ids"]:
        return None

    metadata = tiao_docs["metadatas"][0]
    return LawSearchResult(
        chunk_id=tiao_docs["ids"][0],
        doc_id=metadata.get("doc_id", ""),
        law_title=metadata.get("law_title", ""),
        tiao=metadata.get("tiao", ""),
        content=tiao_docs["documents"][0],
        semantic_score=0.0,
        publish_date=metadata.get("publish_date", ""),
        effective_date=metadata.get("effective_date", ""),
    )