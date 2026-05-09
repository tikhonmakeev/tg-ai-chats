from __future__ import annotations

import logging
import os
from pathlib import Path

import faiss
import numpy as np

import ai
import db
from config import settings

logger = logging.getLogger(__name__)


class FAISSManager:
    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._indices: dict[tuple[str, int], faiss.Index] = {}
        self._dimension: int | None = None

    def _index_path(self, connection_id: str, chat_id: int) -> Path:
        safe_name = f"{connection_id}_{chat_id}.index"
        return self._data_dir / safe_name

    def _load_or_create(self, connection_id: str, chat_id: int) -> faiss.Index:
        key = (connection_id, chat_id)
        if key in self._indices:
            return self._indices[key]

        path = self._index_path(connection_id, chat_id)
        if path.exists():
            index = faiss.read_index(str(path))
            logger.info("Loaded FAISS index from %s (%d vectors)", path, index.ntotal)
        else:
            dim = self._dimension or 1536
            inner = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(inner)

        self._indices[key] = index
        return index

    def add(
        self,
        connection_id: str,
        chat_id: int,
        message_id: int,
        embedding: list[float],
    ) -> None:
        index = self._load_or_create(connection_id, chat_id)
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        if self._dimension is None:
            self._dimension = vec.shape[1]

        ids = np.array([message_id], dtype=np.int64)
        index.add_with_ids(vec, ids)
        self.save(connection_id, chat_id)

    def search(
        self,
        connection_id: str,
        chat_id: int,
        query_embedding: list[float],
        top_k: int,
        exclude_ids: set[int] | None = None,
    ) -> list[int]:
        index = self._load_or_create(connection_id, chat_id)
        if index.ntotal == 0:
            return []

        search_k = top_k + len(exclude_ids or set())
        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        distances, ids = index.search(vec, min(search_k, index.ntotal))

        result = []
        for msg_id in ids[0]:
            if msg_id == -1:
                continue
            if exclude_ids and int(msg_id) in exclude_ids:
                continue
            result.append(int(msg_id))
            if len(result) >= top_k:
                break
        return result

    def save(self, connection_id: str, chat_id: int) -> None:
        key = (connection_id, chat_id)
        index = self._indices.get(key)
        if index is None:
            return
        path = self._index_path(connection_id, chat_id)
        faiss.write_index(index, str(path))

    def save_all(self) -> None:
        for (conn_id, chat_id) in list(self._indices):
            self.save(conn_id, chat_id)
        logger.info("Saved %d FAISS indices", len(self._indices))


faiss_manager = FAISSManager(settings.faiss_data_dir)


async def build_context(
    connection_id: str, chat_id: int, query_text: str
) -> tuple[str, list[dict[str, str]]]:
    """Returns (system_prompt, messages) — provider-agnostic format."""
    recent = await db.get_recent_messages(
        connection_id, chat_id, settings.recent_messages
    )
    recent_ids = {m["id"] for m in recent}

    rag_messages = await _get_rag_messages(
        connection_id, chat_id, query_text, recent_ids
    )

    system = settings.system_prompt
    if rag_messages:
        rag_text = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in rag_messages
        )
        system += f"\n\nRelevant earlier messages from this conversation:\n{rag_text}"

    messages: list[dict[str, str]] = []
    for m in recent:
        messages.append({"role": m["role"], "content": m["content"]})

    return system, messages


async def _get_rag_messages(
    connection_id: str,
    chat_id: int,
    query_text: str,
    exclude_ids: set[int],
) -> list[dict]:
    try:
        query_embedding = await ai.get_embedding(query_text)
    except Exception as e:
        logger.warning("Failed to get embedding for RAG query: %s", e)
        return []

    found_ids = faiss_manager.search(
        connection_id,
        chat_id,
        query_embedding,
        settings.rag_top_k,
        exclude_ids=exclude_ids,
    )
    if not found_ids:
        return []

    return await db.get_messages_by_ids(found_ids)
