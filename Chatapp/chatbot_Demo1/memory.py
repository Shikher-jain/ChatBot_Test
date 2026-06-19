from __future__ import annotations

from threading import RLock


_CHAT_MEMORY: dict[str, list[dict[str, str]]] = {}
_LOCK = RLock()
_MAX_MESSAGES_PER_USER = 20


def get_history(user_id: str) -> list[dict[str, str]]:
    with _LOCK:
        return list(_CHAT_MEMORY.get(user_id, []))


def update_history(user_id: str, role: str, content: str) -> None:
    message = {"role": role, "content": content}

    with _LOCK:
        history = _CHAT_MEMORY.setdefault(user_id, [])
        history.append(message)

        if len(history) > _MAX_MESSAGES_PER_USER:
            del history[: len(history) - _MAX_MESSAGES_PER_USER]


def clear_history(user_id: str) -> None:
    with _LOCK:
        _CHAT_MEMORY.pop(user_id, None)