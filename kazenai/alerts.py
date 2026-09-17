from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WebhookAlert:
    url: str
    timeout_s: float = 2.0
    bearer_token: str = ""


class AlertDispatcher:
    def __init__(self, *, webhook: Optional[WebhookAlert] = None) -> None:
        self._webhook = webhook

    def notify(self, *, kind: str, payload: Dict[str, Any]) -> None:
        if not self._webhook or not self._webhook.url:
            return
        try:
            body = json.dumps({"kind": kind, "payload": payload}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                self._webhook.url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self._webhook.bearer_token}"} if self._webhook.bearer_token else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=float(self._webhook.timeout_s)) as resp:
                _ = resp.read(1)
        except Exception:
            return

