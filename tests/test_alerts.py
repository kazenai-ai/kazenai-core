"""AlertDispatcher webhook tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kazenai.alerts import AlertDispatcher, WebhookAlert


def test_notify_no_webhook_is_noop():
    AlertDispatcher().notify(kind="finops.loop.anomaly", payload={"run_id": "r1"})


def test_notify_posts_webhook():
    dispatcher = AlertDispatcher(webhook=WebhookAlert(url="http://127.0.0.1:9/hook", bearer_token="tok"))
    with patch("urllib.request.urlopen") as urlopen:
        resp = MagicMock()
        resp.read.return_value = b"ok"
        urlopen.return_value.__enter__.return_value = resp
        dispatcher.notify(kind="finops.circuit_breaker.opened", payload={"run_id": "r1"})
        urlopen.assert_called_once()


def test_notify_swallows_webhook_errors():
    dispatcher = AlertDispatcher(webhook=WebhookAlert(url="http://127.0.0.1:9/hook"))
    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        dispatcher.notify(kind="finops.loop.anomaly", payload={"run_id": "r1"})
