"""urllib requests that carry a credential and never follow a redirect.

``urllib.request.urlopen`` follows 301, 302 and 303 and copies every header except
the body ones onto the new request, ``Authorization`` and ``X-API-Key`` included,
to whatever host the ``Location`` names. The SDK sends its keys only to the URL
the operator configured, so a redirect is an error here (``HTTPError``), never a
second request. The FinOps transport (``spine.transport``) uses httpx, which does
not follow redirects either.
"""

from __future__ import annotations

import urllib.request
from typing import Any


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:  # noqa: ARG002
        # None makes urllib raise HTTPError for the 3xx instead of following it.
        return None


_OPENER = urllib.request.build_opener(_RefuseRedirect)


def urlopen(req: urllib.request.Request, timeout: float) -> Any:
    """``urllib.request.urlopen`` without redirects (a 3xx raises ``HTTPError``)."""
    return _OPENER.open(req, timeout=timeout)
