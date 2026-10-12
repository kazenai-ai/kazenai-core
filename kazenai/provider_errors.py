"""Classify provider errors for settlement: did the provider do billable work?"""

from __future__ import annotations

# 4xx statuses that do NOT prove the provider skipped the work: a timeout
# (408), a conflict (409) and a client-closed request (499) can follow billing.
_AMBIGUOUS_4XX = frozenset({408, 409, 499})


def is_definitive_provider_rejection(exc: BaseException) -> bool:
    """True when the provider refused the request before doing billable work.

    HTTP 4xx from the provider SDK (bad request, auth, permission, model not
    found, unprocessable, rate limited) means nothing was generated, so the
    reservation settles at exactly $0 instead of being held as pending.
    Timeouts, connection errors and 5xx stay pending: the provider may have
    done the work.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        return False
    try:
        code = int(status)
    except (TypeError, ValueError):
        return False
    return 400 <= code < 500 and code not in _AMBIGUOUS_4XX
