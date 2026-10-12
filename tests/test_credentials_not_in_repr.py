"""Credentials carried on SDK objects must not appear in repr/str (logs, tracebacks)."""

from kazenai.sinks import HttpSinkConfig
from kazenai.spine.guard import ReservationHandle

SECRET = "kz_live_SYNTHETIC_never_print_me"


def test_reservation_handle_repr_hides_finops_api_key():
    handle = ReservationHandle(reserved_cost_usd=0.01, reservation_id="r-1", finops_api_key=SECRET)
    assert SECRET not in repr(handle)
    assert SECRET not in str(handle)
    assert "r-1" in repr(handle)
    assert handle.finops_api_key == SECRET


def test_http_sink_config_repr_hides_api_key():
    cfg = HttpSinkConfig(ingest_url="https://finops.example/v1/events", api_key=SECRET, start_workers=False)
    assert SECRET not in repr(cfg)
    assert "finops.example" in repr(cfg)
    assert cfg.api_key == SECRET
