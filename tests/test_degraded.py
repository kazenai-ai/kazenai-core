from kazenai.degraded import clear_degraded, degraded_header_value, mark_degraded


def test_mark_degraded_dedupes_and_serializes() -> None:
    clear_degraded()
    mark_degraded("brain", "lens", "brain")
    assert degraded_header_value() == "brain,lens"
    clear_degraded()
    assert degraded_header_value() == ""
