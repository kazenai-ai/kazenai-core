"""P3-1: pin the deliberate monitor-function/submodule shadowing behavior.

kazenai/__init__.py re-exports the `monitor()` function under the same name as
the `kazenai.monitor` submodule (public SDK API — cannot rename). These tests
pin the access patterns that must keep working so the ambiguity stays
understood rather than rediscovered.
"""

from __future__ import annotations

import importlib
import types


def test_kazenai_monitor_attribute_is_the_function():
    import kazenai

    assert callable(kazenai.monitor)
    assert not isinstance(kazenai.monitor, types.ModuleType)


def test_from_import_still_reaches_submodule_members():
    from kazenai.monitor import patch_openai  # resolved via sys.modules

    assert callable(patch_openai)


def test_importlib_returns_the_real_submodule():
    mod = importlib.import_module("kazenai.monitor")
    assert isinstance(mod, types.ModuleType)
    assert hasattr(mod, "_wrap_stream_iterator")
