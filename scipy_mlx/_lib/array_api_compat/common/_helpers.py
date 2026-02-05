from __future__ import annotations

from types import ModuleType


def _compat_module_name(xp: ModuleType) -> str:
    # Best-effort module name for capability tables / doc generation.
    return getattr(xp, "__name__", xp.__class__.__name__)

