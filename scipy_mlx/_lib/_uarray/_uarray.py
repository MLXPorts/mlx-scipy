"""
Pure-Python MLX-friendly stub of SciPy's vendored `uarray` core.

Upstream SciPy vendors `uarray`, which includes a compiled extension providing
backend dispatch. This repository targets a single runtime backend (MLX),
so we provide a minimal implementation that supports import-time needs and
falls back to multimethod defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple


class BackendNotImplementedError(RuntimeError):
    pass


@dataclass
class _BackendState:
    # Minimal state container to satisfy pickling hooks in `_backend.py`.
    globals: Tuple[Any, ...] = ()
    registered: Tuple[Any, ...] = ()
    skip: Tuple[str, ...] = ()

    def _pickle(self):
        return (self.globals, self.registered, self.skip)

    @classmethod
    def _unpickle(cls, globals_, registered_, skip_):
        return cls(tuple(globals_), tuple(registered_), tuple(skip_))


_STATE = _BackendState()
_GLOBAL_BACKEND = None
_REGISTERED_BACKENDS = []


class _SetBackendContext:
    def __init__(self, backend=None, coerce: bool = False, only: bool = False, try_last: bool = False):
        self.backend = backend
        self.coerce = coerce
        self.only = only
        self.try_last = try_last
        self._old = None

    def __enter__(self):
        global _STATE
        self._old = _STATE
        # We don't implement real dispatch; keep a breadcrumb for debugging.
        _STATE = _BackendState(
            globals=_STATE.globals + ((self.backend, self.coerce, self.only, self.try_last),),
            registered=_STATE.registered,
            skip=_STATE.skip,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        global _STATE
        if self._old is not None:
            _STATE = self._old
        return False

    def _pickle(self):
        return (self.backend, self.coerce, self.only, self.try_last, self._old._pickle() if self._old else None)


class _SkipBackendContext:
    def __init__(self, domain: str):
        self.domain = domain
        self._old = None

    def __enter__(self):
        global _STATE
        self._old = _STATE
        _STATE = _BackendState(
            globals=_STATE.globals,
            registered=_STATE.registered,
            skip=_STATE.skip + (self.domain,),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        global _STATE
        if self._old is not None:
            _STATE = self._old
        return False

    def _pickle(self):
        return (self.domain, self._old._pickle() if self._old else None)


class _Function:
    def __init__(
        self,
        argument_extractor: Callable[..., Sequence[Any]],
        argument_replacer: Callable[[Tuple[Any, ...], dict, Sequence[Any]], Tuple[Tuple[Any, ...], dict]],
        domain: str,
        arg_defaults: Any = None,
        kw_defaults: Any = None,
        default: Optional[Callable[..., Any]] = None,
    ):
        self.argument_extractor = argument_extractor
        self.argument_replacer = argument_replacer
        self.domain = domain
        self.arg_defaults = arg_defaults
        self.kw_defaults = kw_defaults
        self._default = default

    def __call__(self, *args, **kwargs):
        # Minimal dispatch: try the selected backend and fall back to the default.
        dispatchables = self.argument_extractor(*args, **kwargs)
        if not isinstance(dispatchables, (tuple, list)):
            dispatchables = (dispatchables,)

        backend = determine_backend(self.domain, dispatchables, coerce=False)
        if backend is not None:
            convert = getattr(backend, "__ua_convert__", None)
            if convert is None:
                converted = [getattr(d, "value", d) for d in dispatchables]
            else:
                converted = convert(dispatchables, False)
                if converted is NotImplemented:
                    converted = [getattr(d, "value", d) for d in dispatchables]

            new_args, new_kwargs = self.argument_replacer(args, kwargs, converted)
            fn = getattr(backend, "__ua_function__", None)
            if fn is not None:
                res = fn(self, new_args, new_kwargs)
                if res is not NotImplemented:
                    return res

        if self._default is None:
            raise BackendNotImplementedError(
                f"No uarray backend registered for domain {self.domain!r} and no default provided."
            )
        return self._default(*args, **kwargs)


def get_state() -> _BackendState:
    return _STATE


def set_state(state: _BackendState, _restore: bool = False) -> None:
    global _STATE
    _STATE = state


def set_global_backend(backend, coerce: bool, only: bool, try_last: bool) -> None:
    global _GLOBAL_BACKEND
    _GLOBAL_BACKEND = backend
    return None


def register_backend(backend) -> None:
    _REGISTERED_BACKENDS.append(backend)
    return None


def clear_backends(domain: str, registered: bool, globals: bool) -> None:
    global _GLOBAL_BACKEND, _REGISTERED_BACKENDS
    if globals:
        _GLOBAL_BACKEND = None
    if registered:
        _REGISTERED_BACKENDS = []
    return None


def determine_backend(domain: str, dispatchables: Iterable[Any], coerce: bool):
    # Very small subset: honor skip list, then prefer context-local backend,
    # then the global backend, then registered ones.
    if domain in _STATE.skip:
        return None
    if _STATE.globals:
        backend = _STATE.globals[-1][0]
        if getattr(backend, "__ua_domain__", None) == domain:
            return backend
    if _GLOBAL_BACKEND is not None and getattr(_GLOBAL_BACKEND, "__ua_domain__", None) == domain:
        return _GLOBAL_BACKEND
    for backend in _REGISTERED_BACKENDS:
        if getattr(backend, "__ua_domain__", None) == domain:
            return backend
    return None


def determine_backend_multi(domain: str, dispatchables: Iterable[Any], coerce: bool):
    backend = determine_backend(domain, dispatchables, coerce)
    return [backend] if backend is not None else []
