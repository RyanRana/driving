"""Process-wide JAX startup config.

The dominant cost of launching any JAX entry point (train/render) is twofold:
the ~1.5s import of the JAX/XLA stack, and re-tracing + re-compiling every jitted
function from scratch on each fresh process. The import is unavoidable, but the
recompile is not: JAX ships a persistent on-disk cache keyed by the compiled
program, so the SECOND run of the same shapes reuses the cached XLA kernels.

Call `enable_compile_cache()` once, early, before the first jitted call. It's
idempotent and safe to call from multiple modules.
"""
from __future__ import annotations

import os

import jax

# Override with SMOOTHRIDE_JAX_CACHE; defaults to <repo>/jax_cache (gitignored).
_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jax_cache"))
_enabled = False


def enable_compile_cache(path: str | None = None) -> None:
    """Point JAX at a persistent compilation cache. Idempotent."""
    global _enabled
    if _enabled:
        return
    cache_dir = os.path.abspath(path or os.environ.get("SMOOTHRIDE_JAX_CACHE", _DEFAULT))
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache everything: don't skip small/fast compiles (defaults exclude them).
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    _enabled = True
