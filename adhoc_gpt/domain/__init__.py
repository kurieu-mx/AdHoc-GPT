"""Phase 3: domain specialization -- diplomacy, resolutions and debate.

Imports are lazy so ``python -m adhoc_gpt.domain.corpus`` does not double-import
the submodule.
"""

__all__ = ["build_corpus", "TOPICS", "DOC_KINDS"]


def __getattr__(name):
    if name in __all__:
        from . import corpus

        return getattr(corpus, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
