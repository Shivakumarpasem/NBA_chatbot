"""
Caller-layer wrapper: run a tool and return a standard shape.
Use this when calling nba_data functions (e.g. from API or tests).
Do not change the tool implementations; wrap at the call site.
"""
from typing import Any, Callable, Dict, Optional


def safe_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Call fn(*args, **kwargs). Return {"ok": True, "data": result, "error": None}
    on success, or {"ok": False, "data": None, "error": "<message>"} on failure.

    Inputs:
        fn: any callable (e.g. player_summary, top_stat_leaderboard).
        *args, **kwargs: passed through to fn.

    Outputs:
        {"ok": True, "data": <result>, "error": None}  on success
        {"ok": False, "data": None, "error": "<why>"}  on exception
    """
    try:
        result = fn(*args, **kwargs)
        return {"ok": True, "data": result, "error": None}
    except Exception as e:
        return {"ok": False, "data": None, "error": str(e)}
