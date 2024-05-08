from langfuse.callback import CallbackHandler

from typing import Optional, List


def langfuse_handler_from_config(
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    return CallbackHandler(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        version=version,
        release=release,
        tags=tags,
    )
