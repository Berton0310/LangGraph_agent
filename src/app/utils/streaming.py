import json
from typing import Any, Dict, Optional


def sse_event(data: Dict[str, Any], event: Optional[str] = None, id: Optional[str] = None) -> str:
    """Format a dict as a Server-Sent Event string.

    The output follows the SSE format:
        event: <event-name>\n
        id: <id>\n
        data: <json-string>\n
        \n
    Only 'data' is required; 'event' and 'id' are optional.
    """
    lines = []
    if event:
        lines.append(f"event: {event}")
    if id:
        lines.append(f"id: {id}")
    lines.append("data: " + json.dumps(data, ensure_ascii=False))
    lines.append("")  # blank line to terminate the event
    return "\n".join(lines)


def sse_error(message: str) -> str:
    """Helper to format an error event."""
    return sse_event({"type": "error", "message": message}, event="error")
