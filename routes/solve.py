"""POST /solve endpoint for dependency solving with SSE streaming."""

import json
from typing import Optional, AsyncIterable
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.sse import ServerSentEvent
from pydantic import ValidationError

from models import SolveRequest, ManifestType, ErrorEvent
from solver_wrapper import solve_manifest

router = APIRouter()


@router.post("/solve", response_class=StreamingResponse)
async def solve(
    file: Optional[UploadFile] = File(None),
    data: Optional[str] = Form(None),
) -> StreamingResponse:
    """
    Solve dependencies and stream progress events via SSE.

    Accepts EITHER:
    - multipart/form-data with file upload
    - application/x-www-form-urlencoded with 'data' field (JSON string)

    Returns an EventSource stream with events:
    - parsing: manifest being parsed
    - solving: constraint solving in progress
    - lockfile: lockfile being generated
    - complete: final result (conflicts, solution, lockfile, solver_time)
    - error: solver error (error message, optional suggestion)

    Args:
        file: Optional uploaded manifest file
        data: Optional JSON-encoded form data with manifest_text and manifest_type

    Returns:
        StreamingResponse with SSE events
    """

    async def event_generator() -> AsyncIterable[str]:
        """Generate SSE events for the solve operation."""
        try:
            # Parse input
            if file:
                try:
                    file_content = await file.read()
                    manifest_text = file_content.decode("utf-8")
                except Exception as e:
                    error_event = ErrorEvent(
                        error=f"Failed to read file: {str(e)}",
                        suggestion="Ensure file is UTF-8 encoded text"
                    )
                    yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
                    return

                # Infer manifest type from filename
                if file.filename:
                    if "environment" in file.filename.lower() and file.filename.endswith(".yml"):
                        manifest_type = ManifestType.ENVIRONMENT_YML
                    elif file.filename.endswith(".toml"):
                        manifest_type = ManifestType.PYPROJECT_TOML
                    else:
                        manifest_type = ManifestType.REQUIREMENTS_TXT
                else:
                    manifest_type = ManifestType.REQUIREMENTS_TXT

            elif data:
                try:
                    payload = json.loads(data)
                    manifest_text = payload.get("manifest_text", "")
                    manifest_type_str = payload.get("manifest_type", "requirements.txt")
                    manifest_type = ManifestType(manifest_type_str)
                except (json.JSONDecodeError, ValueError) as e:
                    error_event = ErrorEvent(
                        error=f"Invalid request data: {str(e)}",
                        suggestion="Ensure data is valid JSON with manifest_text and manifest_type"
                    )
                    yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
                    return

            else:
                error_event = ErrorEvent(
                    error="No file or text data provided",
                    suggestion="Upload a manifest file or provide manifest text as form data"
                )
                yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
                return

            # Validate input
            try:
                SolveRequest(
                    manifest_text=manifest_text,
                    manifest_type=manifest_type,
                )
            except ValidationError as e:
                error_event = ErrorEvent(
                    error="Invalid manifest input",
                    suggestion=str(e)
                )
                yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"
                return

            # Stream events from solver
            async for event in solve_manifest(manifest_text, manifest_type.value):
                if isinstance(event, ServerSentEvent):
                    # Format as SSE
                    event_str = f"event: {event.event}\ndata: {event.data}\n\n"
                    yield event_str
                else:
                    # Shouldn't happen, but handle string events
                    yield event

        except Exception as e:
            error_event = ErrorEvent(
                error=f"Unexpected error: {str(e)}",
                suggestion="Check logs for details"
            )
            yield f"event: error\ndata: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
