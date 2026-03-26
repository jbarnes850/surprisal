from collections.abc import Callable


type ProgressCallback = Callable[[str], None]


def emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
