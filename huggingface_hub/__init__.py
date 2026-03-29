"""Minimal local stub for environments without huggingface_hub.

This project does not rely on snapshot_download at runtime,
but hello_agents imports it transitively during package import.
"""


def snapshot_download(*args, **kwargs):
    raise RuntimeError(
        "huggingface_hub is unavailable in this environment. "
        "snapshot_download was called unexpectedly."
    )
