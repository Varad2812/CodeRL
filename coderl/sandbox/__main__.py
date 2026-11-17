"""Sandbox utilities: ``python -m coderl.sandbox {build-image,cleanup,info}``."""

from __future__ import annotations

import json
import sys
from typing import Annotated

import typer

from coderl.utils.logging import get_logger

log = get_logger(__name__)
app = typer.Typer(add_completion=False, help="CodeRL sandbox utilities.")


@app.command("build-image")
def build_image_cmd() -> None:
    """Build the Docker sandbox image (tagged by content hash and :latest)."""
    from coderl.sandbox.backends.docker_backend import build_image

    log.info("sandbox_image_ready", tag=build_image())


@app.command()
def cleanup(
    all_owners: Annotated[
        bool, typer.Option("--all", help="Also remove containers of live processes.")
    ] = False,
) -> None:
    """Remove leftover sandbox containers (by label)."""
    from coderl.sandbox.backends.docker_backend import cleanup_containers

    log.info("sandbox_cleanup", removed=cleanup_containers(all_owners=all_owners))


@app.command()
def info() -> None:
    """Show which backend would be used and what isolation is available."""
    from coderl.sandbox.backends import default_backend_name
    from coderl.sandbox.backends.docker_backend import docker_available, image_tag
    from coderl.sandbox.backends.subprocess_backend import detect_isolation

    data = {
        "platform": sys.platform,
        "default_backend": default_backend_name(),
        "subprocess_isolation": detect_isolation()[0],
        "docker_available": docker_available(),
        "docker_image": image_tag(),
    }
    sys.stdout.write(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    app()
