# syntax=docker/dockerfile:1
# CodeRL sandbox image: runs untrusted, model-generated Python against unit tests.
# Build context is coderl/sandbox/ (only the three stdlib-only files below are needed):
#   docker build -f docker/sandbox.Dockerfile -t coderl-sandbox coderl/sandbox
# The Docker backend tags images by a hash of these files and builds them on demand.
# python:3.11-slim is multi-arch (linux/amd64 + linux/arm64).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Stdlib only: drop the packaging toolchain the solutions have no business touching.
RUN python -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true; \
    rm -rf /root/.cache /usr/local/lib/python3.11/ensurepip; \
    groupadd --gid 10001 sandbox; \
    useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
            --shell /usr/sbin/nologin sandbox

COPY harness.py runner.py server.py /opt/coderl/
RUN chown -R root:root /opt/coderl && chmod 0555 /opt/coderl && chmod 0444 /opt/coderl/*.py

# Non-root. The backend additionally runs containers read-only, without network or
# capabilities, with no-new-privileges and cgroup memory/CPU/pids limits.
USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-S", "/opt/coderl/server.py"]
