# Pre-baked scientific sandbox for arxiv-reproducer.
#
# The stack is installed at BUILD time (network available). At RUN time the
# container gets --network none, so agent-generated code cannot exfiltrate
# data or phone home. Extra packages are installed by a separate ephemeral
# container into /workspace/.deps (see sandbox.py), never by this one.
# python:3.12-slim pinned by digest so the sandbox base cannot drift or be
# tag-hijacked. To bump: docker pull python:3.12-slim && docker image inspect
# python:3.12-slim --format '{{index .RepoDigests 0}}'
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    pandas \
    sympy \
    scikit-learn \
    networkx \
    pillow \
    tqdm

# Non-root user; containers are additionally started with --user 1000:1000,
# a read-only root filesystem, and all capabilities dropped.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox

# HOME and matplotlib config point at the writable tmpfs.
ENV HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /workspace
