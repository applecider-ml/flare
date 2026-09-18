# FLARE inference image for batch/pipeline photometric classification (e.g. OSG).
# The model bank is baked in so jobs run fully offline (execute nodes have no
# egress). The Rust crate and the training/figure extras are left out — this is
# the LightGBM inference path the SkyPortal analysis service calls.

FROM python:3.11-slim-bookworm

# libgomp1 for LightGBM's OpenMP runtime; everything else ships manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src
COPY . .

# matplotlib for the light-curve plot, requests for the analysis-callback upload;
# neither is a core flare dependency but the SkyPortal job needs both. Then bake
# the models at PKG_ROOT/models (the parent of the installed flare package), where
# config.py resolves them, and drop the build context.
RUN pip install --upgrade pip \
    && pip install . matplotlib requests \
    && PKG="$(cd / && python -c 'import flare, pathlib; print(pathlib.Path(flare.__file__).resolve().parents[1])')" \
    && mkdir -p "$PKG/models" \
    && cp -r models/. "$PKG/models/" \
    && rm -rf /src

# Fail the build if the runtime isn't importable or the baked models don't load.
RUN python -c "from flare.hierarchical import HierarchicalFlare; from flare.config import BTS6; HierarchicalFlare.from_pretrained(BTS6); print('flare ok')"

WORKDIR /work
CMD ["flare", "--help"]
