# FLARE LightGBM inference image; the model bank is baked in so jobs run offline.

FROM python:3.11-slim-bookworm

# libgomp1 for LightGBM's OpenMP runtime; everything else ships manylinux wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src
COPY . .

# [skyportal] = requests for the analysis-callback upload + matplotlib for the PNG; then bake the models at PKG_ROOT/models where config.py resolves them.
RUN pip install --upgrade pip \
    && pip install ".[skyportal]" \
    && PKG="$(cd / && python -c 'import flare, pathlib; print(pathlib.Path(flare.__file__).resolve().parents[1])')" \
    && mkdir -p "$PKG/models" \
    && cp -r models/. "$PKG/models/" \
    && rm -rf /src

# Fail the build if the runtime isn't importable or the baked models don't load.
RUN python -c "import flare.skyportal; from flare.hierarchical import HierarchicalFlare; from flare.config import BTS6; HierarchicalFlare.from_pretrained(BTS6); print('flare ok')"

WORKDIR /work
CMD ["flare", "--help"]
