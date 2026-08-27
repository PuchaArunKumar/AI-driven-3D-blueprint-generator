# Demo assets

These are **real outputs of the pipeline**, not mock-ups. Each was produced by
`backend/scripts/seed_demo.py`, which runs the same code path the Design Studio
uses: prompt → specification → multi-view images → silhouette carving → mesh
cleanup → export.

They are committed so that anyone can clone the repository and immediately open,
rotate, inspect the blueprint and download a model with no GPU, no API key and
no provider configured. The UI labels every one of them **Demo Asset** so they
are never mistaken for something the visitor just generated.

Layout per project:

```
<project-uuid>/
  images/    the generated views the reconstruction consumed
  exports/   glb, obj, stl, ply, blend
  model.glb            raw reconstruction
  model_processed.glb  after mesh cleanup - this is what the viewer loads
```

Regenerate them with:

```bash
cd backend
python scripts/seed_demo.py --force
```

Quality reflects the local `sd-turbo` checkpoint used to make them. A larger
image model or a hosted provider produces cleaner views and, in turn, cleaner
geometry.
