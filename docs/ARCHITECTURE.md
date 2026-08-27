# Architecture

How the pieces fit together, and where to change things.

## Layers

```
┌──────────────────────────────────────────────────────────────────┐
│ frontend/  React + TypeScript + Three.js (Vite)                  │
│   pages/       Home · Studio · Gallery · ProjectDetail           │
│                Settings · About                                  │
│   components/  Viewer3D · BlueprintView · PipelineStatus         │
│                SpecEditor · ViewGallery · ExportPanel            │
│   lib/api.ts   typed client + watchJob() (WebSocket → polling)   │
└───────────────────────────┬──────────────────────────────────────┘
                            │  REST  +  WS /api/ws/jobs/{id}
┌───────────────────────────▼──────────────────────────────────────┐
│ backend/app/                                                     │
│                                                                  │
│  api/          thin routers: validate → submit job → return 202  │
│       ▼                                                          │
│  jobs.py       JobManager: thread pool, stages, progress,        │
│                cancellation, subscriber fan-out                  │
│       ▼                                                          │
│  pipeline/orchestrator.py                                        │
│                one function per stage; persists after each       │
│       ▼                                                          │
│  providers/    abstract interfaces + registry                    │
│       ▼                                                          │
│  storage.py    repository + asset layout + path safety           │
│  db.py         SQLAlchemy engine/session/ORM                     │
└──────────────────────────────────────────────────────────────────┘
```

## Request lifecycle

A long-running action never blocks a request:

1. The router validates the body with a Pydantic model and confirms the project exists.
2. It calls `manager.submit(kind, worker, project_id=…, stages=…)`, which returns a
   `Job` immediately with HTTP **202**.
3. `JobManager` runs `worker` on a `ThreadPoolExecutor` (generation is CPU/GPU-bound
   synchronous work, so it must stay off the event loop).
4. The worker reports progress through `JobContext`, which updates the record and
   pushes it to every WebSocket subscriber via `loop.call_soon_threadsafe`.
5. The frontend's `watchJob()` opens `/api/ws/jobs/{id}` and falls back to polling
   `/api/jobs/{id}` if the socket cannot be established.

Cancellation is cooperative: `JobContext.check_cancelled()` raises `JobCancelled` at
stage boundaries and inside provider loops.

## Provider interfaces

Defined in `providers/base.py`:

| Interface | Method | Contract |
| --- | --- | --- |
| `ImageGenerator` | `generate(spec, views, output_dir, …)` | one `ImageResult` per view |
| `ThreeDGenerator` | `generate(images, output_dir, …)` | one `ModelResult` (a GLB) |
| `MeshProcessor` | `process(model_path, output_dir, …)` | a cleaned `ModelResult` |
| `CADExporter` | `export(model_path, output_path, fmt)` | the written path |

Every provider also implements `availability() -> Availability`, which is what makes
the "never offer what cannot run" rule enforceable. `ProviderRegistry` is the only
place providers are constructed; `available_export_formats()` derives the UI's format
list from live availability, so a format is offered only when something can produce it.

### Adding a provider

1. Subclass the relevant interface in `providers/<kind>/`.
2. Implement `availability()` honestly — return `Availability.missing(reason, requires)`
   with an actionable reason.
3. Register it in `ProviderRegistry.__init__`.
4. Add its name to the `Literal` in `config.py`.

Nothing else changes: the orchestrator, API and UI are provider-agnostic.

## Coordinate frame

glTF convention throughout: **+X right, +Y up, +Z toward the front**, units in metres.

`visual_hull.VIEW_AXES` maps each view's image `(column, row)` onto world axes with a
sign, and `VIEW_NORMALS` gives each camera's outward direction for colour projection.
These two tables are the whole camera model — change them together.

`carve()` combines the back-projected silhouettes by *voting* rather than strict
intersection once there are four or more views: a voxel survives if all but one
view contains it. Text-to-image models reliably produce the occasional unusable
view, and a strict AND lets that one frame delete the entire model.

The blueprint planes follow from the same frame: `front` = (X, Y), `side` = (Z, Y),
`top` = (X, Z).

## Technical drawings

`pipeline/blueprint.py` projects the mesh into line art rather than filled
silhouettes. Three families come out of the face adjacency:

| family | definition | drawn as |
| --- | --- | --- |
| `outline` | view contour - one face towards the camera, one away | heavy continuous |
| `inline` | crease - sharp dihedral angle, both faces towards the camera | light continuous |
| `hidden` | either of the above, occluded by the body | dashed |

Visibility comes from a depth buffer rasterised per triangle over its own
bounding box, so hidden-line removal is a real occlusion test.

Two details matter for correctness:

- The **outer** profile is traced from a raster silhouette, not chained from
  contour edges. Chaining produces hundreds of fragments; tracing gives smooth
  closed loops.
- Drawing runs on a decimated, lightly smoothed copy (smoothing only above
  `SMOOTHING_FACE_THRESHOLD`, since it rounds the corners off a clean low-poly
  mesh). **Quoted dimensions always come from the original mesh**, never that
  copy - otherwise the sheet would report shrunken sizes.

`pipeline/blueprint_sheet.py` composes the three views onto one A3 sheet in
third-angle projection. The plan shares the front elevation's horizontal
placement and the side elevation shares its baseline, which is what makes the
sheet a projection rather than three unrelated pictures. The SVG declares
physical `mm` dimensions, so the scale ratio in the title block is real.

## Prompt understanding

`pipeline/prompt_parser.py` is deliberately rule-based: deterministic, offline,
instant and easy to unit-test. The research proposed BERT/RoBERTa here.

To substitute a model, keep the signature:

```python
def parse_prompt(prompt: str) -> DesignSpec: ...
```

`pipeline/semantic.py` sits between the rules and the rest of the pipeline. It
uses MiniLM (a distilled BERT) to restructure the object phrase and to match
vocabularies by meaning. Two design rules keep it safe:

- it **augments**, never overrides, a confident rule match;
- it **declines** rather than guesses - an unlisted object keeps its own words.

Scoring deliberately blends embeddings with a character ratio. Measured on the
failure that motivated it: for "electric wheel cchair", pure cosine ranks
*wheel* (0.57) above *electric wheelchair* (0.49); the blend reverses that to
0.50 vs 0.71.

Everything downstream consumes `DesignSpec`, not text. `spec_to_base_prompt()` builds
the shared identity clause and `build_view_prompt()` appends the per-view camera
phrase — that pairing is what keeps the views recognisably the same object, so a
replacement parser should keep filling the same fields.

## Persistence

`db.py` holds one table, `projects`, whose variable-shaped fields (`design_spec`,
`images`, `exports`, `stats`, `history`, `extra`) are `JSON` columns. That keeps the
schema stable as the pipeline evolves while remaining portable — pointing
`DATABASE_URL` at PostgreSQL requires no code change.

`storage.py` owns the on-disk layout and is the single boundary for path safety:

```
storage/
  models/<project-uuid>/         images/  exports/  model.glb  model_processed.glb
  demo/<project-uuid>/           same layout, is_demo = true
  logs/backend.log
  projects.db
```

`validate_project_id()` rejects anything that is not a canonical UUID *before* it can
reach a path, and `ensure_within()` re-checks any path served to a client.

## Error handling

`errors.py` defines the exception hierarchy. Each carries a `user_message` safe for the
UI and an optional `hint` naming the remedy. `main.py` registers handlers that return

```json
{ "code": "...", "message": "...", "hint": "..." }
```

and log the full traceback to `storage/logs/backend.log`. Job failures carry the same
`message` and `hint` through the `Job` schema, so the Studio can show the remedy for a
failure that happened minutes after the request returned.

A bare `Exception` handler catches anything unanticipated and returns a generic message
— a stack trace never reaches the browser.

## Security boundaries

- **Path traversal** — UUID validation plus `ensure_within()` on every served path.
- **Uploads** — content-type and extension allowlist, size cap, and the image is
  re-encoded through Pillow (`verify()` then `convert("RGB")`), so the stored file is
  one this process wrote. The filename comes from the view enum, never from the upload.
- **Subprocesses** — Blender and FreeCAD are invoked with an argument list and
  `shell=False`. No user text is ever interpolated into a command line.
- **Secrets** — read from the environment, exposed only as `*_set: bool`.
- **CORS** — an explicit origin allowlist from `CORS_ORIGINS`.

## Frontend notes

- `Viewer3D` clones the loaded scene before applying display modes, so drei's GLB cache
  is never mutated. A `ModelErrorBoundary` catches parse failures and renders the
  reason instead of blanking the canvas.
- Environment lighting is built from in-scene `Lightformer`s rather than drei's
  `preset`, which would fetch an HDRI from a CDN and break the offline workflow.
- Model URLs are cache-busted with `?v={updated_at}` so a regenerated model at the same
  path is refetched.
