# Architecture

How the pieces fit together, and where to change things.

## Layers

```
┌──────────────────────────────────────────────────────────────────┐
│ frontend/  React + TypeScript + Three.js (Vite)                  │
│   pages/       Home · Studio · Generate · Gallery                │
│                ProjectDetail · Settings · About                  │
│   components/  Viewer3D · BlueprintView · PipelineStatus         │
│                SpecEditor · ViewGallery · ExportPanel            │
│   lib/api.ts   typed client + watchJob() (WebSocket → polling);  │
│                in static mode, reads a snapshot instead          │
│   lib/text2voxel/  in-browser text-to-3D: Web Worker running     │
│                onnxruntime-web; calls no API                     │
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
| `TextTo3DGenerator` | `generate_from_text(prompt, output_dir, *, spec, seed, progress)` | one `ModelResult` (a GLB), from text alone |
| `MeshProcessor` | `process(model_path, output_dir, …)` | a cleaned `ModelResult` |
| `CADExporter` | `export(model_path, output_path, fmt)` | the written path |

Every provider also implements `availability() -> Availability`, which is what makes
the "never offer what cannot run" rule enforceable. `ProviderRegistry` is the only
place providers are constructed; `available_export_formats()` derives the UI's format
list from live availability, so a format is offered only when something can produce it.

`TextTo3DGenerator` is a `ThreeDGenerator` subclass, so it lives in the same registry
slot and the same `THREED_PROVIDER` setting. The orchestrator checks which kind it
resolved: `generate_3d` does not ask a text provider for images, and
`run_full_pipeline` marks the images stage skipped, with the reason, instead of
generating views nothing would read. `text2voxel` is the only text provider.

With `THREED_PROVIDER=auto` the registry walks `THREED_PREFERENCE` (TripoSR, Shap-E,
visual hull) and takes the first available one. `text2voxel` is deliberately not in
that list. The exception is a caller with no views to reconstruct from
(`allow_text_fallback`) whose image provider cannot run: then `auto` resolves to
`TEXT_FALLBACK` (`text2voxel`) if it is available, and logs it, because an
image-to-3D backend would have nothing to consume.

### Adding a provider

1. Subclass the relevant interface in `providers/<kind>/`.
2. Implement `availability()` honestly — return `Availability.missing(reason, requires)`
   with an actionable reason.
3. Register it in `ProviderRegistry.__init__`.
4. Add its name to the `Literal` in `config.py`.

Nothing else changes: the orchestrator, API and UI are provider-agnostic. A 3D
provider that works from text rather than images subclasses `TextTo3DGenerator`, and
the orchestrator's text path picks it up from that.

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

Text2Voxel-64 was trained in the same frame, so its output needs no rotation. Its
64³ volumes are C-ordered `[x][y][z]` (flat index `(x·64 + y)·64 + z`, colour channel
outermost), voxel *i* spans `[i, i+1)/64` of the unit cube, and the surface sits at
occupancy 0.5. Meshes are centred on X and Z with their base at Y = 0 before scaling.

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

The browser's Text → 3D page draws its blueprint from the voxel grid rather than a
mesh, but produces the same `BlueprintData` shape (outline / inline / hidden
polylines in metres, per view, on the axes of `blueprint.VIEW_AXES`), so
`BlueprintView` renders both. Its A3 sheet comes from a TypeScript port of `blueprint_sheet.py`.

## Text-to-3D (Text2Voxel-64)

One model, two runtimes, one contract. The model is described in
[MODEL_CARD.md](MODEL_CARD.md) and trained by
[`training/text2voxel/`](../training/text2voxel/README.md).

```
prompt → MiniLM (int8 ONNX) → 384-d embedding
       → 50-step DDIM with classifier-free guidance over prior.onnx → 256-d latent
       → decoder.onnx → occupancy [64³] + rgb [3×64³] → surface at 0.5 → mesh
       → uniform scale from the requested dimensions → GLB
```

| | Browser | Backend |
| --- | --- | --- |
| Code | `frontend/src/lib/text2voxel/` | `backend/app/providers/threed/text2voxel.py` |
| Runtime | onnxruntime-web, WASM backend, in a module Web Worker | onnxruntime, CPU |
| Tokeniser | WordPiece in TypeScript, reading `tokenizer.json` | the `tokenizers` package |
| Surface | marching cubes in TypeScript; ambiguous cube faces are resolved by the asymptotic decider so neighbouring cells agree and the mesh is closed; per-vertex colour | scikit-image marching cubes on the padded grid; occupancy-weighted trilinear vertex colour |
| Output | three.js geometry → GLB / STL / OBJ / PLY, blueprint from voxels | GLB via trimesh, then the normal mesh / CAD / export stages |
| Model files | `frontend/public/models/{text2voxel,minilm}/`, fetched at `BASE_URL/models/…` and kept in Cache Storage | `backend/models/text2voxel/` (+ `minilm/`) |

Both runtimes prepare the grid the same way before surfacing it (`prepareField` in
`mesher.ts`, `prepare_field` in the provider): 26-connected fragments under 8 voxels
are dropped, the largest part always kept, and when fewer than 64 voxels reach the
threshold - the untrained placeholder does this - the surface is taken at the lower
level that encloses 4096 voxels and the result carries a warning saying so. ONNX
Runtime's WASM binary is imported with Vite's `?url`, so the build emits it once
under `BASE_URL/assets/` next to the page; there is no CDN and no copy step.

**The contract is `meta.json`**, written by K2 next to the ONNX files. It carries the
full `alphas_cumprod` table, the sampler defaults (50 steps, guidance 3.0, x0 clipped
to ±6, zero vector as the unconditional input), the text-encoder pin and pooling,
the occupancy threshold and a `reference` block: a prompt, fixed Gaussian noise, the
first 8 values of that prompt's embedding and the latent K2's own sampler produced.
Both runtimes must reproduce that latent from that noise (tolerance 2e-2), which
catches any drift in tokenisation, pooling, timestep rounding or guidance. The
checks are `node frontend/scripts/verify-text2voxel.mjs` and
`backend/tests/test_text2voxel.py`. The seeded RNGs differ between runtimes, so the
same seed gives the same result only within one runtime.

**Distribution.** Model files are never committed. `models/manifest.json` lists each
file with its URL (the `text2voxel-v1` GitHub release for the decoder, prior and
`meta.json`; Hugging Face at a pinned revision for MiniLM) and a `sha256`, which is
`null` for the three release files until the trained model is published. `frontend/scripts/fetch-models.mjs` and
`backend/scripts/install_text2voxel.py` both read it and verify any recorded
checksum. A development placeholder with `"smoke": true` in `meta.json` has the same
formats; the UI flags it and the backend reports it in the result metadata.

## Prompt understanding

`pipeline/prompt_parser.py` extracts the specification with rules: deterministic,
offline, instant and easy to unit-test. The research proposed BERT/RoBERTa here; the
repository uses a smaller BERT-family encoder, as a refinement layer rather than the
parser itself.

To substitute a model, keep the signature:

```python
def parse_prompt(prompt: str) -> DesignSpec: ...
```

`pipeline/semantic.py` sits between the rules and the rest of the pipeline;
`parse_prompt()` calls it after the rules (unless `SEMANTIC_PROMPT=false`). It
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

Two copies of MiniLM exist and are never mixed. `semantic.py` loads the PyTorch
`sentence-transformers/all-MiniLM-L6-v2` checkpoint through `transformers`.
Text2Voxel uses Xenova's int8 ONNX export of the same model at a pinned revision,
because its embeddings have to match the ones it was trained on bit for bit, in a
browser, without PyTorch. The text-to-3D path does not go through `DesignSpec` for
its conditioning: it embeds the prompt text and reads only the dimensions from the
spec (the browser parses them with a TypeScript port of the same rules).

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

Demo projects are committed as files under `storage/demo/<uuid>/`, but the database
is not. On startup the backend therefore registers every demo in the catalogue whose
files are on disk and whose row is missing: the spec comes from `parse_prompt`, the
statistics from the mesh itself, and images and exports are discovered on disk. It is
idempotent, so a fresh clone shows the demos without running `seed_demo.py`.

## Static mode (GitHub Pages)

A build with `VITE_STATIC=true` has no backend. `lib/api.ts` keeps its public
interface but serves every read from a snapshot under `${BASE_URL}static-api/`, and
every mutation throws an `ApiError` whose hint says the hosted demo has no backend
and points at running it locally. `watchJob` is never reached, because no job can be
started. Studio and Settings render an explanation instead of erroring.

The snapshot is written by `backend/scripts/export_static_site.py`, which runs the
real blueprint and blueprint-sheet code over the demo catalogue and needs none of
the heavy dependencies:

```
static-api/projects.json                       ProjectSummary[] (demos only)
static-api/projects/<id>.json                  Project
static-api/projects/<id>/blueprint.json        BlueprintData
static-api/projects/<id>/blueprint-<theme>.svg combined sheet, white and blueprint
static-api/files/demo/<id>/...                 images, model_processed.glb, exports
```

Every URL inside the snapshot is relative to the site root with no leading slash,
and the client resolves it against `BASE_URL`, so the same snapshot works at `/` and
under `/AI-driven-3D-blueprint-generator/`. Vite's `base` comes from `VITE_BASE`,
the router's `basename` follows it, and `frontend/scripts/postbuild.mjs` copies
`index.html` to `404.html` so GitHub Pages serves the app for deep links. The
Text → 3D page needs nothing from the snapshot. `.github/workflows/pages.yml` runs
all of this on every push to `main`; the README's GitHub Pages section lists the
steps.

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
- Text2Voxel inference runs in a module Web Worker, so a generation never blocks the
  UI; progress (stage, fraction, downloaded bytes) is posted back to the page. The
  ONNX Runtime WASM files are served from the site's own origin rather than a CDN,
  and model downloads are cached with the Cache Storage API when the browser allows
  it. The generated GLB is handed to the page as an object URL that the page revokes
  when the result is discarded.
