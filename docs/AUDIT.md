# Repository audit

Audit of [`PuchaArunKumar/AI-driven-3D-blueprint-generator`](https://github.com/PuchaArunKumar/AI-driven-3D-blueprint-generator)
at commit `15734d8` ("Add files via upload"), performed before any code was changed.
The original files are preserved verbatim under [`research/original/`](../research/original/).

## What the repository contained

Eleven tracked files, one commit, no history.

| File | Size | Assessment |
| --- | --- | --- |
| `README.md` | 2.1 KB | Feature list |
| `Readme.md` | 1.9 KB | A **second, different** README |
| `main.ipynb` | 24 KB | The only real pipeline code — 5 code cells |
| `flowchart.ipynb` | **0 bytes** | Empty file |
| `flowchart` | 244 B | Graphviz DOT source, 5 nodes |
| `research` | **1 byte** | A single newline — not a directory |
| `generate_model.py` | 812 B | Blender script |
| `ops.py` | 5.8 KB | **Verbatim copy of Blender's internal `bpy/ops.py`** |
| `requirements.txt` | 261 B | 10 unpinned packages |
| `future_work.txt` | 174 B | Three lines of notes |
| `__init__.py` | 0 bytes | Empty |

There was no web application, no API, no database, no tests, no `.env` handling and no
frontend of any kind.

## Defects found

### The documented pipeline did not exist

Both READMEs describe **Text → Image → 3D → GLB → BLEND**. Nothing in the repository
produced a GLB, and nothing produced multi-view images. The notebook's "3D" step was a
single-image extrusion (below).

### `main.ipynb`

1. **Cell 3 — image generation never completed.** The cell requests
   `torch_dtype=torch.float16` and then calls `.to("cpu")`. fp16 on CPU is
   pathologically slow, and the committed cell output proves it never finished:

   ```
   2%|▏         | 1/50 [1:14:14<60:38:14, 4454.98s/it]
   ```

   74 minutes for 1 of 50 steps — a projected 60-hour runtime. It also passes the
   deprecated `revision="fp16"`, which diffusers warns about in the same output.

2. **Cell 5 — this is not 3D reconstruction.** The "voxelization" thresholds one
   grayscale image and then does:

   ```python
   voxel_grid = np.stack([binary_image for _ in range(64)], axis=2)
   ```

   That is a linear extrusion of a single 2-D silhouette — a cookie-cutter prism.
   Marching cubes over it recovers no depth information whatsoever. There is no SLAT,
   no NeRF and no Gaussian splatting anywhere in the repository, despite the research
   naming all three.

3. **Cell 7 — guaranteed `TypeError`.** The cell reads:

   ```python
   voxel_grid = ...          # literally the Ellipsis object
   verts, faces, _, _ = measure.marching_cubes(voxel_grid, level=0)
   ```

   `measure` is also not imported in that cell.

4. **Cell 9 — `import FreeCAD`** only resolves inside FreeCAD's bundled interpreter.

5. **Cell 11 — removed API.** `bpy.ops.import_mesh.stl` was **removed in Blender 4.0**
   (replaced by `bpy.ops.wm.stl_import`).

### `generate_model.py`

- The "generation" is a hardcoded lookup: `"chair"` → a squashed cube, `"table"` → a
  flatter cube, anything else → a cube. No AI is involved at any point.
- Hardcoded Colab output path `/content/generated_model.obj`.
- `bpy.ops.export_scene.obj` was **removed in Blender 4.0** (now `bpy.ops.wm.obj_export`).

### `ops.py`

Not project code. It is a verbatim copy of Blender's own `bpy/ops.py` module, carrying
its `SPDX-FileCopyrightText: 2009-2023 Blender Authors` / `GPL-2.0-or-later` header. It
imports `_bpy`, which exists only inside the Blender process, so importing it in the
project raises `ImportError`. It was removed.

### `requirements.txt`

Ten unpinned packages. Missing `open3d`, `Pillow`, `streamlit` and `FreeCAD` despite
all four being claimed in the READMEs. No `fastapi`, `pytest` or anything else needed
for an application.

### Documentation vs. reality

The READMEs claimed BERT/RoBERTa prompt understanding, 3D-GAN, DreamFusion, CLIPMesh,
MeshCNN, OpenSCAD and a Streamlit UI. **None of these appeared anywhere in the code.**
`README.md` also embeds a placeholder image link, `https://your-image-link-if-any.com`.

Two READMEs differing only in case (`README.md` / `Readme.md`) collide on
case-insensitive filesystems; `git clone` on Windows warns and checks out only one.

## What was kept

- The **pipeline concept and terminology** — Text → Image → 3D → GLB → BLEND, and the
  five flowchart stages (prompt understanding, image generation, image-to-3D, CAD
  integration, rendering) — now map onto the seven implemented pipeline stages.
- The **research motivation**: accessible, personalised product design for people
  underserved by mass-produced goods.
- The **application areas**: assistive technology, custom automotive/bike design,
  furniture, wearables, AR/VR assets, rapid prototyping.
- The **technology intent**: Stable Diffusion, Blender, FreeCAD, PyTorch and trimesh
  are all now genuinely wired up.
- `future_work.txt` — its three items (CAD export, better voxel optimisation,
  real-time visualisation) are all now implemented.

## What replaced the broken parts

| Original | Replacement |
| --- | --- |
| fp16-on-CPU diffusion that never finished | `DiffusersImageProvider` — device auto-detection, fp16 variant on CUDA, attention/VAE slicing, OOM handled as a readable error |
| Single-image extrusion called "voxelization" | `VisualHullProvider` — multi-view silhouette segmentation, cross-view axis alignment, voxel carving, marching cubes, colour projection |
| `voxel_grid = ...` (`Ellipsis`) | A working orchestrator that persists real geometry between stages |
| Hardcoded cube "generator" | The real pipeline; Blender is used for scene repair and `.blend` output, not for inventing geometry |
| Blender 3.x operators | Blender 4.x/5.x operator names, verified against Blender 5.2.0 LTS |
| `import FreeCAD` in a notebook | `FreeCADProvider` driving `FreeCADCmd` as a subprocess, reporting unavailability cleanly |
| Vendored Blender `ops.py` | Deleted |
| Claims of unimplemented tech | A per-technology status table in the About page and README, marking what is and is not implemented |

## Verification performed

- The original `main.ipynb` cells were read along with their committed outputs, which
  is how the 60-hour runtime and the never-completed generation were established.
- Blender 5.2.0 LTS was driven end to end against the new script, producing a real
  `.blend` (112 KB), `.obj` + `.mtl` and `.stl` from a test GLB.
- `import ops` was confirmed to fail outside Blender.
