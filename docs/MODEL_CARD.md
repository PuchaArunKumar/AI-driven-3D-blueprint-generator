# Model card: Text2Voxel-64

A small text-to-3D model trained for this project. It turns an English prompt into
a coloured 64³ voxel shape, which the browser or the backend then meshes, scales and
exports. It is deliberately small enough to run on a laptop CPU inside a web page.

> **Non-commercial research use only.** The model is trained on Text2Shape (whose
> shapes come from ShapeNet) and ModelNet40. ShapeNet's terms of use restrict it to
> non-commercial research and education, and ModelNet40 is provided for academic
> research only. The trained weights inherit those restrictions, and the safe
> reading is to treat the meshes it generates the same way. See
> [Licences](#licences).

> **Released as `text2voxel-v1`** (trained 2026-09-27 on Kaggle, 2 x Tesla T4). The
> metrics below come from that run's `meta.json` and `k2_report.json`. A development
> checkout that has not fetched the release may still hold the **placeholder**: the
> same architecture and file formats, trained for seconds on a few synthetic shapes.
> Its `meta.json` carries `"smoke": true` and the web UI shows an "untrained
> placeholder" banner for it.

| | |
| --- | --- |
| Name / version | Text2Voxel-64, version 1 |
| Type | Text-conditioned latent diffusion prior over the latent space of a 3D shape VAE |
| Input | English text, up to 128 tokens |
| Output | 64 × 64 × 64 occupancy probabilities plus RGB per voxel; meshed downstream |
| Parameters | prior 12.08 M, VAE decoder 4.22 M (the 15.80 M VAE encoder is used only in training) |
| Text encoder | `Xenova/all-MiniLM-L6-v2`, int8 ONNX, pinned revision `751bff37182d3f1213fa05d7196b954e230abad9` |
| Download | `decoder.onnx` ~8.5 MB, `prior.onnx` ~24 MB, MiniLM ~23 MB + `tokenizer.json` 0.7 MB: ~56 MB in total |
| Runtimes | browser: onnxruntime-web (WASM) in a Web Worker; backend: onnxruntime (CPU) |
| Training code | [`training/text2voxel/`](../training/text2voxel/) - see its [README](../training/text2voxel/README.md) |
| Distribution | GitHub release `text2voxel-v1` (decoder, prior, meta.json); MiniLM from Hugging Face |
| Licence | Non-commercial research use only (inherited from the training data) |

The file sizes are fixed by the architecture, so the placeholder and the trained
model are the same size.

---

## How it works

```
prompt ──MiniLM──▶ 384-d embedding ──┐
                                     ▼
seeded N(0, I) noise [256] ──▶ 50 DDIM steps of the prior, guidance 3.0 ──▶ latent [256]
                                                                               │
                                          VAE decoder ◀────────────────────────┘
                                               │
                     occupancy [64³] + RGB [3 × 64³]  (sigmoid probabilities)
                                               │
                  iso-surface at 0.5 ──▶ coloured mesh ──▶ scaled to the prompt's size
```

### Text encoder

`all-MiniLM-L6-v2`, a 6-layer distilled BERT sentence encoder, used as Xenova's int8
ONNX export at a pinned revision so training, the browser and the backend compute the
same embedding. Tokenisation is the model's own `tokenizer.json` (lower-casing BERT
normaliser, WordPiece, `[CLS] … [SEP]`), truncated to 128 tokens. The embedding is
the attention-mask-weighted mean of the last hidden state, L2-normalised
(`TextEncoder.encode` in `common.py`). The encoder is frozen; it is not fine-tuned.

This is a different copy of MiniLM from the one `backend/app/pipeline/semantic.py`
loads through `transformers`. The two are never mixed.

### Shape VAE (stage A)

Trained on occupancy + colour volumes `[4, 64, 64, 64]` (RGB is multiplied by
occupancy, so empty space is black).

| Part | Layers |
| --- | --- |
| Encoder | four stride-2 `Conv3d` (k=4) stages, 4→32→64→128→256 channels, 64³→4³, with a residual block after each of the last three; `Linear(256·4³ → 512)` gives μ and log σ² (clamped to [-12, 6]) |
| Decoder | `Linear(256 → 128·4³)`; three `ConvTranspose3d` (k=4, stride 2) stages 128→96→64→32, 4³→32³, each followed by a residual block; a final `Conv3d` predicts the eight 64³ children of every 32³ cell (4 channels × 8) and a 3D pixel shuffle puts them in place |
| Residual block | GroupNorm → SiLU → `Conv3d` 3³ → GroupNorm → SiLU → `Conv3d` 3³, plus the skip |

Predicting the eight children and shuffling them is about half the cost of a
convolution at 64³, which matters because the browser runs the decoder on the CPU.

### Diffusion prior (stage B)

A residual MLP that denoises **standardised** VAE latents (posterior means, shifted
and scaled per dimension by the mean and standard deviation over the whole training
set), conditioned on the text embedding.

| | |
| --- | --- |
| Width / depth / MLP hidden | 512 / 6 blocks / 1024 |
| Block | LayerNorm (no affine) → adaLN modulation `(1 + scale)·h + shift` → MLP (SiLU) → gated residual. The modulation layer is zero-initialised, so every block starts as the identity. |
| Conditioning | `SiLU(time_mlp(sinusoidal(t, 128)) + cond_mlp(embedding))` drives every block's shift, scale and gate |
| Parameterisation | v-prediction |
| Noise schedule | cosine (s = 0.008), per-step α clipped to [1e-4, 0.9999], 1000 training steps |
| Unconditional branch | an all-zero 384-d embedding (10% of training conditions are dropped to zero) |

### Sampler

Deterministic DDIM (η = 0) with classifier-free guidance, identical in `k2_train.py`
(`ddim_sample`), the browser and the backend:

```
times = round(linspace(999, 0, steps))        # numpy rounding (half to even)
x = seeded standard normal noise [256]
for i, t in enumerate(times):
    a = ᾱ[t];  a_prev = ᾱ[times[i+1]] if i + 1 < len(times) else 1
    v = v_uncond + g · (v_cond − v_uncond)     # the runtimes batch both branches (B = 2)
    x0 = clamp(√a · x − √(1−a) · v, −6, 6)
    eps = √(1−a) · x + √a · v
    x = √a_prev · x0 + √(1−a_prev) · eps
occupancy, rgb = decoder(x)
```

Defaults: 50 steps, guidance g = 3.0. Both are adjustable in the UI and through
`TEXT2VOXEL_STEPS` / `TEXT2VOXEL_GUIDANCE`. The seed makes a result reproducible on
the same runtime.

### From voxels to a mesh

The surface is extracted at `threshold` 0.5 with per-vertex colour sampled from the
RGB volume. The backend uses scikit-image marching cubes; the browser runs its own
marching cubes in the worker, resolving ambiguous cube faces consistently so the
surface is closed. Before that, both drop floating fragments under 8 voxels, and if
fewer than 64 voxels reach 0.5 (as with the untrained placeholder) they surface the
grid at the lower level that encloses 4096 voxels and report a warning. The mesh is centred on x and z with its base at y = 0, then
scaled **uniformly**: to the requested height if one is given, otherwise so that
the larger of its x/z extents matches the larger of the requested length and width,
otherwise so that the largest extent is 1000 mm. The shape is never stretched to
fit all three dimensions.

---

## Files

Identical names in every location (`frontend/public/models/`,
`backend/models/text2voxel/`, the release):

| File | Contents |
| --- | --- |
| `text2voxel/decoder.onnx` | input `z [1,256]` float32 (standardised latent; the graph un-standardises it) → `occupancy [1,64,64,64]`, `rgb [1,3,64,64,64]`, both sigmoid probabilities in 0..1 |
| `text2voxel/prior.onnx` | inputs `x [B,256]`, `t [B]` (float timestep 0..999), `cond [B,384]` → `v [B,256]` |
| `text2voxel/meta.json` | dimensions, frame, the full `alphas_cumprod` table, sampler defaults, text-encoder pin, threshold, the 40 ModelNet classes, the colour palette, metrics, training provenance, export parity and a reference sample (below) |
| `minilm/model_quantized.onnx`, `minilm/tokenizer.json` | the text encoder, straight from Hugging Face at the pinned revision |

**Frame.** glTF convention: x right, y up, z towards the viewer (the object's front
faces +z). The occupancy volume is C-ordered `[x][y][z]`, flat index
`(x·64 + y)·64 + z`; voxel *i* spans `[i, i+1)/64` of the unit cube.

**Precision.** Weights are rounded to fp16 before export and every float initialiser
with at least 1024 elements is stored as fp16 followed by a `Cast` to fp32. That
halves the download; onnxruntime folds the casts when the session is created, so
the arithmetic stays fp32. The metrics below are measured with the same fp16-rounded
weights that ship.

---

## Training data

| Source | What is used | Size (K1 run) | Terms |
| --- | --- | --- | --- |
| [Text2Shape](http://text2shape.stanford.edu/) (Chen et al., 2018) | ShapeNet chairs and tables as 64³ solid coloured voxels (`nrrd_256_filter_div_64_solid.zip`) with crowd-sourced English captions (`captions.tablechair.csv`) | 15,032 shapes (6,589 chairs, 8,443 tables), 75,360 captions | Derived from ShapeNet: ShapeNet terms of use, non-commercial research and education |
| [ModelNet40](https://modelnet.cs.princeton.edu/) (Wu et al., 2015) | 12,311 CAD meshes in 40 categories (both official splits), via the Kaggle mirror `balraj98/modelnet40-princeton-3d-object-dataset` | 12,311 meshes, 84 (bowl) to 989 (chair) per class | Provided for academic research only; the original authors keep copyright of the models |

ModelNet has no captions or colours, so its shapes are paired with template captions
defined in `common.py` (embedded by K1, sampled by K2): 1-4 readable names per class (`"sofa"`, `"couch"`), four plain
templates (`"a sofa"`, `"a 3d model of a sofa"`, …) and three colour templates
(`"a red sofa"`, `"a sofa in red"`, `"red sofa"`) over a 12-colour palette. That is
3,000 template strings. A ModelNet shape is coloured either a neutral CAD grey or
one uniform palette colour, matched to the caption.

**Held-out split.** 5% of shapes, chosen by `crc32(model id or file name) % 20 == 0`
so the split is stable across runs. On the K1 data this holds out 773 Text2Shape
shapes (3,892 captions) and 628 ModelNet shapes. ModelNet's own train/test split is
not used.

### Preprocessing (K1, CPU)

- **Text2Shape**: occupancy is `alpha >= 128`; shapes with fewer than 20 occupied
  voxels are dropped. Colour is stored at 32³ as the mean colour of each cell's
  occupied children and upsampled (nearest) to 64³ in training, so colour detail is
  effectively 32³.
- **ModelNet40**: each OFF mesh is centred and scaled isotropically so its largest
  extent spans 60 of the 64 voxels, surface-sampled (about 16 samples per unit
  voxel area, 20k to 3M points, plus the vertices) into a shell, and the enclosed
  interior filled with `scipy.ndimage.binary_fill_holes`. The K1 run voxelised all
  12,311 meshes with no failures; the median filled/shell ratio was 1.42 (meshes that
  do not enclose a volume stay as shells).
- **Captions**: every distinct caption and every template string is embedded once
  with the pinned MiniLM and stored as float16.
- **Orientation**: arrays keep each source's raw axis order. K2's `ORIENT` table maps
  each source to the canonical frame, chosen from K1's projection montages. Text2Shape's
  NRRD header maps raw axes (0, 1, 2) to ShapeNet world (y, z, x); y is up and chair
  backs sit at +z, so a 180° turn about y gives x = −raw2, y = raw0, z = −raw1.
  ModelNet40 is z-up: x = raw0, y = raw2, z = −raw1. Both are proper rotations, never
  mirrors.
- **Centring**: Text2Shape grids place every object against the minimum corner while
  ModelNet voxelisation is centred, so K2 rolls every shape so its bounding box is
  centred (lossless, the margins are empty).

---

## Training procedure (K2, Kaggle "GPU T4 x2")

Training is time-budgeted: each stage runs for a fixed number of wall-clock minutes,
estimates its total step count from a timing probe early on, and fits its cosine
learning-rate schedule to that estimate.

| | Stage A: VAE | Stage B: prior |
| --- | --- | --- |
| Budget | 110 min | 20 min |
| Batch | 32 (split across both T4s with `DataParallel`) | 1024 (one GPU) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | AdamW, lr 3e-4, weight decay 0.01 |
| Schedule | linear warm-up for 200 steps, then cosine to 1e-5 | linear warm-up for 500 steps, then cosine to 1e-6 |
| Precision | fp16 autocast + gradient scaling | fp32 |
| Gradient clipping | 1.0 | 1.0 |
| Loss | BCE on occupancy (positive weight 2) + 0.5 × (1 − soft IoU) + L1 colour error on occupied voxels + 2e-5 × KL | MSE on v |
| Weights kept | last step | EMA, decay `min(0.9995, (1 + step)/(10 + step))` |
| Steps reached | 10,775 | 39,413 |

**Stage A sampling and augmentation.** Shapes are drawn uniformly from the pooled
training set (about 55% Text2Shape and 45% ModelNet by count). Every shape is
mirrored across x with probability 0.5. A ModelNet shape is neutral grey with
probability 0.5, otherwise one of the 12 palette colours; Text2Shape shapes keep
their own colours.

**Stage B latents.** After stage A, every training shape is encoded once (posterior
mean, not a sample): Text2Shape shapes plain and mirrored, ModelNet shapes plain and
mirrored in each of the 13 colours. These latents are standardised per dimension and
paired with captions as follows, per batch:

| Share | Pairs |
| --- | --- |
| 55% | a Text2Shape shape with one of its human captions |
| 10% | a Text2Shape shape with a plain category caption (`"a chair"`, `"the table"`, …) |
| 35% | a ModelNet shape, classes sampled with weight 1/√(class size); 40% of these use a plain caption with the neutral-grey latent, 60% a colour caption with the matching coloured latent |

Every pair picks the plain or mirrored latent at random, and 10% of conditions are
replaced by the zero embedding for classifier-free guidance.

**Reproducibility.** Budgets are wall-clock, `cudnn.benchmark` is on and the VAE runs
data-parallel, so a re-run does not reproduce the weights bit for bit. What is pinned
is the exported artefact: `meta.json` records a reference sample (below) that every
runtime must reproduce.

---

## Export and parity checks

1. The prior (EMA weights) and the decoder, wrapped so it takes a standardised latent
   and returns probabilities, are rounded to fp16 and exported with the TorchScript
   ONNX exporter at opset 17 (plain graphs that onnxruntime-web runs). The prior has
   a dynamic batch axis; the decoder takes a batch of one.
2. Large initialisers are converted to fp16 + `Cast` and the graph is checked with
   `onnx.checker`.
3. **ONNX vs PyTorch.** Both graphs are run in onnxruntime on random inputs and
   compared with PyTorch; the kernel **fails** if either differs by more than 1e-3.
   The measured differences are recorded in `meta.json` → `parity`.
4. **Reference sample.** K2 embeds the prompt `"a red office chair with armrests and
   wheels"`, draws seed-123 noise, runs the 50-step guided sampler in fp32 on the CPU
   and stores the noise, the first 8 embedding values (`cond_head`) and the final
   latent in `meta.json` → `reference`.

The two runtimes in this repository check themselves against that reference:

- browser: `node frontend/scripts/verify-text2voxel.mjs` (tokeniser ids and
  embeddings against Python, sampler parity against `reference`, mesher and
  blueprint checks);
- backend: `backend/tests/test_text2voxel.py` (reference parity whenever model files
  are installed; skipped otherwise).

The embedding head must match `cond_head` to about 1e-3 and the final latent must
match `reference.latent` to within 2e-2 (the tolerance absorbs small arithmetic
differences between runtimes and CPUs).

---

## Evaluation

All metrics are computed in K2 on held-out shapes with the fp16-rounded weights.

| Metric | Value | What it measures |
| --- | --- | --- |
| VAE IoU, all held-out shapes | 0.824 | Reconstruction quality: occupancy IoU at p > 0.5, encoder mean → decoder |
| VAE IoU, Text2Shape | 0.831 | the same, chairs and tables only |
| VAE IoU, ModelNet40 | 0.816 | the same, ModelNet only |
| VAE colour MAE | 0.047 | mean absolute RGB error (0-1 scale) on occupied voxels |
| Prior: held-out caption IoU | 0.118 | IoU between one sample per held-out Text2Shape caption and the shape that caption describes (first 256 held-out captions, about 50 shapes) |
| Prior: shuffled-caption IoU | 0.054 | the same samples scored against a *different* held-out shape. The gap to the line above is how much the caption, rather than the generic chair/table prior, determines the shape |
| Prior: ModelNet class accuracy | 0.913 | 4 samples for each of the 40 classes from the prompt `"a <class>"`, re-encoded by the VAE encoder and labelled by the nearest training latent; the fraction labelled with the prompted class (chance is roughly 1/40). Per-class values are in `k2_report.json` |
| ONNX parity (decoder / prior, max abs) | 5.7e-6 / 2.1e-6 | step 3 above |
| Training time (K2 wall clock) | 8,205 s (2 h 17 min) | `k2_report.json` → `seconds` |
| Hardware | 2 x Tesla T4 (Kaggle) | `meta.json` → `training.hardware` |

Reading the numbers: a held-out caption produces a shape about twice as close to the
one it describes as to a random other shape (0.118 vs 0.054). Absolute IoU is low
because a caption rarely pins down one exact chair, and thin legs and slats make IoU
harsh. 31 of the 40 ModelNet classes score 4/4. The lowest are plant and wardrobe
(2/4) and `table` (0/4). The `table` score is a limitation of the metric: "a table"
draws mostly on the 8,443 Text2Shape tables, and the metric only compares against
ModelNet latents, where the nearest neighbours are desks.

**Text-encoder drift across machines.** The MiniLM model is int8-quantised, and its
dynamic quantisation gives slightly different embeddings on different CPUs. Between
the Kaggle training machine and a Windows laptop the difference is up to 8e-3 per
component, native or WebAssembly alike. The latent a fixed seed produces then moves
by up to 8e-2, which is about the change from a paraphrase and makes no visible
difference to the shape. Both verification suites therefore check the sampler
against an independent DDIM fed the same condition, and report the drift from the
training machine only as information.

These are automatic proxies. The class-accuracy metric uses the model's own encoder,
so it measures whether samples land near the right category in the model's latent
space, not whether a person would recognise them. No human evaluation has been done.
The sample montage `samples_final.png` (16 fixed prompts, front/side/top views) from
the same run is the qualitative check.

---

## Intended use

- Research and teaching about text-conditioned 3D generation.
- Quick, offline concept blocking of **chairs and tables** from a description, and
  coarse placeholders for the other 39 ModelNet categories, inside this app: viewed
  in 3D, drawn as a blueprint, exported as GLB/STL/OBJ/PLY.
- A fully in-browser, no-install demonstration of the pipeline's output stages.

### Out of scope

- **Commercial use of any kind** (licence).
- Anything that will be manufactured without engineering review. The output is a
  64³ design-intent sketch, not a drawing: no tolerances, no wall thicknesses, no
  materials.
- **Assistive devices.** The project's research motivation is bespoke assistive
  products, but the training data has no assistive-device category (no wheelchair,
  tray, grip or brace classes). A prompt for one gets the closest thing the model
  knows, most likely a chair or a table.
- Anything where a person's safety depends on the geometry.

---

## Limitations

- **Resolution.** 64 voxels across the largest dimension: about 16 mm per voxel on a
  1 m object. Parts thinner than a voxel (thin legs, rods, handles) break up or come
  out thickened.
- **Vocabulary.** Free-form descriptions are only learnt for chairs and tables, the
  only shapes with human captions. The 40 ModelNet categories were trained on
  templates (`"a red car"`), so for them the model knows the category name, a few
  synonyms and 12 colours (red, orange, yellow, green, blue, purple, pink, brown,
  black, white, grey, silver) and little else: expect `"a futuristic silver sports
  car"` to give a plain silver car.
- **Colour.** ModelNet shapes were trained with one uniform synthetic colour, so
  those categories come out in one colour. Text2Shape colour is effectively 32³.
- **No refusal or novelty detection.** Every prompt produces a shape. A prompt
  outside the training distribution maps to the nearest familiar one without warning.
- **Left and right.** Training mirrors shapes at random while keeping the caption, so
  a caption that says "armrest on the left" is not learnt.
- **Solid output.** Training volumes are filled, so outputs are solid; interiors,
  drawers and hollow parts that a caption mentions are not modelled.
- **Scale and proportion.** The model decides proportions. Requested dimensions only
  set a single uniform scale, so the other two dimensions will not match a
  `2000 x 500 x 750 mm` request.
- **Orientation.** One fixed orientation per source dataset. Up is reliable (every
  sample stands upright), but ModelNet40's original release has an inconsistent
  heading in several classes: cars run along x, and some airplanes, lamps and
  guitars sit diagonally. So a generated car may face sideways, which swaps the
  front and side elevations in its blueprint. Chairs and tables (Text2Shape) face
  +z consistently. An aligned ModelNet40 variant exists and would fix this.
- **Data bias.** Captions are English only and crowd-sourced; ShapeNet furniture
  skews towards Western catalogue designs; ModelNet is imbalanced (84 bowls, 989
  chairs), which the prior partly offsets with 1/√n class weighting.
- **Scale of the model.** 16 M shipped parameters, trained for about two hours. It is far
  below current large text-to-3D models, and is meant to be small enough to run in a
  web page, not to compete with them.

---

## Reproducing

Everything runs on Kaggle; nothing needs a local GPU. The short version:

```bash
python training/text2voxel/build_kernels.py                  # writes kaggle/k1, kaggle/k2
python -m kaggle kernels push -p training/text2voxel/kaggle/k1   # data prep (CPU)
# ...inspect K1's montages, set ORIENT in k2_train.py, rebuild, then:
python -m kaggle kernels push -p training/text2voxel/kaggle/k2   # training (T4 x2)
```

[`training/text2voxel/README.md`](../training/text2voxel/README.md) has the full
procedure: accounts, outputs, checking a run and publishing the `text2voxel-v1`
release that the web build and the backend install from.

---

## Licences

| Component | Terms | Role |
| --- | --- | --- |
| Text2Shape captions and voxels | derived from ShapeNet: ShapeNet terms of use ([shapenet.org](https://shapenet.org/)), non-commercial research and education | training data |
| ModelNet40 | [Princeton ModelNet](https://modelnet.cs.princeton.edu/): provided for academic research only; model copyright stays with the original authors | training data |
| `all-MiniLM-L6-v2` (Xenova ONNX export) | Apache-2.0 | frozen text encoder, redistributed as downloaded |
| onnxruntime / onnxruntime-web | MIT | inference |
| **Text2Voxel-64 weights** | **non-commercial research use only**, inherited from the training data | `decoder.onnx`, `prior.onnx` |

If you need a commercially usable model, it has to be retrained on data whose terms
allow that. K2 does not depend on these datasets beyond K1's loaders and the caption
templates.

## References

- K. Chen, C. B. Choy, M. Savva, A. X. Chang, T. Funkhouser, S. Savarese.
  *Text2Shape: Generating Shapes from Natural Language by Learning Joint
  Embeddings.* ACCV 2018.
- A. X. Chang et al. *ShapeNet: An Information-Rich 3D Model Repository.*
  arXiv:1512.03012, 2015.
- Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, J. Xiao. *3D ShapeNets: A
  Deep Representation for Volumetric Shapes.* CVPR 2015.
- W. Wang et al. *MiniLM: Deep Self-Attention Distillation for Task-Agnostic
  Compression of Pre-Trained Transformers.* NeurIPS 2020; N. Reimers, I. Gurevych.
  *Sentence-BERT.* EMNLP 2019.
- R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.*
  CVPR 2022.
- J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
- A. Nichol, P. Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML
  2021 (cosine schedule).
- T. Salimans, J. Ho. *Progressive Distillation for Fast Sampling of Diffusion
  Models.* ICLR 2022 (v-prediction).
- J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance.* 2022.
- W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers.* ICCV 2023
  (adaLN conditioning).
