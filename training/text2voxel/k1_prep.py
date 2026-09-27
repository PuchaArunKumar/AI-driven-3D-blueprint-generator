"""K1 - data preparation for Text2Voxel (Kaggle CPU kernel, internet on).

Produces, in /kaggle/working:

* ``t2s_occ.npy``    uint8 [N, 32768]  packed 64^3 occupancy, Text2Shape
* ``t2s_col32.npy``  uint8 [N, 32, 32, 32, 3]  mean colour of occupied children
* ``t2s_models.json`` model ids + ShapeNet category per row
* ``t2s_captions.json`` [{"shape": row, "text": caption}]
* ``mn_occ.npy``     uint8 [M, 32768]  packed 64^3 occupancy, ModelNet40
* ``mn_meta.json``   class / split / file per row, solid-fill statistics
* ``text_strings.json`` + ``text_emb.npy`` (float16 [S, 384]) - MiniLM embeddings
  of every caption and every template string
* ``montage_*.png`` axis-projection montages for checking orientation
* ``k1_report.json``

Arrays are stored in each source's RAW axis order; orientation to the
canonical frame (x right, y up, z towards the viewer) is applied by K2 once
the montages have been inspected.
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
import time
import urllib.request
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# `common` is inlined above this line by tools/build_kernels.py.

OUT = Path("/kaggle/working")
TMP = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp/t2v")
TMP.mkdir(parents=True, exist_ok=True)

T2S_BASE = "http://text2shape.stanford.edu/dataset"
T2S_CAPTIONS = f"{T2S_BASE}/captions.tablechair.csv"
T2S_VOXELS = f"{T2S_BASE}/shapenet/nrrd_256_filter_div_64_solid.zip"
RES = 64
MARGIN = 2  # voxels of empty border around ModelNet shapes

REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(*args) -> None:
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def download(url: str, target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    log("GET", url)
    started = time.time()
    with urllib.request.urlopen(url, timeout=120) as response, open(target, "wb") as sink:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 0.0
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            sink.write(block)
            done += len(block)
            if total and done / total >= next_report:
                log(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                next_report += 0.1
    log(f"  done in {time.time() - started:.0f}s")
    return target


# --------------------------------------------------------------------------
# Text2Shape
# --------------------------------------------------------------------------


def read_nrrd(raw: bytes) -> tuple[dict, np.ndarray]:
    """Minimal NRRD reader returning data indexed like pynrrd's 'F' order."""
    head_end = raw.find(b"\n\n")
    header_text = raw[:head_end].decode("latin-1")
    body = raw[head_end + 2:]
    header: dict[str, str] = {}
    for line in header_text.splitlines()[1:]:
        if line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip().lower()] = value.lstrip("=").strip()
    sizes = [int(v) for v in header["sizes"].split()]
    encoding = header.get("encoding", "raw").lower()
    if encoding in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif encoding != "raw":
        raise ValueError(f"unsupported NRRD encoding {encoding}")
    kind = header.get("type", "uint8").lower()
    dtype = {"uint8": np.uint8, "unsigned char": np.uint8, "uchar": np.uint8,
             "float": np.float32, "double": np.float64}.get(kind)
    if dtype is None:
        raise ValueError(f"unsupported NRRD type {kind}")
    data = np.frombuffer(body, dtype=dtype, count=int(np.prod(sizes)))
    # NRRD stores the first listed axis fastest.
    data = data.reshape(sizes[::-1]).transpose()
    return header, data


def pool_colour(occ: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """64^3 colours -> 32^3 mean colour over occupied children (0 where empty)."""
    weights = occ.reshape(32, 2, 32, 2, 32, 2).astype(np.float32)
    count = weights.sum(axis=(1, 3, 5))
    colour = rgb.reshape(32, 2, 32, 2, 32, 2, 3).astype(np.float32)
    summed = (colour * weights[..., None]).sum(axis=(1, 3, 5))
    mean = summed / np.clip(count, 1, None)[..., None]
    return np.clip(np.rint(mean), 0, 255).astype(np.uint8)


def prepare_text2shape() -> tuple[list[str], list[dict]]:
    caption_csv = download(T2S_CAPTIONS, TMP / "captions.tablechair.csv")
    zip_path = download(T2S_VOXELS, TMP / "t2s64.zip")

    captions_by_model: dict[str, list[str]] = {}
    category_by_model: dict[str, str] = {}
    with open(caption_csv, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        REPORT["t2s_csv_columns"] = reader.fieldnames
        for row in reader:
            model = (row.get("modelId") or "").strip()
            text = " ".join((row.get("description") or "").split())
            if not model or not text:
                continue
            captions_by_model.setdefault(model, []).append(text)
            category_by_model[model] = (row.get("category") or "").strip()
    log(f"captions: {sum(map(len, captions_by_model.values())):,} for "
        f"{len(captions_by_model):,} models")

    archive = zipfile.ZipFile(zip_path)
    members = [m for m in archive.namelist() if m.endswith(".nrrd")]
    log(f"nrrd members: {len(members):,}")

    occ_rows: list[np.ndarray] = []
    colour_rows: list[np.ndarray] = []
    models: list[str] = []
    alpha_values: dict[int, int] = {}
    first_header = None
    for index, member in enumerate(members):
        model = Path(member).stem
        if model not in captions_by_model:
            continue
        header, data = read_nrrd(archive.read(member))
        if first_header is None:
            first_header = header
            REPORT["t2s_nrrd_header"] = header
            REPORT["t2s_nrrd_shape"] = list(data.shape)
            log("nrrd header", header, "shape", data.shape)
        channel_axis = [i for i, s in enumerate(data.shape) if s == 4]
        if not channel_axis or data.ndim != 4:
            raise ValueError(f"unexpected voxel layout {data.shape} in {member}")
        vox = np.moveaxis(data, channel_axis[0], -1)
        if vox.shape[:3] != (RES, RES, RES):
            raise ValueError(f"unexpected resolution {vox.shape} in {member}")
        alpha = vox[..., 3]
        if index < 200:
            values, counts = np.unique(alpha, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                alpha_values[value] = alpha_values.get(value, 0) + count
        occ = alpha >= 128
        if occ.sum() < 20:
            continue
        occ_rows.append(np.packbits(occ.reshape(-1)))
        colour_rows.append(pool_colour(occ, vox[..., :3]))
        models.append(model)
        if len(models) % 2000 == 0:
            log(f"  parsed {len(models):,}")

    REPORT["t2s_alpha_histogram_first200"] = {str(k): v for k, v in sorted(alpha_values.items())}
    np.save(OUT / "t2s_occ.npy", np.stack(occ_rows))
    np.save(OUT / "t2s_col32.npy", np.stack(colour_rows))
    write_json(OUT / "t2s_models.json",
               [{"model": m, "category": category_by_model.get(m, "")} for m in models])

    captions: list[dict] = []
    for row, model in enumerate(models):
        for text in captions_by_model[model]:
            captions.append({"shape": row, "text": text})
    write_json(OUT / "t2s_captions.json", captions)
    REPORT["t2s_shapes"] = len(models)
    REPORT["t2s_captions"] = len(captions)
    categories: dict[str, int] = {}
    for model in models:
        categories[category_by_model.get(model, "")] = categories.get(category_by_model.get(model, ""), 0) + 1
    REPORT["t2s_categories"] = categories
    log(f"text2shape: {len(models):,} shapes, {len(captions):,} captions, {categories}")
    return models, captions


# --------------------------------------------------------------------------
# ModelNet40
# --------------------------------------------------------------------------


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OFF reader tolerant of ModelNet's 'OFF<n> <m> <k>' header defect."""
    tokens = path.read_bytes().split()
    first = tokens[0].decode()
    if first == "OFF":
        cursor = 1
    elif first.startswith("OFF"):
        tokens = [first[3:].encode()] + tokens[1:]
        cursor = 0
    else:
        raise ValueError("not an OFF file")
    n_vertices, n_faces = int(tokens[cursor]), int(tokens[cursor + 1])
    cursor += 3
    vertices = np.array(tokens[cursor:cursor + 3 * n_vertices], dtype=np.float64).reshape(-1, 3)
    cursor += 3 * n_vertices
    rest = np.array(tokens[cursor:], dtype=np.int64)
    faces: list[np.ndarray] = []
    position = 0
    # Faces are usually all triangles - take the fast path when they are.
    if rest.size >= 4 * n_faces and np.all(rest[0:4 * n_faces:4] == 3):
        tri = rest[:4 * n_faces].reshape(-1, 4)[:, 1:]
        return vertices, tri
    for _ in range(n_faces):
        count = int(rest[position])
        polygon = rest[position + 1:position + 1 + count]
        for k in range(1, count - 1):
            faces.append(np.array([polygon[0], polygon[k], polygon[k + 1]]))
        position += 1 + count
    return vertices, np.array(faces, dtype=np.int64)


def voxelize_mesh(vertices: np.ndarray, faces: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Surface-sample a mesh into a 64^3 grid, then fill enclosed space."""
    from scipy import ndimage

    lo, hi = vertices.min(0), vertices.max(0)
    extent = float((hi - lo).max())
    if extent <= 0:
        raise ValueError("degenerate mesh")
    span = RES - 2 * MARGIN
    scale = (span - 1e-6) / extent
    centred = (vertices - (lo + hi) / 2) * scale + RES / 2  # voxel coordinates

    tri = centred[faces]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    total = float(area.sum())
    # ~16 samples per voxel face of surface keeps the shell gap-free.
    count = int(np.clip(total * 16, 20_000, 3_000_000))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=area / total if total > 0 else None)
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    flip = (u + v) > 1
    u = np.where(flip, 1 - u, u)
    v = np.where(flip, 1 - v, v)
    points = tri[chosen, 0] + u * edge_a[chosen] + v * edge_b[chosen]
    points = np.concatenate([points, centred], axis=0)

    grid = np.zeros((RES, RES, RES), dtype=bool)
    index = np.clip(np.floor(points).astype(np.int64), 0, RES - 1)
    grid[index[:, 0], index[:, 1], index[:, 2]] = True
    shell = int(grid.sum())
    filled = ndimage.binary_fill_holes(grid)
    return filled, float(filled.sum()) / max(shell, 1)


def _modelnet_worker(job: tuple[int, str]) -> tuple[int, bytes | None, float, str]:
    index, path = job
    try:
        vertices, faces = read_off(Path(path))
        grid, fill_ratio = voxelize_mesh(vertices, faces, seed=index)
        return index, np.packbits(grid.reshape(-1)).tobytes(), fill_ratio, ""
    except Exception as exc:  # noqa: BLE001 - report and skip bad meshes
        return index, None, 0.0, f"{type(exc).__name__}: {exc}"


def prepare_modelnet() -> list[dict]:
    anchor = find_input("airplane_0001.off")
    root = anchor.parent.parent.parent  # .../ModelNet40
    files = sorted(root.glob("*/*/*.off"))
    log(f"modelnet root {root}: {len(files):,} files")
    jobs = [(i, str(p)) for i, p in enumerate(files)]
    rows: dict[int, bytes] = {}
    fills: dict[int, float] = {}
    failures: list[str] = []
    started = time.time()
    with Pool(cpu_count()) as pool:
        for done, (index, packed, fill_ratio, error) in enumerate(
                pool.imap_unordered(_modelnet_worker, jobs, chunksize=8), start=1):
            if packed is None:
                failures.append(f"{files[index].name}: {error}")
            else:
                rows[index] = packed
                fills[index] = fill_ratio
            if done % 1000 == 0:
                log(f"  voxelised {done:,}/{len(files):,} ({time.time() - started:.0f}s)")

    kept = sorted(rows)
    occ = np.stack([np.frombuffer(rows[i], dtype=np.uint8) for i in kept])
    np.save(OUT / "mn_occ.npy", occ)
    meta = []
    for i in kept:
        path = files[i]
        meta.append({"class": path.parent.parent.name, "split": path.parent.name,
                     "file": path.name, "fill_ratio": round(fills[i], 3)})
    write_json(OUT / "mn_meta.json", meta)
    ratios = np.array([fills[i] for i in kept])
    REPORT["mn_shapes"] = len(kept)
    REPORT["mn_failures"] = failures[:50]
    REPORT["mn_failure_count"] = len(failures)
    REPORT["mn_fill_ratio_quantiles"] = np.quantile(ratios, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
    REPORT["mn_unfilled_fraction"] = float((ratios < 1.05).mean())
    log(f"modelnet: {len(kept):,} voxelised, {len(failures)} failed, "
        f"median fill ratio {np.median(ratios):.2f}")
    return meta


# --------------------------------------------------------------------------
# Embeddings and montages
# --------------------------------------------------------------------------


def embed_strings(captions: list[dict]) -> None:
    strings = sorted({c["text"] for c in captions} | set(all_template_strings()))
    encoder = TextEncoder(fetch_minilm(TMP / "minilm"), threads=cpu_count())
    started = time.time()
    embeddings = encoder.encode(strings)
    log(f"embedded {len(strings):,} strings in {time.time() - started:.0f}s")
    np.save(OUT / "text_emb.npy", embeddings.astype(np.float16))
    write_json(OUT / "text_strings.json", strings)
    REPORT["text_strings"] = len(strings)
    # Sanity: paraphrases should be closer than unrelated phrases.
    probe = encoder.encode(["a wooden chair", "a chair made of wood", "a red sports car"])
    REPORT["embedding_probe"] = {
        "chair~wood-chair": float(probe[0] @ probe[1]),
        "chair~car": float(probe[0] @ probe[2]),
    }


def projection(occ: np.ndarray, colour: np.ndarray | None, axis: int) -> np.ndarray:
    """First-hit image looking along +axis from index 0."""
    moved = np.moveaxis(occ, axis, 0)
    hit = moved.any(0)
    depth = moved.argmax(0)
    if colour is None:
        shade = 1.0 - 0.6 * depth / RES
        image = np.stack([shade] * 3, -1)
    else:
        col = np.moveaxis(colour, axis, 0)
        a, b = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
        image = col[depth, a, b] / 255.0
    image[~hit] = 1.0
    return image


def montage(name: str, items: list[tuple[str, np.ndarray, np.ndarray | None]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(items), 3, figsize=(6.6, 2.2 * len(items)))
    for row, (title, occ, colour) in enumerate(items):
        for axis in range(3):
            ax = axes[row, axis]
            # imshow: array dim0 -> screen rows (down), dim1 -> screen columns.
            ax.imshow(projection(occ, colour, axis), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"along raw axis {axis}\nrows=next axis, cols=last", fontsize=7)
        axes[row, 0].set_ylabel(title[:38], fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / f"montage_{name}.png", dpi=110)
    plt.close(fig)


def unpack(row: np.ndarray) -> np.ndarray:
    return np.unpackbits(row)[: RES ** 3].reshape(RES, RES, RES).astype(bool)


def make_montages(t2s_models: list[str], captions: list[dict], mn_meta: list[dict]) -> None:
    t2s_occ = np.load(OUT / "t2s_occ.npy", mmap_mode="r")
    t2s_col = np.load(OUT / "t2s_col32.npy", mmap_mode="r")
    first_caption: dict[int, str] = {}
    for caption in captions:
        first_caption.setdefault(caption["shape"], caption["text"])
    rng = np.random.default_rng(0)
    picks = rng.choice(len(t2s_models), size=min(12, len(t2s_models)), replace=False)
    items = []
    for row in picks:
        colour = np.repeat(np.repeat(np.repeat(t2s_col[row], 2, 0), 2, 1), 2, 2)
        items.append((first_caption.get(int(row), ""), unpack(t2s_occ[row]), colour))
    montage("t2s", items)

    mn_occ = np.load(OUT / "mn_occ.npy", mmap_mode="r")
    by_class: dict[str, int] = {}
    for row, meta in enumerate(mn_meta):
        by_class.setdefault(meta["class"], row)
    wanted = ["airplane", "car", "chair", "monitor", "toilet", "bed", "lamp", "guitar",
              "sofa", "person", "piano", "bathtub"]
    items = [(name, unpack(mn_occ[by_class[name]]), None) for name in wanted if name in by_class]
    montage("modelnet", items)


def main() -> None:
    started = time.time()
    ensure_packages("onnxruntime", "tokenizers")
    t2s_models, captions = prepare_text2shape()
    mn_meta = prepare_modelnet()
    embed_strings(captions)
    make_montages(t2s_models, captions, mn_meta)
    REPORT["seconds"] = round(time.time() - started)
    write_json(OUT / "k1_report.json", REPORT)
    log("K1 COMPLETE", REPORT)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
