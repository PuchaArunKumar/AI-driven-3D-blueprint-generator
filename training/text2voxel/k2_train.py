"""K2 - train Text2Voxel (Kaggle T4 kernel).

Stage A  shape VAE: 64^3 occupancy + colour  <->  256-d latent.
Stage B  latent diffusion prior: p(latent | MiniLM caption embedding), an
         adaLN residual MLP trained with v-prediction and classifier-free
         guidance (the unconditional branch is an all-zero embedding).

Exports to /kaggle/working/model:
    decoder.onnx   z_std[1,256] -> occ[1,64,64,64], rgb[1,3,64,64,64]  (sigmoid)
    prior.onnx     x[B,256], t[B], cond[B,384] -> v[B,256]
    meta.json      schedule, dims, orientation, metrics, provenance

Canonical voxel frame (index order [x, y, z]): x to the right, y up, z towards
the viewer - the object's front faces +z.
"""

from __future__ import annotations

import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# `common` is inlined above this line by training/text2voxel/build_kernels.py.

OUT = Path(os.environ.get("T2V_OUT", "/kaggle/working"))
MODEL_DIR = OUT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
RES = 64
LATENT = 256
SMOKE = os.environ.get("T2V_SMOKE") == "1"  # seconds-long run for local shape checks

# --- orientation: RAW axes -> canonical (x right, y up, z front) -----------
# perm lists which raw axis becomes canonical x, y, z; flip lists canonical
# axes to reverse afterwards. Decided from K1's montages.
#
# Text2Shape: the NRRD header maps raw axes (0, 1, 2) to ShapeNet world
# (y, z, x); y is up and chair backs sit at +z, so the front faces -z. A 180
# degree turn about y gives x = -raw2, y = raw0, z = -raw1 (a proper
# rotation, never a mirror).
# ModelNet40 (OFF coordinates): z is up. x = raw0, y = raw2, z = -raw1 is a
# proper rotation; azimuth is inconsistent between ModelNet classes in the
# source data (cars run along x, some airplanes sit diagonally), which the
# model card records as a limitation.
ORIENT = {
    "t2s": {"perm": (2, 0, 1), "flip": (0, 2)},
    "mn": {"perm": (0, 2, 1), "flip": (2,)},
}

# --- budgets ----------------------------------------------------------------
AE_MINUTES = 0.3 if SMOKE else 110
PRIOR_MINUTES = 0.2 if SMOKE else 20
AE_BATCH = 2 if SMOKE else 32
PRIOR_BATCH = 64 if SMOKE else 1024
KL_WEIGHT = 2e-5
DIFFUSION_T = 1000
SAMPLE_STEPS = 50
GUIDANCE = 3.0
COND_DROP = 0.1

REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}
T0 = time.time()


def log(*args) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *args, flush=True)


def stable_hash(text: str) -> int:
    return zlib.crc32(text.encode())


# ==========================================================================
# Data (kept on the GPU as packed bits)
# ==========================================================================

SHIFTS = None


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [B, 32768] (np.packbits order) -> float [B, 64, 64, 64]."""
    global SHIFTS
    if SHIFTS is None or SHIFTS.device != packed.device:
        SHIFTS = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> SHIFTS) & 1
    return bits.reshape(packed.shape[0], RES, RES, RES).float()


def orient(volume: torch.Tensor, source: str) -> torch.Tensor:
    """Apply ORIENT to the last three (spatial) axes of ``volume``."""
    spec = ORIENT[source]
    lead = volume.dim() - 3
    order = list(range(lead)) + [lead + a for a in spec["perm"]]
    volume = volume.permute(*order)
    if spec["flip"]:
        volume = volume.flip([lead + a for a in spec["flip"]])
    return volume


class ShapeBank:
    """All training shapes resident on the device, with colour targets."""

    def __init__(self) -> None:
        t2s_occ = np.load(find_input("t2s_occ.npy"))
        t2s_col = np.load(find_input("t2s_col32.npy"))
        mn_occ = np.load(find_input("mn_occ.npy"))
        self.t2s_models = json_load(find_input("t2s_models.json"))
        self.mn_meta = json_load(find_input("mn_meta.json"))
        if SMOKE:
            t2s_occ, t2s_col, self.t2s_models = t2s_occ[:64], t2s_col[:64], self.t2s_models[:64]
            mn_occ, self.mn_meta = mn_occ[:64], self.mn_meta[:64]
        self.n_t2s = len(t2s_occ)
        self.n_mn = len(mn_occ)
        self.occ = torch.from_numpy(np.concatenate([t2s_occ, mn_occ])).to(DEVICE)
        # colour: [N_t2s, 3, 32, 32, 32] uint8, raw axes
        self.col = torch.from_numpy(t2s_col).permute(0, 4, 1, 2, 3).contiguous().to(DEVICE)
        self.classes = sorted({m["class"] for m in self.mn_meta})
        self.mn_class = torch.tensor([self.classes.index(m["class"]) for m in self.mn_meta],
                                     device=DEVICE)
        palette = [NEUTRAL_RGB] + list(PALETTE.values())
        self.palette = torch.tensor(palette, dtype=torch.float32, device=DEVICE) / 255.0
        self.palette_names = ["neutral"] + list(PALETTE.keys())
        # Held-out shapes: 5% by stable hash of their identity.
        val_t2s = [stable_hash(m["model"]) % 20 == 0 for m in self.t2s_models]
        val_mn = [stable_hash(m["file"]) % 20 == 0 for m in self.mn_meta]
        self.is_val = torch.tensor(val_t2s + val_mn, device=DEVICE)
        self.shift = self._centring_shifts()
        self.train_index = torch.nonzero(~self.is_val).squeeze(1)
        self.val_index = torch.nonzero(self.is_val).squeeze(1)
        log(f"shapes: t2s {self.n_t2s:,}  modelnet {self.n_mn:,}  "
            f"train {len(self.train_index):,}  val {len(self.val_index):,}")

    def _centring_shifts(self) -> torch.Tensor:
        """Integer roll per shape (raw axes) that centres its bounding box.

        Text2Shape grids place every object against the minimum corner while
        ModelNet shapes are centred; mixing the two would teach the model two
        placements. Rolling is lossless because the margins are empty.
        """
        shifts = torch.zeros(len(self.occ), 3, dtype=torch.long, device=DEVICE)
        coords = torch.arange(RES, device=DEVICE)
        for chunk in torch.arange(len(self.occ), device=DEVICE).split(256):
            occ = unpack(self.occ[chunk]) > 0
            for axis in range(3):
                other = tuple(a for a in (1, 2, 3) if a != axis + 1)
                present = occ.any(dim=other)  # [B, RES]
                lo = torch.where(present, coords, RES).min(1).values
                hi = torch.where(present, coords, -1).max(1).values
                size = (hi - lo + 1).clamp_min(0)
                shifts[chunk, axis] = (RES - size) // 2 - lo
        empty = shifts.abs().max(1).values > RES
        shifts[empty] = 0
        log(f"centring: mean |shift| t2s {shifts[:self.n_t2s].abs().float().mean().item():.1f} "
            f"modelnet {shifts[self.n_t2s:].abs().float().mean().item():.1f} voxels")
        return shifts

    def _centre(self, volume: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Roll each [..., X, Y, Z] raw-frame volume by its shape's shift."""
        out = torch.empty_like(volume)
        for row, shape in enumerate(index.tolist()):
            out[row] = torch.roll(volume[row], tuple(self.shift[shape].tolist()), dims=(-3, -2, -1))
        return out

    def batch(self, index: torch.Tensor, colour_id: torch.Tensor | None = None,
              mirror: torch.Tensor | None = None) -> torch.Tensor:
        """Build [B, 4, 64, 64, 64] = occupancy + RGB in the canonical frame.

        ``colour_id`` picks the synthetic colour for ModelNet rows
        (0 = neutral, 1.. = palette); Text2Shape rows keep their own colour.
        """
        index = index.to(DEVICE)
        occ = self._centre(unpack(self.occ[index]), index)  # raw axes, centred
        out = torch.zeros(len(index), 4, RES, RES, RES, device=DEVICE)
        is_t2s = index < self.n_t2s
        if is_t2s.any():
            rows = torch.nonzero(is_t2s).squeeze(1)
            occ_t = orient(occ[rows], "t2s")
            col = self.col[index[rows]].float() / 255.0
            col = F.interpolate(col, scale_factor=2, mode="nearest")
            col = orient(self._centre(col, index[rows]), "t2s")
            out[rows, 0] = occ_t
            out[rows, 1:] = col * occ_t.unsqueeze(1)
        if (~is_t2s).any():
            rows = torch.nonzero(~is_t2s).squeeze(1)
            occ_m = orient(occ[rows], "mn")
            if colour_id is None:
                cid = torch.zeros(len(rows), dtype=torch.long, device=DEVICE)
            else:
                cid = colour_id.to(DEVICE)[rows]
            rgb = self.palette[cid][:, :, None, None, None]
            out[rows, 0] = occ_m
            out[rows, 1:] = rgb * occ_m.unsqueeze(1)
        if mirror is not None and mirror.any():
            rows = torch.nonzero(mirror.to(DEVICE)).squeeze(1)
            out[rows] = out[rows].flip(2)  # mirror along x
        return out


def json_load(path: Path):
    import json
    return json.loads(Path(path).read_text())


# ==========================================================================
# Stage A - shape VAE
# ==========================================================================


def norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1),
            norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 32, 4, 2, 1), nn.SiLU(),                     # 32^3
            nn.Conv3d(32, 64, 4, 2, 1), ResBlock(64), nn.SiLU(),       # 16^3
            nn.Conv3d(64, 128, 4, 2, 1), ResBlock(128), nn.SiLU(),     # 8^3
            nn.Conv3d(128, 256, 4, 2, 1), ResBlock(256), nn.SiLU(),    # 4^3
        )
        self.head = nn.Linear(256 * 64, 2 * LATENT)

    def forward(self, x):
        h = self.net(x).flatten(1)
        mu, logvar = self.head(h).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 6)


def voxel_shuffle(x: torch.Tensor, channels: int) -> torch.Tensor:
    """[B, C*8, D, H, W] -> [B, C, 2D, 2H, 2W]: the 3D analogue of PixelShuffle."""
    b, _, d, h, w = x.shape
    x = x.view(b, channels, 2, 2, 2, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, channels, 2 * d, 2 * h, 2 * w)


class Decoder(nn.Module):
    """Latent -> 64^3 logits (occupancy + RGB).

    The last stage predicts the eight 64^3 children of every 32^3 cell and
    shuffles them into place, which is ~2x cheaper than convolving at 64^3 -
    that matters because the browser runs this network on the CPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(LATENT, 128 * 64)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose3d(128, 96, 4, 2, 1), ResBlock(96), nn.SiLU(),   # 8^3
            nn.ConvTranspose3d(96, 64, 4, 2, 1), ResBlock(64), nn.SiLU(),    # 16^3
            nn.ConvTranspose3d(64, 32, 4, 2, 1), ResBlock(32), nn.SiLU(),    # 32^3
            nn.Conv3d(32, 4 * 8, 3, padding=1),                              # 8 children
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4, 4)
        return voxel_shuffle(self.net(h), 4)  # logits: occupancy + rgb at 64^3


def recon_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    occ_logit = logits[:, 0].float()
    occ = target[:, 0]
    bce = F.binary_cross_entropy_with_logits(
        occ_logit, occ, pos_weight=torch.tensor(2.0, device=logits.device))
    prob = torch.sigmoid(occ_logit)
    inter = (prob * occ).flatten(1).sum(1)
    union = (prob + occ - prob * occ).flatten(1).sum(1)
    soft_iou = 1 - (inter / union.clamp_min(1)).mean()
    rgb = torch.sigmoid(logits[:, 1:].float())
    mask = occ.unsqueeze(1)
    colour = ((rgb - target[:, 1:]).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1)
    total = bce + 0.5 * soft_iou + colour
    return total, {"bce": bce.item(), "soft_iou": soft_iou.item(), "colour": colour.item()}


def iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = logits[:, 0] > 0
    occ = target[:, 0] > 0.5
    inter = (pred & occ).flatten(1).sum(1).float()
    union = (pred | occ).flatten(1).sum(1).float().clamp_min(1)
    return inter / union


def random_colour_ids(bank: ShapeBank, n: int) -> torch.Tensor:
    """Half neutral, half a random palette colour (ModelNet rows only)."""
    neutral = torch.rand(n, device=DEVICE) < 0.5
    ids = torch.randint(1, len(bank.palette), (n,), device=DEVICE)
    return torch.where(neutral, torch.zeros_like(ids), ids)


def train_vae(bank: ShapeBank) -> tuple[Encoder, Decoder]:
    encoder, decoder = Encoder().to(DEVICE), Decoder().to(DEVICE)
    # Kaggle's T4 machine has two GPUs; split each batch across both.
    gpus = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
    run_encoder = nn.DataParallel(encoder) if gpus > 1 else encoder
    run_decoder = nn.DataParallel(decoder) if gpus > 1 else decoder
    REPORT["gpus"] = gpus
    params = list(encoder.parameters()) + list(decoder.parameters())
    log(f"VAE params: encoder {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M, "
        f"decoder {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")
    optimiser = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    deadline = time.time() + AE_MINUTES * 60
    # Estimate total steps from a timing probe so the cosine schedule ends on time.
    step, total_steps, probe_started = 0, None, time.time()
    history = []
    while time.time() < deadline:
        index = bank.train_index[torch.randint(len(bank.train_index), (AE_BATCH,), device=DEVICE)]
        target = bank.batch(index, random_colour_ids(bank, AE_BATCH),
                            torch.rand(AE_BATCH, device=DEVICE) < 0.5)
        if total_steps is not None:
            progress = min(step / total_steps, 1.0)
            lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * progress))
        else:
            lr = 1e-3 * min(1.0, (step + 1) / 200)
        for group in optimiser.param_groups:
            group["lr"] = lr
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            mu, logvar = run_encoder(target)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            logits = run_decoder(z)
        loss, parts = recon_loss(logits, target)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
        loss = loss + KL_WEIGHT * kl
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimiser)
        scaler.update()
        step += 1
        if step == (5 if SMOKE else 200) and total_steps is None:
            per_step = (time.time() - probe_started) / step
            total_steps = int(step + (deadline - time.time()) / per_step)
            log(f"VAE: {per_step * 1000:.0f} ms/step -> ~{total_steps:,} steps")
        if step % 1000 == 0 or (SMOKE and step % 5 == 0):
            val = evaluate_vae(bank, encoder, decoder, limit=512)
            history.append({"step": step, "loss": round(loss.item(), 4), "kl": round(kl.item(), 1),
                            **{k: round(v, 4) for k, v in parts.items()}, **val})
            log(f"VAE step {step:,} lr {lr:.2e} loss {loss.item():.4f} kl {kl.item():.0f} "
                f"{parts} val {val}")
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                       OUT / "vae.pt")
    REPORT["vae_steps"] = step
    REPORT["vae_history"] = history[-10:]
    return encoder, decoder


@torch.no_grad()
def evaluate_vae(bank: ShapeBank, encoder: Encoder, decoder: Decoder, limit: int = 4096) -> dict:
    encoder.eval(), decoder.eval()
    ious, colour_err, t2s_ious, mn_ious = [], [], [], []
    for chunk in bank.val_index[:limit].split(64):
        target = bank.batch(chunk)
        logits = decoder(encoder(target)[0])
        value = iou(logits, target)
        ious.append(value)
        t2s = chunk < bank.n_t2s
        t2s_ious.append(value[t2s])
        mn_ious.append(value[~t2s])
        rgb = torch.sigmoid(logits[:, 1:])
        mask = target[:, :1]
        colour_err.append(((rgb - target[:, 1:]).abs() * mask).flatten(1).sum(1)
                          / (mask.flatten(1).sum(1) * 3).clamp_min(1))
    encoder.train(), decoder.train()
    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(1)  # noqa: E731
    return {"val_iou": round(cat(ious).mean().item(), 4),
            "val_iou_t2s": round(cat(t2s_ious).mean().item(), 4),
            "val_iou_modelnet": round(cat(mn_ious).mean().item(), 4),
            "val_colour_mae": round(cat(colour_err).mean().item(), 4)}


# ==========================================================================
# Stage B - text-conditioned latent diffusion prior
# ==========================================================================


def cosine_alphas_cumprod(steps: int = DIFFUSION_T, s: float = 0.008) -> np.ndarray:
    t = np.linspace(0, steps, steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas = np.clip(f[1:] / f[:-1], 1e-4, 0.9999)
    return np.cumprod(alphas).astype(np.float64)


def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class PriorBlock(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 3 * width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, h, emb):
        shift, scale, gate = self.modulation(emb).chunk(3, dim=1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class Prior(nn.Module):
    """v-prediction denoiser over standardised shape latents."""

    def __init__(self, width: int = 512, depth: int = 6, hidden: int = 1024) -> None:
        super().__init__()
        self.inp = nn.Linear(LATENT, width)
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width))
        self.cond = nn.Sequential(nn.Linear(EMBED_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(PriorBlock(width, hidden) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, LATENT)

    def forward(self, x, t, cond):
        emb = F.silu(self.time(timestep_embedding(t)) + self.cond(cond))
        h = self.inp(x)
        for block in self.blocks:
            h = block(h, emb)
        return self.out(self.out_norm(h))


class CaptionSampler:
    """Draws (latent, caption embedding) training pairs from three sources."""

    def __init__(self, bank: ShapeBank, latents_t2s: torch.Tensor, latents_mn: torch.Tensor) -> None:
        strings = json_load(find_input("text_strings.json"))
        emb = torch.from_numpy(np.load(find_input("text_emb.npy")).astype(np.float32))
        lookup = {s: i for i, s in enumerate(strings)}
        extra = [s for s in all_template_strings() if s not in lookup]
        if extra:  # templates changed since K1 - embed the new strings here
            log(f"embedding {len(extra)} new template strings")
            encoder = TextEncoder(fetch_minilm(Path(os.environ.get("T2V_MINILM", "/tmp/minilm"))))
            emb = torch.cat([emb, torch.from_numpy(encoder.encode(extra))])
            for s in extra:
                lookup[s] = len(lookup)
        self.emb = emb.to(DEVICE)
        self.lookup = lookup
        self.bank = bank
        self.lat_t2s = latents_t2s  # [N_t2s, 2(mirror), D]
        self.lat_mn = latents_mn    # [N_mn, C(colours), 2(mirror), D]

        captions = json_load(find_input("t2s_captions.json"))
        if SMOKE:
            captions = [c for c in captions if c["shape"] < bank.n_t2s]
        val = bank.is_val[:bank.n_t2s].cpu().numpy()
        train_caps = [c for c in captions if not val[c["shape"]]]
        self.val_caps = [c for c in captions if val[c["shape"]]]
        self.cap_shape = torch.tensor([c["shape"] for c in train_caps], device=DEVICE)
        self.cap_text = torch.tensor([lookup[c["text"]] for c in train_caps], device=DEVICE)

        # Plain category templates for Text2Shape shapes ("a chair", ...).
        cats = [("table" if "table" in m["category"].lower() else "chair") for m in bank.t2s_models]
        self.t2s_cat = torch.tensor([0 if c == "chair" else 1 for c in cats], device=DEVICE)
        self.t2s_plain = self._table([[lookup[s] for s in plain_captions(c)]
                                      for c in ("chair", "table")])
        self.t2s_train = torch.nonzero(~bank.is_val[:bank.n_t2s]).squeeze(1)

        # ModelNet: class-balanced (sqrt) sampling and caption tables. Row
        # 0 of the colour axis is "no colour named" (plain captions), rows
        # 1.. follow PALETTE order, matching ShapeBank.palette.
        mn_val = bank.is_val[bank.n_t2s:]
        self.mn_train = torch.nonzero(~mn_val).squeeze(1)
        counts = torch.bincount(bank.mn_class[self.mn_train], minlength=len(bank.classes)).float()
        weights = 1.0 / counts.clamp_min(1).sqrt()
        self.mn_weights = weights[bank.mn_class[self.mn_train]]
        lists = []
        for cls in bank.classes:
            names = MODELNET_NAMES.get(cls, [cls.replace("_", " ")])
            lists.append([lookup[s] for n in names for s in plain_captions(n)])
            for colour in PALETTE:
                lists.append([lookup[s] for n in names for s in colour_captions(n, colour)])
        self.mn_table = self._table(lists)  # (ids [C*13, K], lengths [C*13])
        self.n_colours = 1 + len(PALETTE)

    @staticmethod
    def _table(lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Ragged caption-id lists -> padded tensor + lengths for vectorised picks."""
        width = max(len(x) for x in lists)
        ids = torch.tensor([x + [x[0]] * (width - len(x)) for x in lists], device=DEVICE)
        return ids, torch.tensor([len(x) for x in lists], device=DEVICE)

    @staticmethod
    def _pick(table: tuple[torch.Tensor, torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
        ids, lengths = table
        column = (torch.rand(len(rows), device=DEVICE) * lengths[rows]).long()
        return ids[rows, column]

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_cap = int(n * 0.55)
        n_plain = int(n * 0.10)
        n_mn = n - n_cap - n_plain
        mirror = lambda k: torch.randint(0, 2, (k,), device=DEVICE)  # noqa: E731

        pick = torch.randint(len(self.cap_shape), (n_cap,), device=DEVICE)
        shapes = self.cap_shape[pick]
        lat_a = self.lat_t2s[shapes, mirror(n_cap)].float()
        txt_a = self.emb[self.cap_text[pick]]

        shapes = self.t2s_train[torch.randint(len(self.t2s_train), (n_plain,), device=DEVICE)]
        lat_b = self.lat_t2s[shapes, mirror(n_plain)].float()
        txt_b = self.emb[self._pick(self.t2s_plain, self.t2s_cat[shapes])]

        rows = self.mn_train[torch.multinomial(self.mn_weights, n_mn, replacement=True)]
        colour = torch.where(torch.rand(n_mn, device=DEVICE) < 0.4,
                             torch.zeros(n_mn, dtype=torch.long, device=DEVICE),
                             torch.randint(1, self.n_colours, (n_mn,), device=DEVICE))
        lat_c = self.lat_mn[rows, colour, mirror(n_mn)].float()
        txt_c = self.emb[self._pick(self.mn_table, self.bank.mn_class[rows] * self.n_colours + colour)]

        return torch.cat([lat_a, lat_b, lat_c]), torch.cat([txt_a, txt_b, txt_c])


@torch.no_grad()
def encode_all(bank: ShapeBank, encoder: Encoder) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior means for every shape, mirrored, and every ModelNet colour."""
    encoder.eval()
    n_colours = len(bank.palette)
    lat_t2s = torch.zeros(bank.n_t2s, 2, LATENT, device=DEVICE)
    lat_mn = torch.zeros(bank.n_mn, n_colours, 2, LATENT, device=DEVICE, dtype=torch.float16)
    amp = torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda")
    for m in (0, 1):
        for chunk in torch.arange(bank.n_t2s, device=DEVICE).split(64):
            x = bank.batch(chunk, mirror=torch.full((len(chunk),), bool(m), device=DEVICE))
            with amp:
                lat_t2s[chunk, m] = encoder(x)[0].float()
        for c in range(n_colours):
            for chunk in torch.arange(bank.n_mn, device=DEVICE).split(64):
                index = chunk + bank.n_t2s
                x = bank.batch(index, colour_id=torch.full((len(chunk),), c, device=DEVICE),
                               mirror=torch.full((len(chunk),), bool(m), device=DEVICE))
                with amp:
                    lat_mn[chunk, c, m] = encoder(x)[0].half()
    log("encoded all latents")
    return lat_t2s, lat_mn


def train_prior(sampler: CaptionSampler, mean: torch.Tensor, std: torch.Tensor) -> Prior:
    prior = Prior().to(DEVICE)
    ema = Prior().to(DEVICE)
    ema.load_state_dict(prior.state_dict())
    log(f"prior params {sum(p.numel() for p in prior.parameters()) / 1e6:.2f}M")
    optimiser = torch.optim.AdamW(prior.parameters(), lr=3e-4, weight_decay=0.01)
    alphas = torch.tensor(cosine_alphas_cumprod(), device=DEVICE, dtype=torch.float32)
    deadline = time.time() + PRIOR_MINUTES * 60
    step, total_steps, probe_started = 0, None, time.time()
    while time.time() < deadline:
        z, cond = sampler.sample(PRIOR_BATCH)
        x0 = (z.float() - mean) / std
        drop = torch.rand(len(cond), device=DEVICE) < COND_DROP
        cond = torch.where(drop.unsqueeze(1), torch.zeros_like(cond), cond)
        t = torch.randint(0, DIFFUSION_T, (len(x0),), device=DEVICE)
        a = alphas[t].unsqueeze(1)
        noise = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        v_target = a.sqrt() * noise - (1 - a).sqrt() * x0
        loss = F.mse_loss(prior(xt, t.float(), cond), v_target)
        if total_steps is not None:
            progress = min(step / total_steps, 1.0)
            lr = 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + math.cos(math.pi * progress))
        else:
            lr = 3e-4 * min(1.0, (step + 1) / 500)
        for group in optimiser.param_groups:
            group["lr"] = lr
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optimiser.step()
        with torch.no_grad():
            decay = min(0.9995, (1 + step) / (10 + step))
            for e, p in zip(ema.parameters(), prior.parameters()):
                e.mul_(decay).add_(p, alpha=1 - decay)
        step += 1
        if step == (20 if SMOKE else 500) and total_steps is None:
            per_step = (time.time() - probe_started) / step
            total_steps = int(step + (deadline - time.time()) / per_step)
            log(f"prior: {per_step * 1000:.1f} ms/step -> ~{total_steps:,} steps")
        if step % 5000 == 0 or (SMOKE and step % 20 == 0):
            log(f"prior step {step:,} lr {lr:.2e} loss {loss.item():.4f}")
    REPORT["prior_steps"] = step
    REPORT["prior_final_loss"] = round(loss.item(), 4)
    return ema


@torch.no_grad()
def ddim_sample(prior: Prior, cond: torch.Tensor, steps: int = SAMPLE_STEPS,
                guidance: float = GUIDANCE, seed: int = 0) -> torch.Tensor:
    """Deterministic DDIM with classifier-free guidance. Mirrors the JS sampler."""
    alphas = cosine_alphas_cumprod()
    times = np.linspace(DIFFUSION_T - 1, 0, steps).round().astype(int)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(len(cond), LATENT, generator=generator).to(DEVICE)
    null = torch.zeros_like(cond)
    for i, t in enumerate(times):
        a = float(alphas[t])
        a_prev = float(alphas[times[i + 1]]) if i + 1 < len(times) else 1.0
        tt = torch.full((len(x),), float(t), device=DEVICE)
        v_c = prior(x, tt, cond)
        v_u = prior(x, tt, null)
        v = v_u + guidance * (v_c - v_u)
        x0 = math.sqrt(a) * x - math.sqrt(1 - a) * v
        eps = math.sqrt(1 - a) * x + math.sqrt(a) * v
        x0 = x0.clamp(-6, 6)
        x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    return x


# ==========================================================================
# Evaluation, montage, export
# ==========================================================================

TEST_PROMPTS = [
    "a modern minimalist oak dining chair with tapered legs",
    "a red office chair with armrests and wheels",
    "a round glass coffee table with metal legs",
    "a long wooden dining table",
    "a blue sofa",
    "a futuristic silver sports car",
    "an airplane",
    "a floor lamp",
    "a toilet",
    "a guitar",
    "a computer monitor",
    "a bed",
    "a wooden bookshelf",
    "a flower pot",
    "a bench",
    "a green bathtub",
]


@torch.no_grad()
def evaluate_prior(bank: ShapeBank, sampler: CaptionSampler, prior: Prior, decoder: Decoder,
                   mean: torch.Tensor, std: torch.Tensor, encoder: Encoder) -> dict:
    prior.eval(), decoder.eval(), encoder.eval()
    results: dict = {}
    # 1) held-out Text2Shape captions: IoU of the sample with its own shape,
    #    against a caption-shuffled baseline.
    caps = sampler.val_caps[:256]
    if caps:
        cond = torch.stack([sampler.emb[sampler.lookup[c["text"]]] for c in caps])
        target_index = torch.tensor([c["shape"] for c in caps], device=DEVICE)
        z = ddim_sample(prior, cond, seed=1) * std + mean
        own, shuffled = [], []
        for chunk in torch.arange(len(caps), device=DEVICE).split(32):
            logits = decoder(z[chunk])
            target = bank.batch(target_index[chunk])
            own.append(iou(logits, target))
            other = target_index[(chunk + 97) % len(caps)]
            shuffled.append(iou(logits, bank.batch(other)))
        results["t2s_heldout_iou"] = round(torch.cat(own).mean().item(), 4)
        results["t2s_shuffled_iou"] = round(torch.cat(shuffled).mean().item(), 4)

    # 2) ModelNet class prompts: classify each sample by its nearest training
    #    latent and count how often the class matches the prompt.
    reference = sampler.lat_mn[sampler.mn_train, 0, 0].float()
    ref_class = bank.mn_class[sampler.mn_train]
    correct, total = 0, 0
    per_class = {}
    for cls_id, cls in enumerate(bank.classes):
        name = MODELNET_NAMES.get(cls, [cls])[0]
        text = f"{article(name)} {name}"
        cond = sampler.emb[sampler.lookup[text]].unsqueeze(0).repeat(4, 1) \
            if text in sampler.lookup else None
        if cond is None:
            continue
        z = ddim_sample(prior, cond, seed=cls_id) * std + mean
        shape = decoder(z)
        occ = (torch.sigmoid(shape[:, :1]) > 0.5).float()
        mu = encoder(torch.cat([occ, torch.sigmoid(shape[:, 1:]) * occ], 1))[0]
        nearest = torch.cdist(mu, reference).argmin(1)
        hits = (ref_class[nearest] == cls_id).sum().item()
        per_class[cls] = hits / len(cond)
        correct += hits
        total += len(cond)
    results["modelnet_class_accuracy"] = round(correct / max(total, 1), 4)
    results["modelnet_per_class"] = {k: round(v, 2) for k, v in per_class.items()}
    return results


def first_hit(occ: np.ndarray, rgb: np.ndarray, axis: int) -> np.ndarray:
    """Shaded colour of the first voxel met looking down ``axis`` from its high end.

    Returns an image indexed by the two remaining axes in increasing order.
    """
    vol = np.moveaxis(occ, axis, 0)[::-1]
    colour = np.moveaxis(rgb.transpose(1, 2, 3, 0), axis, 0)[::-1]  # [depth, a, b, 3]
    hit = vol.any(0)
    depth = vol.argmax(0)
    a, b = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
    image = colour[depth, a, b] * (1.0 - 0.45 * depth[..., None] / RES)
    image[~hit] = 1.0
    return image


def render_views(occ: np.ndarray, rgb: np.ndarray) -> list[np.ndarray]:
    """Third-angle views in the canonical frame (x right, y up, z front).

    front: camera on +z  -> rows = y (up at top), cols = x
    side:  camera on +x  -> rows = y, cols = -z (the object's front on the left)
    top:   camera on +y  -> rows = z (front at the bottom), cols = x
    """
    front = np.flipud(first_hit(occ, rgb, 2).transpose(1, 0, 2))        # [y, x] -> flip y
    side = np.flipud(first_hit(occ, rgb, 0))[:, ::-1]                   # [y, z] -> flip y, -z
    top = first_hit(occ, rgb, 1).transpose(1, 0, 2)                     # [z, x]
    return [front, side, top]


@torch.no_grad()
def montage_prompts(prior: Prior, decoder: Decoder, encoder_text: TextEncoder,
                    mean: torch.Tensor, std: torch.Tensor, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cond = torch.from_numpy(encoder_text.encode(TEST_PROMPTS)).to(DEVICE)
    z = ddim_sample(prior, cond, seed=7) * std + mean
    fig, axes = plt.subplots(len(TEST_PROMPTS), 3, figsize=(5.4, 1.9 * len(TEST_PROMPTS)))
    for row, prompt in enumerate(TEST_PROMPTS):
        out = decoder(z[row:row + 1])
        occ = (torch.sigmoid(out[0, 0]) > 0.5).cpu().numpy()
        rgb = torch.sigmoid(out[0, 1:]).cpu().numpy()
        for col, image in enumerate(render_views(occ, rgb)):
            axes[row, col].imshow(np.clip(image, 0, 1), interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(prompt[:30], fontsize=6)
    for col, title in enumerate(("front", "side", "top")):
        axes[0, col].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"samples_{name}.png", dpi=100)
    plt.close(fig)


class DecoderExport(nn.Module):
    """Standardised latent in, probabilities out - what the browser runs."""

    def __init__(self, decoder: Decoder, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.decoder = decoder
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, z_std):
        logits = self.decoder(z_std * self.std + self.mean)
        return torch.sigmoid(logits[:, :1]).squeeze(1), torch.sigmoid(logits[:, 1:])


def fp16_initializers(path: Path, min_elements: int = 1024) -> None:
    """Store large float initialisers as fp16 + Cast, halving the download.

    Compute stays fp32: onnxruntime folds the Cast at session creation.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    new_inits, casts = [], []
    for init in list(graph.initializer):
        array = numpy_helper.to_array(init)
        if array.dtype != np.float32 or array.size < min_elements:
            continue
        half = numpy_helper.from_array(array.astype(np.float16), init.name + "__fp16")
        new_inits.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__cast"))
        graph.initializer.remove(init)
    graph.initializer.extend(new_inits)
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def round_to_fp16(module: nn.Module) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(p.half().float())


def export(prior: Prior, decoder: Decoder, mean: torch.Tensor, std: torch.Tensor) -> dict:
    import onnxruntime as ort

    prior = prior.float().eval().cpu()
    wrapped = DecoderExport(decoder.float().eval().cpu(), mean.cpu(), std.cpu()).eval()
    round_to_fp16(prior)
    round_to_fp16(wrapped)

    def onnx_export(module, args, path, **kwargs):
        # The TorchScript exporter gives plain opset-17 graphs that
        # onnxruntime-web runs; newer torch defaults to the dynamo exporter.
        try:
            torch.onnx.export(module, args, str(path), opset_version=17, dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(module, args, str(path), opset_version=17, **kwargs)

    dec_path = MODEL_DIR / "decoder.onnx"
    onnx_export(wrapped, (torch.zeros(1, LATENT),), dec_path,
                input_names=["z"], output_names=["occupancy", "rgb"])
    prior_path = MODEL_DIR / "prior.onnx"
    onnx_export(prior, (torch.zeros(2, LATENT), torch.zeros(2), torch.zeros(2, EMBED_DIM)),
                prior_path, input_names=["x", "t", "cond"], output_names=["v"],
                dynamic_axes={"x": {0: "batch"}, "t": {0: "batch"}, "cond": {0: "batch"},
                              "v": {0: "batch"}})
    for path in (dec_path, prior_path):
        fp16_initializers(path)

    # Parity: onnxruntime vs torch on random inputs.
    parity = {}
    z = torch.randn(1, LATENT)
    session = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    occ_onnx, rgb_onnx = session.run(None, {"z": z.numpy()})
    with torch.no_grad():
        occ_t, rgb_t = wrapped(z)
    parity["decoder_max_abs"] = float(max(np.abs(occ_onnx - occ_t.numpy()).max(),
                                          np.abs(rgb_onnx - rgb_t.numpy()).max()))
    session = ort.InferenceSession(str(prior_path), providers=["CPUExecutionProvider"])
    x, t, c = torch.randn(2, LATENT), torch.tensor([10.0, 900.0]), torch.randn(2, EMBED_DIM)
    v_onnx = session.run(None, {"x": x.numpy(), "t": t.numpy(), "cond": c.numpy()})[0]
    with torch.no_grad():
        v_t = prior(x, t, c)
    parity["prior_max_abs"] = float(np.abs(v_onnx - v_t.numpy()).max())
    parity["decoder_bytes"] = dec_path.stat().st_size
    parity["prior_bytes"] = prior_path.stat().st_size
    log("export parity", parity)
    if parity["decoder_max_abs"] > 1e-3 or parity["prior_max_abs"] > 1e-3:
        raise RuntimeError(f"ONNX parity failed: {parity}")

    # Reference vector for the JS/Python samplers: fixed noise, fixed prompt.
    return parity


def main() -> None:
    ensure_packages("onnxruntime", "onnx", "tokenizers")
    torch.manual_seed(0)
    np.random.seed(0)
    log("device", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
    bank = ShapeBank()
    REPORT["orient"] = {k: {"perm": list(v["perm"]), "flip": list(v["flip"])} for k, v in ORIENT.items()}

    encoder, decoder = train_vae(bank)
    REPORT["vae_eval"] = evaluate_vae(bank, encoder, decoder)
    log("VAE eval", REPORT["vae_eval"])

    lat_t2s, lat_mn = encode_all(bank, encoder)
    stack = torch.cat([lat_t2s.reshape(-1, LATENT), lat_mn.float().reshape(-1, LATENT)])
    mean, std = stack.mean(0), stack.std(0).clamp_min(1e-3)
    sampler = CaptionSampler(bank, lat_t2s, lat_mn)
    prior = train_prior(sampler, mean, std)
    torch.save({"prior": prior.state_dict(), "mean": mean, "std": std}, OUT / "prior.pt")

    # Evaluate with fp16-rounded weights - exactly what ships.
    round_to_fp16(prior)
    round_to_fp16(decoder)
    REPORT["prior_eval"] = evaluate_prior(bank, sampler, prior, decoder, mean, std, encoder)
    log("prior eval", REPORT["prior_eval"])
    text_encoder = TextEncoder(fetch_minilm(Path(os.environ.get("T2V_MINILM", "/tmp/minilm"))))
    montage_prompts(prior, decoder, text_encoder, mean, std, "final")

    # Reference sample for cross-checking the browser/backend samplers.
    ref_prompt = "a red office chair with armrests and wheels"
    ref_cond = torch.from_numpy(text_encoder.encode([ref_prompt])).to(DEVICE)
    parity = export(prior, decoder, mean, std)

    alphas = cosine_alphas_cumprod()
    meta = {
        "name": "Text2Voxel-64",
        "version": 1,
        "created": time.strftime("%Y-%m-%d"),
        "resolution": RES,
        "latent_dim": LATENT,
        "cond_dim": EMBED_DIM,
        "frame": "index [x, y, z]; x right, y up, z front (+z faces the viewer)",
        "diffusion": {
            "parameterization": "v",
            "train_steps": DIFFUSION_T,
            "schedule": "cosine",
            "alphas_cumprod": [round(float(a), 10) for a in alphas],
            "sample_steps": SAMPLE_STEPS,
            "guidance": GUIDANCE,
            "x0_clip": 6.0,
            "null_cond": "zeros",
        },
        "text_encoder": {
            "repo": MINILM_REPO, "revision": MINILM_REVISION,
            "file": "onnx/model_quantized.onnx", "pooling": "mean", "normalize": True,
            "max_tokens": MAX_TOKENS,
        },
        "threshold": 0.5,
        "classes_modelnet": bank.classes,
        "palette": {"neutral": list(NEUTRAL_RGB), **{k: list(v) for k, v in PALETTE.items()}},
        "metrics": {"vae": REPORT["vae_eval"], "prior": {
            k: v for k, v in REPORT["prior_eval"].items() if k != "modelnet_per_class"}},
        "training": {
            "data": {
                "text2shape": {"shapes": bank.n_t2s, "source": "http://text2shape.stanford.edu/",
                               "licence": "ShapeNet terms of use (non-commercial research)"},
                "modelnet40": {"shapes": bank.n_mn, "source": "https://modelnet.cs.princeton.edu/",
                               "licence": "academic research use"},
            },
            "vae_steps": REPORT["vae_steps"], "prior_steps": REPORT["prior_steps"],
            "hardware": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        },
        "parity": parity,
    }
    # Reference output: the exact latent the samplers must reproduce given the
    # same Gaussian noise (seed + generator are Python-side only, so the noise
    # itself is stored).
    generator = torch.Generator(device="cpu").manual_seed(123)
    noise = torch.randn(1, LATENT, generator=generator)
    meta["reference"] = {"prompt": ref_prompt, "noise": [round(float(v), 6) for v in noise[0]]}
    prior_cpu = prior.float().cpu().eval()
    x = noise.clone()
    cond = ref_cond.cpu()
    times = np.linspace(DIFFUSION_T - 1, 0, SAMPLE_STEPS).round().astype(int)
    with torch.no_grad():
        for i, t in enumerate(times):
            a = float(alphas[t])
            a_prev = float(alphas[times[i + 1]]) if i + 1 < len(times) else 1.0
            tt = torch.full((1,), float(t))
            v_c, v_u = prior_cpu(x, tt, cond), prior_cpu(x, tt, torch.zeros_like(cond))
            v = v_u + GUIDANCE * (v_c - v_u)
            x0 = (math.sqrt(a) * x - math.sqrt(1 - a) * v).clamp(-6, 6)
            eps = math.sqrt(1 - a) * x + math.sqrt(a) * v
            x = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    meta["reference"]["latent"] = [round(float(v), 5) for v in x[0]]
    meta["reference"]["cond_head"] = [round(float(v), 6) for v in cond[0, :8]]
    write_json(MODEL_DIR / "meta.json", meta)

    REPORT["seconds"] = round(time.time() - T0)
    write_json(OUT / "k2_report.json", REPORT)
    log("K2 COMPLETE")


if __name__ == "__main__":
    main()
