"""nanoVLA: SigLIP + 2-layer GELU MLP projector + Qwen2.5-0.5B with vocab-tail action tokens.

The conceptual heart, in ~30 lines of forward():
  1. Vision tower turns each image into N patch tokens.
  2. Projector maps vision tokens into the LM's hidden size (LLaVA-style).
  3. LM input = [image_tokens, instruction_tokens, action_query × T_act]
     — action positions are filled with T_act *per-position* learned query
     embeddings, not the ground-truth action tokens. The LM "fills them in."
  4. LM forward gives logits at every position; all action tokens are
     predicted in ONE pass (parallel decoding, OFT-style).
  5. Cross-entropy is computed ONLY at action positions (image+instruction
     positions are masked out of the loss).

Three non-obvious moves, all there to keep the line count low:

  (a) The vocab-tail trick. We discretize each action dim into 256 bins and map
      bin index b -> LM token id (V - 256 + b), where V is the LM tokenizer
      vocab size. The last 256 tokens of Qwen2.5's tokenizer are byte-fallback /
      reserved tokens that are never produced by normal text training, so
      colliding with them is harmless. This means: action labels are valid LM
      token ids, the LM's existing input embeddings double as action embeddings,
      and the LM head IS the action head. No new params, no vocab resize, no
      separate action head module.

  (b) Loss masking via slicing, not -100 labels. We slice action-position
      logits out of the full logits tensor and run cross-entropy directly,
      instead of building a full-length labels tensor padded with -100 at the
      image/text positions. Same math, ~5 fewer lines, easier to read.

  (c) Parallel action decoding (OFT-style). Per-position learned query
      embeddings fill every action slot; the action region of the attention
      mask is *bidirectional* (every slot sees every other), making the chunk
      a true joint prediction. RoPE adds a second axis of slot identity on
      top of the per-position learnables. One forward per chunk, no teacher
      forcing. See README#parallel-action-decoding for the full rationale.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# SigLIP normalization constants (siglip-base-patch16-224 uses 0.5/0.5/0.5).
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclass
class VLAConfig:
    vision_model_id: str = "google/siglip-base-patch16-224"
    lm_model_id: str = "Qwen/Qwen2.5-0.5B"
    use_wrist_camera: bool = False
    chunk_size: int = 8
    action_dim: int = 7
    num_bins: int = 256
    image_size: int = 224
    freeze_vision: bool = True
    prompt_template: str = "In: What action should the robot take to {instruction}? Out:"
    max_instruction_tokens: int = 64


def preprocess_image(arr_uint8: torch.Tensor, image_size: int) -> torch.Tensor:
    """uint8 (B,H,W,3) -> float32 (B,3,image_size,image_size), SigLIP-normalized."""
    x = arr_uint8.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
    mean = x.new_tensor(SIGLIP_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(SIGLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------- Action tokenizer ----------

class ActionTokenizer:
    """Per-dim uniform 256-bin discretizer with the vocab-tail mapping.

    Bin b for action dim d is encoded as LM token id (V - num_bins + b),
    where V is the LM vocab size. Reverse mapping decodes a token id back
    to a continuous action value via the stored q01/q99 quantiles.
    """

    def __init__(self, q01, q99, num_bins: int, vocab_size: int):
        self.q01 = np.asarray(q01, dtype=np.float32)
        self.q99 = np.asarray(q99, dtype=np.float32)
        self.num_bins = num_bins
        self.vocab_size = vocab_size
        self.token_offset = vocab_size - num_bins
        self.span = np.maximum(self.q99 - self.q01, 1e-6)

    def _to_bins(self, a: np.ndarray) -> np.ndarray:
        normed = (np.clip(a, self.q01, self.q99) - self.q01) / self.span
        return np.round(normed * (self.num_bins - 1)).astype(np.int64)

    def encode(self, actions):
        """actions: array of shape (..., action_dim) in raw units.
        Returns LM token ids of the same shape (matching input type)."""
        if isinstance(actions, torch.Tensor):
            bins = self._to_bins(actions.detach().cpu().numpy())
            return torch.from_numpy(bins + self.token_offset)
        return self._to_bins(np.asarray(actions)) + self.token_offset

    def decode(self, token_ids) -> np.ndarray:
        """token_ids: (..., action_dim) -> action values, np.float32."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        bins = np.clip(np.asarray(token_ids) - self.token_offset, 0, self.num_bins - 1)
        normed = bins.astype(np.float32) / max(self.num_bins - 1, 1)
        return normed * self.span + self.q01


# ---------- Vision tower & projector ----------

class VisionTower(nn.Module):
    """Wraps a HF SigLIP vision encoder; returns patch tokens (no pool)."""

    def __init__(self, vision_model_id: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(vision_model_id).vision_model
        self.hidden_size = self.encoder.config.hidden_size
        self.patch_size = self.encoder.config.patch_size

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state


class Projector(nn.Module):
    """LLaVA-style 2-layer MLP with GELU."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)


# ---------- NanoVLA ----------

class NanoVLA(nn.Module):
    """SigLIP -> Projector -> Qwen2.5-0.5B LM, with vocab-tail action tokens.

    The policy interface (eval_libero.py contract):
        chunk_size: int
        predict(images: dict, instruction: str) -> (chunk_size, action_dim) np.float32
    """

    def __init__(self, config: VLAConfig, action_q01, action_q99):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad once, here. predict() and Collator both rely on this.
        # See the Collator docstring and the attention-mask construction for
        # why left + fixed-length is load-bearing.
        self.tokenizer.padding_side = "left"
        self._struct_mask_cache: dict = {}
        self.lm = AutoModelForCausalLM.from_pretrained(config.lm_model_id)
        self.vocab_size = self.lm.config.vocab_size

        self.vision = VisionTower(config.vision_model_id)
        if config.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.proj = Projector(self.vision.hidden_size, self.lm.config.hidden_size)
        self.action_tokenizer = ActionTokenizer(
            q01=action_q01, q99=action_q99,
            num_bins=config.num_bins, vocab_size=self.vocab_size,
        )
        # Per-position learnable embedding (one per action slot). Each slot
        # carries its own identity in the embedding itself; RoPE adds a second
        # axis of distinguishability on top. OpenVLA-OFT uses zero/empty
        # embeddings + RoPE only; we add per-position learnables for a richer
        # slot identity.
        self.action_query = nn.Parameter(
            torch.zeros(1, self.action_seq_len, self.lm.config.hidden_size)
        )
        nn.init.normal_(self.action_query, std=0.02)

    @property
    def chunk_size(self):
        return self.config.chunk_size

    @property
    def action_seq_len(self):
        return self.config.chunk_size * self.config.action_dim

    # ---- shared encode helpers ----

    def _encode_images(self, primary_uint8, wrist_uint8=None):
        """uint8 (B,H,W,3) -> projected tokens (B, N, D_lm)."""
        primary = preprocess_image(primary_uint8, self.config.image_size)
        feats = self.vision(primary)
        if wrist_uint8 is not None:
            wrist = preprocess_image(wrist_uint8, self.config.image_size)
            feats = torch.cat([feats, self.vision(wrist)], dim=1)
        return self.proj(feats)

    def _embed_tokens(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def _build_attention_mask(self, instruction_mask, N_img, dtype):
        """4D additive attention mask for [image | prompt | action_query].

        Causal within [image | prompt] (Qwen2.5 default for text), bidirectional
        within the action block (OFT-style parallel decoding — every action
        slot attends to every other, so the chunk is a true joint prediction
        rather than 56 quasi-AR siblings). Pad keys in the prompt are masked.

        Returns (B, 1, L, L) additive mask: 0 where attendable, finfo.min else.
        """
        T_act = self.action_seq_len
        L_text = N_img + instruction_mask.shape[1]
        device = instruction_mask.device
        struct = self._structural_mask(L_text + T_act, L_text, device)     # (L, L) bool
        valid_keys = F.pad(instruction_mask.bool(), (N_img, T_act), value=True)
        attendable = struct[None] & valid_keys[:, None, :]                 # (B, L, L)
        zero = attendable.new_zeros((), dtype=dtype)
        return torch.where(attendable[:, None], zero,
                           torch.full_like(zero, torch.finfo(dtype).min))

    def _structural_mask(self, L: int, L_text: int, device):
        """(L, L) bool mask: causal for [0,L_text), bidirectional for [L_text,L).

        Constant across batches/steps for a given (L, device); cached.
        """
        key = (L, L_text, str(device))
        if key not in self._struct_mask_cache:
            m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
            m[L_text:, L_text:] = True
            self._struct_mask_cache[key] = m
        return self._struct_mask_cache[key]

    # ---- training forward (the conceptual heart) ----

    def forward(self, batch):
        """Compute training loss.

        Expected batch keys (all torch tensors on the right device):
            primary           (B, H, W, 3) uint8       — primary camera frame
            wrist             same                     — only if use_wrist_camera
            instruction_ids   (B, T_text) long         — LM-tokenized instruction (padded)
            instruction_mask  (B, T_text) long/bool    — 1 for real tokens, 0 for pad
            action_token_ids  (B, T_act)  long         — LM-vocab IDs (vocab tail),
                                                         used as TARGETS only.
                                                         T_act = chunk_size * action_dim
        """
        wrist = batch.get("wrist") if self.config.use_wrist_camera else None
        img = self._encode_images(batch["primary"], wrist)             # (B, N_img, D)
        txt = self._embed_tokens(batch["instruction_ids"])             # (B, T_text, D)
        B, T_act = img.shape[0], self.action_seq_len
        act = self.action_query.expand(B, -1, -1).to(txt.dtype)        # (B, T_act, D)

        inputs_embeds = torch.cat([img, txt, act], dim=1)              # (B, L, D)
        attn_mask = self._build_attention_mask(
            batch["instruction_mask"], img.shape[1], inputs_embeds.dtype,
        )

        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn_mask, use_cache=False)
        logits = out.logits                                            # (B, L, V)

        action_logits = logits[:, -T_act:, :]                          # (B, T_act, V)
        loss = F.cross_entropy(
            action_logits.reshape(-1, self.vocab_size),
            batch["action_token_ids"].reshape(-1),
        )
        return loss, action_logits

    # ---- inference ----

    @torch.no_grad()
    def predict(self, images: dict, instruction: str) -> np.ndarray:
        """One-shot parallel decoding over the vocab tail.

        Constraining argmax to the last `num_bins` logits is what makes the
        vocab-tail trick safe at inference: even if the LM gives some text
        token slightly higher logit than every action bin (e.g. early in
        training), we still pick the best ACTION bin and stay on-distribution.
        """
        device = next(self.parameters()).device
        primary = torch.from_numpy(images["primary"][None]).to(device)
        wrist = (torch.from_numpy(images["wrist"][None]).to(device)
                 if self.config.use_wrist_camera else None)

        # Match the training Collator: left-pad to fixed max_instruction_tokens
        # so the action_query RoPE positions are identical to training. The
        # tokenizer was set to padding_side="left" in __init__.
        prompt = self.config.prompt_template.format(instruction=instruction.strip())
        tok = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            padding="max_length",
            max_length=self.config.max_instruction_tokens,
        )
        instruction_ids = tok.input_ids.to(device)
        instruction_mask = tok.attention_mask.to(device)

        # Match training: forward in bf16 autocast so vision (fp32) and LM
        # (often loaded as bf16 from HF config) agree on dtype.
        amp = torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=device.type == "cuda")
        offset = self.vocab_size - self.config.num_bins
        T_act = self.action_seq_len
        with amp:
            img = self._encode_images(primary, wrist)
            txt = self._embed_tokens(instruction_ids)
            seq = torch.cat([
                img, txt, self.action_query.expand(1, -1, -1).to(txt.dtype),
            ], dim=1)
            attn_mask = self._build_attention_mask(instruction_mask, img.shape[1], seq.dtype)
            logits = self.lm(inputs_embeds=seq, attention_mask=attn_mask,
                             use_cache=False).logits[:, -T_act:, :]
            bin_idx = logits[..., offset:].argmax(dim=-1)              # (1, T_act)

        token_ids = (bin_idx + offset).view(T_act).cpu()
        grid = token_ids.view(self.config.chunk_size, self.config.action_dim)
        return self.action_tokenizer.decode(grid).astype(np.float32)

    # ---- save / load ----

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "action_q01": self.action_tokenizer.q01.tolist(),
            "action_q99": self.action_tokenizer.q99.tolist(),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = VLAConfig(**ckpt["config"])
        model = cls(config, action_q01=ckpt["action_q01"], action_q99=ckpt["action_q99"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
