"""Memory-careful streaming QEP quantizer for DeepSeek-V4-Flash.

DeepSeek-V4-Flash breaks OneCompression's generic ``run_quantize_with_qep_arch``
loop in three ways: (1) a bf16 dequant of the 284B-param model is ~280GB and
won't fit host RAM; (2) one bf16 block is ~12.8GB so the loop's
``block_f = deepcopy(block_q)`` needs ~25.8GB > 24GB GPU; (3) all 768 expert
Hessians at once is ~51GB. This driver sidesteps all three with a
**FP-first / quantize-in-place / then-propagate** scheme that keeps exactly ONE
block resident on the GPU at a time:

  per layer L (one bf16 block on GPU, experts included, ~12.8GB):
    1. FP pass   : forward inps_f, hook targets -> capture x_f; save out -> next inps_f
    2. Q   pass  : forward inps_q (still-FP block), hook targets -> capture x_q
    3. quantize  : H = (2/n) Xq^T Xq ; cross-term C = (2/n)(Xf - Xq)^T Xq
                   dense modules use QEP weight-correction with C (the inter-block
                   error-propagation term); experts use Hessian only (routing
                   mismatch makes a cross-term invalid).  Assign dequantized weight.
    4. propagate : forward inps_q through the now-quantized block -> next inps_q

The cross-term ``C`` is zero on layer 0 (inps_q == inps_f) and grows as the
quantized chain diverges from the FP chain — this is exactly QEP's contribution.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import json
import gc
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import nn

from onecomp.adapters.dsv4_weights import Dsv4ShardReader


def _build_args(cfg: dict, max_seq_len: int, max_batch_size: int):
    import model as M  # shipped inference/model.py (on PYTHONPATH)

    fn = {f.name for f in fields(M.ModelArgs)}
    kw = {k: v for k, v in cfg.items() if k in fn}
    kw.update(
        dtype="bf16", expert_dtype=None, scale_fmt=None, scale_dtype="fp32",
        n_layers=cfg["num_hidden_layers"], n_hash_layers=cfg["num_hash_layers"],
        moe_inter_dim=cfg["moe_intermediate_size"],
        n_activated_experts=cfg["num_experts_per_tok"],
        compress_ratios=tuple(cfg["compress_ratios"]),
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,
    )
    return M.ModelArgs(**kw)


class _CastLinear(nn.Linear):
    """nn.Linear that casts its input to the weight dtype before the matmul.

    DeepSeek-V4's custom ``linear()`` does a plain ``F.linear(x, weight)`` with no
    input→weight dtype coercion, and some internal paths (the Compressor's MLA, which
    runs RMSNorm in fp32) feed a float32 activation into a bf16 weight.  The stock
    model never crashes because its fp8/fp4 kernels handle that; a swapped nn.Linear
    would.  Casting here preserves the model's dtype-robustness while staying
    ``isinstance(_, nn.Linear)`` for the quantizer/QEP hooks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias)


def _swap_linears(mod, M) -> int:
    """Replace DeepSeek-V4 custom Linear with torch.nn.Linear (same bf16 weight)
    so OneCompression's nn.Linear-based quantizer + QEP loop accept the block."""
    n = 0
    for name, child in list(mod.named_children()):
        if isinstance(child, M.Linear) and not isinstance(child, nn.Linear):
            lin = _CastLinear(child.in_features, child.out_features,
                              bias=child.bias is not None)
            lin.weight = nn.Parameter(child.weight.data, requires_grad=False)
            if child.bias is not None:
                lin.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(mod, name, lin)
            n += 1
        else:
            n += _swap_linears(child, M)
    return n


class Dsv4StreamingQEP:
    def __init__(self, ckpt_dir: str, device: str = "cuda",
                 max_seq_len: int = 4096, max_batch_size: int = 1):
        import model as M
        self.M = M
        self.ckpt_dir = ckpt_dir
        self.device = torch.device(device)
        self.reader = Dsv4ShardReader(ckpt_dir)
        self.cfg = json.load(open(f"{ckpt_dir}/config.json"))
        self.args = _build_args(self.cfg, max_seq_len, max_batch_size)
        # force single-GPU bf16 module globals
        M.world_size, M.rank = 1, 0
        M.default_dtype = torch.bfloat16
        M.scale_fmt, M.scale_dtype, M.block_size = None, torch.float32, 128

    # -- layer construction -------------------------------------------------
    def build_layer(self, L: int) -> nn.Module:
        block = self.M.Block(L, self.args)
        prefix = f"layers.{L}."
        sd = {}
        for name in self.reader.weight_map:
            if not name.startswith(prefix):
                continue
            short = name[len(prefix):]
            if short.endswith(".scale"):
                continue
            if short.endswith(".weight") and self.reader.has(name.replace(".weight", ".scale")):
                sd[short] = self.reader.dequant_linear(name[:-len(".weight")], torch.bfloat16)
            else:
                sd[short] = self.reader.plain(name)
        block.load_state_dict(sd, strict=False)
        _swap_linears(block, self.M)
        return block.to(self.device).eval()

    # -- calibration input --------------------------------------------------
    def build_calib_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [N, S] (cpu/cuda int) -> initial mHC hidden [N, S, hc, dim] bf16."""
        embed_w = self.reader.plain("embed.weight", torch.bfloat16).to(self.device)
        h = F.embedding(input_ids.to(self.device), embed_w)
        h = h.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)
        del embed_w
        return h

    # -- target selection ---------------------------------------------------
    @staticmethod
    def _target_modules(block) -> dict[str, nn.Linear]:
        """name -> nn.Linear for every quantizable Linear (dense attn + shared +
        routed experts). Excludes gate (it's the router, kept fp)."""
        out = {}
        for name, m in block.named_modules():
            if isinstance(m, nn.Linear) and "gate" not in name:
                out[name] = m
        return out

    @staticmethod
    def _is_expert(name: str) -> bool:
        # routed experts only (".experts.N."); shared_experts process every token
        # so they keep the dense QEP cross-term path.
        return ".experts." in name

    # -- the streaming quantize loop ---------------------------------------
    @torch.no_grad()
    def quantize(self, input_ids: torch.Tensor, *, wbits_dense: int = 4,
                 wbits_expert: int = 2, groupsize: int = 128, batch_size: int = 1,
                 out_dir: str | None = None, n_layers: int | None = None):
        """Quantize every block sequentially keeping ONE bf16 block on GPU.

        ``input_ids``: [N, S] int calibration tokens (host).  For each layer the
        loop (1) runs the still-FP block on the FP chain inps_f capturing per-target
        x_f, (2) runs it on the quantized chain inps_q capturing x_q, accumulating
        the QEP Hessian H=(2/n)Xq^T Xq and cross-term C=(2/n)(Xf-Xq)^T Xq for dense /
        shared modules and a Hessian-only X^T X for routed experts (routing makes a
        cross-term ill-defined), (3) quantizes — dense @ wbits_dense with C, experts
        @ wbits_expert without — assigning the dequantized weight back in place, then
        (4) forwards the now-quantized block to advance inps_q.  Packed results are
        streamed to ``out_dir/layer_{L}.pt`` so host RAM never holds the whole model.
        """
        import os
        from onecomp.quantizer.gptq import GPTQ

        dev = self.device
        L_total = n_layers if n_layers is not None else self.args.n_layers
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        ids = input_ids.to(torch.long)
        N = ids.size(0)
        inps_f = self.build_calib_inputs(ids).cpu()   # FP chain hidden [N,S,hc,dim]
        inps_q = inps_f.clone()                        # layer 0: identical chains

        for L in range(L_total):
            t0 = __import__("time").time()

            # ---- resume: a saved layer only needs its two chains advanced ----
            done = out_dir and os.path.exists(f"{out_dir}/layer_{L}.pt")
            if done:
                block = self.build_layer(L)
                of = []
                for first in range(0, N, batch_size):
                    last = min(first + batch_size, N)
                    with torch.inference_mode(), torch.device(dev):
                        o = block(inps_f[first:last].to(dev, torch.bfloat16), 0,
                                  ids[first:last].to(dev))
                    of.append(o.detach().to(torch.bfloat16).cpu())
                inps_f = torch.cat(of, 0)
                saved = torch.load(f"{out_dir}/layer_{L}.pt", map_location="cpu")
                tgt = self._target_modules(block)
                for n, m in tgt.items():
                    if n in saved["modules"]:
                        m.weight.data = self._dequant_saved(saved["modules"][n]).to(dev).to(torch.bfloat16)
                oq = []
                for first in range(0, N, batch_size):
                    last = min(first + batch_size, N)
                    with torch.inference_mode(), torch.device(dev):
                        o = block(inps_q[first:last].to(dev, torch.bfloat16), 0,
                                  ids[first:last].to(dev))
                    oq.append(o.detach().to(torch.bfloat16).cpu())
                inps_q = torch.cat(oq, 0)
                print(f"[L{L:02d}] RESUMED from saved in {__import__('time').time()-t0:.1f}s "
                      f"(out_q std={inps_q.float().std().item():.4f})")
                del block, saved, tgt, of, oq
                gc.collect(); torch.cuda.empty_cache()
                continue

            block = self.build_layer(L)
            targets = self._target_modules(block)
            dense = {n: m for n, m in targets.items() if not self._is_expert(n)}
            experts = {n: m for n, m in targets.items() if self._is_expert(n)}

            # accumulators ---------------------------------------------------
            H = {n: torch.zeros(m.in_features, m.in_features, device=dev, dtype=torch.float32) for n, m in dense.items()}
            C = {n: torch.zeros(m.in_features, m.in_features, device=dev, dtype=torch.float32) for n, m in dense.items()}
            ns = {n: 0 for n in dense}
            ex_in: dict[str, list] = {}            # routed-expert inputs (CPU, sparse)

            out_f_chunks = []
            dest_f, dest_q = {}, {}

            def mk(store, nm):
                def h(mod, inp, _):
                    store[nm] = inp[0] if isinstance(inp, tuple) else inp
                return h

            for first in range(0, N, batch_size):
                last = min(first + batch_size, N)
                xb_f = inps_f[first:last].to(dev, torch.bfloat16)
                xb_q = inps_q[first:last].to(dev, torch.bfloat16)
                ib = ids[first:last].to(dev)

                with torch.inference_mode(), torch.device(dev):
                    # --- FP pass: capture x_f, advance FP chain ---
                    hs = [m.register_forward_hook(mk(dest_f, n)) for n, m in dense.items()]
                    out_f = block(xb_f, 0, ib)
                    for hd in hs:
                        hd.remove()
                    out_f_chunks.append(out_f.detach().to(torch.bfloat16).cpu())

                    # --- Q pass: capture x_q (dense) + expert inputs ---
                    hs = [m.register_forward_hook(mk(dest_q, n)) for n, m in dense.items()]
                    hs += [m.register_forward_hook(mk(dest_q, n)) for n, m in experts.items()]
                    _ = block(xb_q, 0, ib)
                    for hd in hs:
                        hd.remove()

                # online H / C for dense+shared modules.  A few projections
                # (e.g. attn.wo_a) are applied as a raw einsum on ``.weight`` and
                # never trigger the module hook — those stay at ns==0 and fall back
                # to identity-Hessian RTN at quantize time.
                for n, m in dense.items():
                    if n not in dest_q or n not in dest_f:
                        continue
                    xq = dest_q[n].reshape(-1, m.in_features).float()
                    xf = dest_f[n].reshape(-1, m.in_features).float()
                    t = xq.size(0)
                    H[n] *= ns[n] / (ns[n] + t)
                    C[n] *= ns[n] / (ns[n] + t)
                    ns[n] += t
                    s = (2.0 / ns[n]) ** 0.5
                    xqs, xfs = s * xq, s * xf
                    H[n] += xqs.t() @ xqs
                    C[n] += (xfs - xqs).t() @ xqs
                # stash routed-expert inputs (sparse → cheap on CPU)
                for n in experts:
                    if n in dest_q and dest_q[n].numel():
                        ex_in.setdefault(n, []).append(
                            dest_q[n].reshape(-1, experts[n].in_features).float().cpu())
                dest_f.clear(); dest_q.clear()

            inps_f = torch.cat(out_f_chunks, 0)  # next FP-chain input
            del out_f_chunks

            # --- quantize dense / shared with QEP cross-term ---
            qd = GPTQ(wbits=wbits_dense, groupsize=groupsize, sym=False,
                      actorder=False, calc_quant_error=False)
            qd.setup(block)
            packed = {}
            cmax = 0.0
            for n, m in dense.items():
                if ns[n] == 0:  # einsum-applied weight (no activation hook) → RTN
                    Hn, Cn = torch.eye(m.in_features, device=dev, dtype=torch.float32), None
                else:
                    cmax = max(cmax, C[n].abs().max().item())
                    Hn, Cn = H[n].clone(), C[n].clone()
                qd.quantize_with_qep(m, quant_input_activation=None,
                                     hessian=Hn, delta_hatX=Cn)
                res = qd.results[n]
                m.weight.data = res.compute_dequantized_weight().to(dev).to(torch.bfloat16)
                packed[n] = self._pack(res)
                del H[n], C[n]
            del qd; torch.cuda.empty_cache()

            # --- quantize routed experts, Hessian only ---
            qe = GPTQ(wbits=wbits_expert, groupsize=groupsize, sym=False,
                      actorder=False, calc_quant_error=False)
            qe.setup(block)
            for n, m in experts.items():
                if n in ex_in:
                    X = torch.cat(ex_in[n], 0).to(dev)
                    He = (2.0 / X.size(0)) * (X.t() @ X)
                    He = He.float()
                    del X, ex_in[n]
                else:  # expert never routed in calib → damping-only (RTN-like)
                    He = torch.eye(m.in_features, device=dev, dtype=torch.float32)
                qe.quantize_with_qep(m, quant_input_activation=None, hessian=He, delta_hatX=None)
                res = qe.results[n]
                m.weight.data = res.compute_dequantized_weight().to(dev).to(torch.bfloat16)
                packed[n] = self._pack(res)
                del He
            del qe; torch.cuda.empty_cache()

            # --- propagate: forward quantized block to advance inps_q ---
            out_q_chunks = []
            for first in range(0, N, batch_size):
                last = min(first + batch_size, N)
                with torch.inference_mode(), torch.device(dev):
                    out_q = block(inps_q[first:last].to(dev, torch.bfloat16), 0,
                                  ids[first:last].to(dev))
                out_q_chunks.append(out_q.detach().to(torch.bfloat16).cpu())
            inps_q = torch.cat(out_q_chunks, 0)
            del out_q_chunks

            print(f"[L{L:02d}] {len(dense)} dense + {len(experts)} experts quantized "
                  f"in {__import__('time').time()-t0:.1f}s "
                  f"(crossterm|C|max={cmax:.3e} out_q std={inps_q.float().std().item():.4f})")
            if out_dir:
                torch.save({"layer": L, "wbits_dense": wbits_dense,
                            "wbits_expert": wbits_expert, "groupsize": groupsize,
                            "modules": packed}, f"{out_dir}/layer_{L}.pt")
            # drop EVERY reference to the old block's GPU modules — the targets /
            # dense / experts dicts hold its nn.Linear weights alive otherwise, so
            # the next build_layer().to(device) would stack a 2nd 12.8GB → OOM.
            del block, packed, H, C, ns, ex_in, targets, dense, experts
            gc.collect(); torch.cuda.empty_cache()

        return inps_q

    @staticmethod
    def _pack(res) -> dict:
        """Extract bit-packed int weights from a GPTQResult for disk storage.

        ``qweight`` comes back unpacked (one int per element, ~25GB/layer for the
        768 experts).  Bit-pack ``wbits`` codes into uint8 lanes along the in-dim so
        a layer's int4/int2 weights land near their information size (~3GB/layer)."""
        wbits, gs = res.wbits, res.groupsize
        out = {"wbits": wbits, "groupsize": gs, "sym": res.sym}
        qw = res.qweight
        if qw is not None:
            qw = qw.to(torch.int32).cpu()
            out_f, in_f = qw.shape
            per = 8 // wbits  # int4 -> 2/byte, int2 -> 4/byte
            pad = (-in_f) % per
            if pad:
                qw = torch.cat([qw, torch.zeros(out_f, pad, dtype=torch.int32)], 1)
            qw = qw.view(out_f, -1, per)
            shift = (torch.arange(per, dtype=torch.int32) * wbits)
            packed = (qw << shift).sum(-1).to(torch.uint8)  # [out, ceil(in/per)]
            out["qweight_packed"] = packed
            out["in_features"] = in_f
        for k in ("scales", "qzeros", "perm"):
            v = getattr(res, k, None)
            if v is not None:
                out[k] = v.half().cpu() if v.is_floating_point() else v.cpu()
        return out

    @staticmethod
    def _dequant_saved(p: dict) -> torch.Tensor:
        """Inverse of ``_pack``: unpack the bit-packed uint8 lanes and dequantize
        back to a (fp16) weight via the GPTQ result reconstruction path."""
        from onecomp.quantizer.gptq._gptq import GPTQResult
        wbits = p["wbits"]
        per = 8 // wbits
        packed = p["qweight_packed"].to(torch.int32)            # [out, ncol]
        out_f, ncol = packed.shape
        shift = torch.arange(per, dtype=torch.int32) * wbits
        qw = (packed.unsqueeze(-1) >> shift) & ((1 << wbits) - 1)  # [out, ncol, per]
        qw = qw.reshape(out_f, ncol * per)[:, :p["in_features"]]
        res = GPTQResult(wbits=wbits, groupsize=p["groupsize"], sym=p.get("sym", False),
                         actorder=False, qweight=qw, scales=p["scales"],
                         qzeros=p["qzeros"], perm=p.get("perm"))
        return res.compute_dequantized_weight()
