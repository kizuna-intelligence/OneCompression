"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Akihiro Yoshida

"""

import torch
import torch.nn.functional as F
from torch import nn

from onecomp.utils.blockwise import (
    backward_input,
    get_blocks_and_inputs,
    forward_input,
    move_kwargs_to_device,
)


def _find_head_modules(model, blocks, adapter=None):
    """Find the (norm, head) modules that follow the transformer blocks.

    When ``adapter`` is supplied, delegates to
    :meth:`ModelAdapter.get_head_modules` so non-HF architectures (DiT,
    etc.) can return their own ``(out_norm, out_proj)``.  Otherwise
    falls back to attribute-name heuristics for HF causal LMs.

    Returns:
        tuple[nn.Module, nn.Module]: (norm, head)
    """
    if adapter is not None:
        return adapter.get_head_modules(model)

    parent = None
    for name, module in model.named_modules():
        if module is blocks:
            parent_name = name.rpartition(".")[0]
            parent = model.get_submodule(parent_name) if parent_name else model
            break

    norm = None
    for attr in ("norm", "final_layer_norm", "ln_f"):
        norm = getattr(parent, attr, None)
        if norm is not None:
            break

    lm_head = None
    for attr in ("lm_head", "embed_out", "output"):
        lm_head = getattr(model, attr, None)
        if lm_head is not None:
            break

    return norm, lm_head


def _map_candidates_to_blocks(blocks, candidates):
    """Map each candidate (name, module) to its parent block index."""
    module_to_block = {}
    for idx, block in enumerate(blocks):
        for m in block.modules():
            module_to_block[m] = idx

    block_to_candidates = {}
    for name, module in candidates:
        idx = module_to_block.get(module)
        if idx is not None:
            block_to_candidates.setdefault(idx, []).append((name, module))
    return block_to_candidates


def _is_kv_shared_block(block):
    """Return True if the block reuses KV states from an earlier layer.
    """
    attn = getattr(block, "self_attn", None)
    return getattr(attn, "is_kv_shared_layer", False)


def collect_activation_stats_blockwise(
    model,
    candidates,
    calibration_config,
    *,
    use_curvature_b=True,
    batch_size=16,
    device=None,
    logger=None,
    adapter=None,
):
    """Collect full Gram and curvature matrices via block-wise processing.

    When ``adapter`` is supplied, calibration data is produced via the
    adapter (so non-HF architectures can plug in here), and the
    curvature loss for ``b_diag`` uses ``adapter.compute_curvature_loss``.

    Returns:
        tuple[dict, dict]: (a_diag, b_diag)
    """
    if device is None:
        device = torch.device("cuda")

    original_device = next(model.parameters()).device
    if original_device.type != "cpu":
        if logger:
            logger.info("Moving model to CPU for block-wise activation collection")
        model.to("cpu")
        torch.cuda.empty_cache()

    if adapter is not None:
        calib_data = adapter.prepare_calibration_inputs(
            model=model,
            calibration_config=calibration_config,
            device=torch.device("cpu"),
            logger=logger,
        )
        num_samples = calibration_config.num_calibration_samples
        actual_samples = min(num_samples, adapter.num_calibration_samples(calib_data))
        model_inputs = adapter.slice_calibration_inputs(
            calib_data, 0, actual_samples
        )
        seqlen_for_log = getattr(calibration_config, "max_length", "?")
    else:
        from transformers import AutoTokenizer
        from onecomp.calibration import prepare_calibration_dataset

        model_id = getattr(model.config, "_name_or_path", None)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        calib_data = prepare_calibration_dataset(
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            calibration_config=calibration_config,
            model=model,
        )
        num_samples = calibration_config.num_calibration_samples
        actual_samples = min(num_samples, calib_data["input_ids"].shape[0])
        model_inputs = {k: v[:actual_samples] for k, v in calib_data.items()}
        seqlen_for_log = calibration_config.max_length

    blocks, inps, kwargs = get_blocks_and_inputs(
        model, model_inputs, batch_size, adapter=adapter
    )
    kwargs = move_kwargs_to_device(kwargs, device)
    block_to_candidates = _map_candidates_to_blocks(blocks, candidates)

    a_accum = {}
    b_accum = {}
    counts = {}
    for name, module in candidates:
        d_in = module.weight.shape[1]
        d_out = module.weight.shape[0]
        a_accum[name] = torch.zeros(d_in, dtype=torch.float64)
        counts[name] = 0
        if use_curvature_b:
            b_accum[name] = torch.zeros(d_out, dtype=torch.float64)

    if logger:
        tag = "Gram + Curvature" if use_curvature_b else "Gram only"
        logger.info(
            "Block-wise activation stats (%s): %d samples, seqlen=%s, " "%d blocks, %d layers",
            tag,
            actual_samples,
            seqlen_for_log,
            len(blocks),
            len(candidates),
        )

    # Collect a_diag
    saved_inps = [inps]

    for block_idx, block in enumerate(blocks):

        block.to(device)

        hooks = []
        for name, module in block_to_candidates.get(block_idx, []):
            hooks.append(module.register_forward_hook(_make_fwd_hook(name, a_accum, counts)))

        inps = forward_input(inps, block, kwargs, batch_size, device)
        saved_inps.append(inps)

        for h in hooks:
            h.remove()
        block.cpu()
        torch.cuda.empty_cache()

    # Collect b_diag
    if use_curvature_b:
        norm, lm_head = _find_head_modules(model, blocks, adapter=adapter)
        if norm is None or lm_head is None:
            raise RuntimeError(
                "Cannot compute curvature: "
                f"norm={'found' if norm else 'NOT found'}, "
                f"lm_head={'found' if lm_head else 'NOT found'}. "
                "The model may use non-standard module names. "
                "Set use_curvature_b=False to skip curvature estimation."
            )

        grad = _compute_loss_grad(
            saved_inps[-1],
            norm,
            lm_head,
            model_inputs,
            device,
            adapter=adapter,
        )

        for block_idx in range(len(blocks) - 1, -1, -1):
            block = blocks[block_idx]

            block.to(device)

            orig_grad_flags = {}
            for p in block.parameters():
                orig_grad_flags[p] = p.requires_grad
                p.requires_grad_(True)

            hooks = []
            for name, module in block_to_candidates.get(block_idx, []):
                hooks.append(module.register_full_backward_hook(_make_bwd_hook(name, b_accum)))

            grad = backward_input(
                saved_inps[block_idx],
                block,
                grad,
                kwargs,
                batch_size,
                device,
            )

            for p, flag in orig_grad_flags.items():
                p.requires_grad_(flag)
            for h in hooks:
                h.remove()
            block.cpu()
            torch.cuda.empty_cache()

    a_diag = {}
    b_diag = {}
    for name in a_accum:
        cnt = max(counts[name], 1)
        a_diag[name] = (a_accum[name] / cnt).float()
        if use_curvature_b:
            b_diag[name] = (b_accum[name] / cnt).float()
        else:
            b_diag[name] = None

    if original_device.type != "cpu":
        if logger:
            logger.info("Restoring model to %s", original_device)
        model.to(original_device)

    return a_diag, b_diag


def _make_fwd_hook(key, A_accum, counts):
    def hook(_mod, inp, _out):
        x = inp[0].detach().float()
        x_flat = x.reshape(-1, x.shape[-1])
        A_accum[key].add_((x_flat**2).sum(dim=0).cpu().double())
        counts[key] += x_flat.shape[0]

    return hook


def _make_bwd_hook(key, B_accum):
    def hook(_mod, _grad_in, grad_out):
        g = grad_out[0]
        if g is None:
            return
        g = g.detach().float()
        g_flat = g.reshape(-1, g.shape[-1])
        B_accum[key].add_((g_flat**2).sum(dim=0).cpu().double())

    return hook


def _compute_loss_grad(final_hidden, norm, lm_head, model_inputs, device, adapter=None):
    """Backprop ``∂loss/∂final_hidden`` for one calibration sample at a time.

    For HF causal LMs ``model_inputs["input_ids"]`` drives a cross-entropy
    loss over shifted labels.  For other architectures the adapter's
    :meth:`compute_curvature_loss` is invoked with the per-sample input
    slice (e.g. DiT MSE on velocity).
    """
    all_grads = []

    norm.to(device)
    lm_head.to(device)

    n_samples = final_hidden.shape[0]
    for i in range(n_samples):
        out_i = final_hidden[i : i + 1].to(device)
        out_i = out_i.detach().requires_grad_(True)

        if adapter is not None:
            sample_inputs = adapter.slice_calibration_inputs(model_inputs, i, i + 1)
        else:
            sample_inputs = {"input_ids": model_inputs["input_ids"][i : i + 1].to(device)}

        with torch.enable_grad():
            if adapter is not None:
                loss = adapter.compute_curvature_loss(
                    final_hidden=out_i,
                    norm=norm,
                    head=lm_head,
                    sample_inputs=sample_inputs,
                    device=device,
                )
            else:
                normed = norm(out_i)
                logits = lm_head(normed)
                ids_i = sample_inputs["input_ids"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = ids_i[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            loss.backward()

        all_grads.append(out_i.grad.cpu())

    norm.cpu()
    lm_head.cpu()
    torch.cuda.empty_cache()

    return torch.cat(all_grads)
