"""Tests for the Gemma 4 E2B fixed-int4 example contracts."""

import pytest
import torch
from torch import nn

from example.gemma4_e2b_int4.contracts import audit_fixed_int4_config
from example.gemma4_e2b_int4.export_mlx import (
    convert_gptq_v1_to_mlx,
    split_mlx_quantized_embedding,
)
from example.gemma4_e2b_int4.pack_embeddings import pack_embedding_weight
from onecomp.quantizer.gptq.gptq_layer import unpack_int_weights, unpack_zeros
from onecomp.quantized_model_loader import QuantizedModelLoader
from onecomp.utils.blockwise import _PER_LAYER_INPUTS_KEY, get_blocks_and_inputs


def _config(bits):
    return {
        "quantization_config": {
            "bits": 4,
            "group_size": 128,
            "quantization_bits": [
                {"self_attn.q_proj": {"bits": bit}} for bit in bits
            ],
        }
    }


def test_fixed_int4_audit_accepts_uniform_four_bit_layers():
    assert audit_fixed_int4_config(_config([4, 4])) == {"4": 2}


def test_fixed_int4_audit_rejects_mixed_bit_layers():
    with pytest.raises(ValueError, match="mixed-bit"):
        audit_fixed_int4_config(_config([4, 8]))


def test_fixed_int4_audit_requires_declared_layers():
    with pytest.raises(ValueError, match="no quantized modules"):
        audit_fixed_int4_config(_config([]))


class _PerLayerBlock(nn.Module):
    hidden_size_per_layer_input = 2
    layer_idx = 0

    def forward(self, hidden_states, per_layer_input=None, **_kwargs):
        if per_layer_input is None:
            raise RuntimeError("missing per-layer input")
        return hidden_states + per_layer_input


class _PerLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 2)
        self.layers = nn.ModuleList([_PerLayerBlock()])

    def get_per_layer_inputs(self, input_ids, _inputs_embeds):
        return input_ids.to(torch.float32).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)

    def project_per_layer_inputs(self, _inputs_embeds, per_layer_inputs):
        return per_layer_inputs

    def forward(self, input_ids, **_kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
        return self.layers[0](inputs_embeds, per_layer_inputs[:, :, 0, :])


class _PerLayerAdapter:
    def get_blocks(self, model):
        return model.layers

    def num_calibration_samples(self, inputs):
        return inputs["input_ids"].shape[0]

    def slice_calibration_inputs(self, inputs, start, end):
        return {"input_ids": inputs["input_ids"][start:end]}

    def run_calibration_forward(self, model, inputs):
        return model(**inputs)


def test_adapter_path_preserves_gemma4_per_layer_inputs():
    model = _PerLayerModel()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    _blocks, block_inputs, kwargs = get_blocks_and_inputs(
        model,
        {"input_ids": input_ids},
        batch_size=1,
        adapter=_PerLayerAdapter(),
    )

    assert block_inputs.shape == (2, 3, 2)
    assert torch.equal(
        kwargs[_PER_LAYER_INPUTS_KEY],
        model.get_per_layer_inputs(input_ids, model.embed_tokens(input_ids)),
    )


def test_embedding_packer_emits_gptq_v1_layout():
    torch.manual_seed(0)
    weight = torch.randn(16, 16, dtype=torch.float32)
    qweight, scales, qzeros = pack_embedding_weight(
        weight, groupsize=8, chunk_rows=8
    )

    values = unpack_int_weights(qweight, 4, (16, 16)).to(torch.float32)
    zeros = ((unpack_zeros(qzeros, 4, 16) + 1) & 0xF).to(torch.float32)
    reconstructed = (
        (values.reshape(16, 2, 8) - zeros.T.unsqueeze(-1))
        * scales.T.unsqueeze(-1)
    ).reshape_as(weight)

    assert qweight.shape == (2, 16)
    assert scales.shape == (2, 16)
    assert qzeros.shape == (2, 2)
    assert torch.mean((weight - reconstructed) ** 2).item() < 0.01


def test_mlx_export_preserves_gptq_v1_dequantized_values():
    torch.manual_seed(1)
    original = torch.randn(16, 16, dtype=torch.float32)
    qweight, scales, qzeros = pack_embedding_weight(
        original, groupsize=8, chunk_rows=8
    )

    mlx_weight, mlx_scales, mlx_biases = convert_gptq_v1_to_mlx(
        qweight, scales, qzeros
    )

    shifts = torch.arange(8, dtype=torch.int64) * 4
    mlx_values = (
        (mlx_weight.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
    ).reshape(16, 16)
    mlx_reconstructed = (
        mlx_values.reshape(16, 2, 8).to(torch.float32)
        * mlx_scales.to(torch.float32).unsqueeze(-1)
        + mlx_biases.to(torch.float32).unsqueeze(-1)
    ).reshape_as(original)

    gptq_values = unpack_int_weights(qweight, 4, (16, 16)).to(torch.float32)
    gptq_zeros = ((unpack_zeros(qzeros, 4, 16) + 1) & 0xF).to(torch.float32)
    gptq_reconstructed = (
        (gptq_values.reshape(16, 2, 8) - gptq_zeros.T.unsqueeze(-1))
        * scales.T.unsqueeze(-1)
    ).reshape_as(original)

    assert mlx_weight.dtype == torch.uint32
    # MLX stores the derived affine bias in the scale dtype (fp16 here), so the
    # only allowed difference is that final fp16 bias rounding.
    torch.testing.assert_close(
        mlx_reconstructed, gptq_reconstructed, rtol=0, atol=1e-3
    )


def test_mlx_embedding_shards_preserve_packed_codes_and_affine_parameters():
    rows = 8
    group_size = 8
    bits = 4
    groups = 7
    packed_columns = groups * group_size * bits // 32
    weight = torch.arange(rows * packed_columns, dtype=torch.int64).reshape(
        rows, packed_columns
    ).to(torch.uint32)
    scales = torch.arange(rows * groups, dtype=torch.float16).reshape(rows, groups)
    biases = -scales

    shards = split_mlx_quantized_embedding(
        weight,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        max_packed_tensor_bytes=rows * 2 * weight.element_size(),
    )

    assert len(shards) > 1
    assert all(
        shard_weight.nbytes <= rows * 2 * weight.element_size()
        for shard_weight, _, _ in shards
    )
    assert torch.equal(torch.cat([part[0] for part in shards], dim=1), weight)
    assert torch.equal(torch.cat([part[1] for part in shards], dim=1), scales)
    assert torch.equal(torch.cat([part[2] for part in shards], dim=1), biases)


def test_quantized_loader_reties_weights_after_assign(monkeypatch, tmp_path):
    class _TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(2, 2)
            self.lm_head = nn.Linear(2, 2, bias=False)
            self.tie_calls = 0

        def load_state_dict(self, _state, strict=False, assign=False):
            assert strict is False and assign is True
            self.embed.weight = nn.Parameter(torch.ones(2, 2))
            return [], []

        def tie_weights(self):
            self.tie_calls += 1
            self.lm_head.weight = self.embed.weight

    model = _TiedModel()
    monkeypatch.setattr(
        QuantizedModelLoader,
        "_load_config_and_quant_config",
        lambda _path: ({"model_type": "fake"}, {"quant_method": "gptq"}),
    )
    monkeypatch.setattr(
        QuantizedModelLoader,
        "_build_empty_model_from_config",
        lambda _config, _dtype: model,
    )
    monkeypatch.setattr(
        QuantizedModelLoader, "_load_state_dict_from_dir", lambda _path: {}
    )
    monkeypatch.setattr(
        QuantizedModelLoader, "_reconcile_state_dict_keys", lambda _model, state: state
    )
    monkeypatch.setattr(
        QuantizedModelLoader, "_replace_quantized_layers", lambda *_args: None
    )
    monkeypatch.setattr(
        "onecomp.quantized_model_loader.needs_bfloat16", lambda _path: False
    )
    monkeypatch.setattr(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: object(),
    )

    loaded, _tokenizer = QuantizedModelLoader.load_quantized_model(
        str(tmp_path), device_map=""
    )

    assert loaded.tie_calls == 1
    assert loaded.lm_head.weight is loaded.embed.weight
