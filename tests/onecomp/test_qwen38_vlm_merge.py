from __future__ import annotations

from onecomp.vlm_checkpoint import (
    _link_or_copy,
    choose_group_size,
    merged_vlm_config,
)


def test_group_size_uses_largest_power_of_two_divisor() -> None:
    assert choose_group_size(1152, 128) == 128
    assert choose_group_size(4304, 128) == 16


def test_link_or_copy_resolves_relative_snapshot_symlink(tmp_path) -> None:
    blob_dir = tmp_path / "blobs"
    snapshot_dir = tmp_path / "snapshots" / "revision"
    output_dir = tmp_path / "output"
    blob_dir.mkdir()
    snapshot_dir.mkdir(parents=True)
    output_dir.mkdir()
    blob = blob_dir / "weight"
    blob.write_bytes(b"packed")
    source = snapshot_dir / "model.safetensors"
    source.symlink_to("../../blobs/weight")

    target = output_dir / "model.safetensors"
    _link_or_copy(source, target)

    assert target.is_file()
    assert not target.is_symlink()
    assert target.read_bytes() == b"packed"


def test_merged_config_restores_vlm_and_namespaces_text_layers() -> None:
    base = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 1},
        "vision_config": {"depth": 1},
    }
    text = {
        "quantization_config": {
            "bits": 4,
            "group_size": 128,
            "modules_in_block_to_quantize": ["model.layers.0.mlp.up_proj"],
            "quantized_layer_names": ["model.layers.0.mlp.up_proj"],
            "quantization_bits": [{"mlp.up_proj": 4}],
        }
    }

    merged = merged_vlm_config(
        base,
        text,
        {"model.visual.blocks.0.mlp.linear_fc1": 128},
        vision_bits=4,
    )

    assert merged["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert merged["vision_config"] == {"depth": 1}
    quantized = merged["quantization_config"]["quantized_layer_names"]
    assert "model.language_model.layers.0.mlp.up_proj" in quantized
    assert "model.visual.blocks.0.mlp.linear_fc1" in quantized
    assert merged["quantization_config"]["vision_quantization"] == {
        "bits": 4,
        "groupsizes": {"model.visual.blocks.0.mlp.linear_fc1": 128},
    }
