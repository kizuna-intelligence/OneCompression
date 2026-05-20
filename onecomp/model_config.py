"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from logging import getLogger

from .utils.dtype import needs_bfloat16


class ModelConfig:
    """Model and Tokenizer configuration.

    Holds either a HuggingFace model id / path (and an auto-constructed
    :class:`HFLLMAdapter`) or an explicit :class:`ModelAdapter` for
    non-HF architectures (e.g. :class:`IrodoriDiTAdapter` for diffusion
    transformers).
    """

    def __init__(
        self,
        model_id: str = None,
        path: str = None,
        dtype: str = "float16",
        device: str = "auto",
        adapter=None,
    ):
        """__init__ method

        Args:
            model_id (str): Model ID (Hugging Face Hub ID).
            path (str): Path to the saved model and tokenizer.
            dtype (str, optional): Data type. Defaults to "float16".
            device (str, optional): Device to use ("cpu", "cuda", "auto"). Defaults to "auto".
            adapter (ModelAdapter, optional): Custom adapter to use for
                non-HuggingFace models.  When set, takes precedence over
                ``model_id`` / ``path``; ``load_model`` and friends
                delegate to it.

        Example:
            >>> model_config = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
            >>> model = model_config.load_model()
            >>> tokenizer = model_config.load_tokenizer()

            >>> from onecomp.adapters import IrodoriDiTAdapter
            >>> adapter = IrodoriDiTAdapter(checkpoint_path="...")
            >>> model_config = ModelConfig(adapter=adapter)
            >>> model = model_config.load_model()

        """
        self.logger = getLogger(__name__)

        if adapter is None and model_id is None and path is None:
            raise ValueError("Either model_id, path, or adapter must be provided")

        if adapter is None:
            if needs_bfloat16(model_id or path):
                if dtype != "bfloat16":
                    self.logger.warning(
                        "Overriding dtype to bfloat16 for %s "
                        "to prevent performance degradation.",
                        model_id or path,
                    )
                dtype = "bfloat16"

            from .adapters.hf_llm import HFLLMAdapter

            adapter = HFLLMAdapter(
                model_id=model_id, path=path, dtype=dtype, device=device
            )

        self.model_id = model_id
        self.path = path
        self.dtype = dtype
        self.device = device
        self.adapter = adapter

    def get_model_id_or_path(self):
        """Get the model ID or path, or ``None`` for adapter-only configs."""
        if self.model_id is not None:
            return self.model_id
        if self.path is not None:
            return self.path
        # Adapters with their own loader expose this hook.
        if hasattr(self.adapter, "get_model_id_or_path"):
            return self.adapter.get_model_id_or_path()
        return None

    def load_config(self):
        """Load and cache the underlying HuggingFace config (if available)."""
        if hasattr(self.adapter, "load_config"):
            return self.adapter.load_config()
        return None

    def load_model(self, device_map=None):
        """Load the model via the configured adapter."""
        return self.adapter.load_model(device_map=device_map)

    def load_tokenizer(self):
        """Load the tokenizer via the configured adapter (may return None)."""
        return self.adapter.load_tokenizer()

    def has_additional_data(self):
        """Whether the adapter installs extra hooks/state on the model."""
        return self.adapter.has_additional_data()
