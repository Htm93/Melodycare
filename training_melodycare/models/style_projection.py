"""
MelodyCare Style Vector Projection Layer
"""
import torch
import torch.nn as nn
import config


class StyleVectorProjection(nn.Module):
    """
    Projects CLAP style vector (STYLE_VECTOR_DIM) into U-Net cross-attention space.
    Produces (B, 1, MODEL_DIM) style token appended to text token sequence.
    """

    def __init__(
        self,
        style_dim : int = config.STYLE_VECTOR_DIM,
        model_dim : int = config.MODEL_DIM,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.model_dim = model_dim

        self.proj = nn.Sequential(
            nn.Linear(style_dim, model_dim, bias=True),
            nn.LayerNorm(model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=True),
            nn.LayerNorm(model_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-init the FINAL LayerNorm's gamma/beta so this module outputs
        # exactly zero at step 0, regardless of the (randomly-initialized)
        # layers before it — LayerNorm output = normalize(x) * weight + bias,
        # so weight=0, bias=0 forces output=0 no matter what x is.
        #
        # WHY: without this, the style token starts as a random, disruptive
        # vector injected directly into the pretrained model's cross-attention
        # from the very first training step — for a large pretrained network
        # fine-tuned on a small dataset, this risks destabilizing/forgetting
        # what the model already knows before it's learned to use the new
        # conditioning usefully (the ControlNet "zero convolution" trick).
        # Gradients still flow normally, so this only affects initialization —
        # the model gradually learns to use non-zero style tokens over training.
        final_layer_norm = self.proj[-1]
        nn.init.zeros_(final_layer_norm.weight)
        nn.init.zeros_(final_layer_norm.bias)

    def forward(self, style_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_vector: (B, STYLE_VECTOR_DIM)
        Returns:
            (B, 1, MODEL_DIM)
        """
        return self.proj(style_vector).unsqueeze(1)