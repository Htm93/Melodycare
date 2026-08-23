"""
MelodyCare Input-Audio Conditioning Projection

WHY THIS EXISTS:
    The pretrained stable-audio-open-1.0 checkpoint has NO built-in mechanism
    for conditioning on a full audio clip:
        model.input_concat_ids    == []
        model.prepend_cond_ids    == []
        model.local_add_cond_ids  == []
    It's a text-to-audio generator, not an audio-to-audio translator.

    Concatenating input_latents onto the noised latents along the channel
    dim (as the original loss.py did) would feed 128 channels into a DiT
    input projection built for exactly 64 (model.io_channels == 64) —
    this crashes, or if forced to "work" via a hand-resized weight matrix,
    corrupts the pretrained projection's learned meaning.

    Cross-attention conditioning tokens don't have this problem: attention's
    K/V projection is applied per-token, so the token SEQUENCE LENGTH is
    free to vary — you can append tokens without touching any pretrained
    weight shape. This is exactly how style_projection.py already injects
    the style vector. We use the same trick here for source-audio content.

WHAT IT DOES:
    Downsamples the input audio's VAE latents (B, latent_channels, T) into a
    short sequence of tokens via a strided Conv1d ("patch embedding" style),
    then projects each patch to cond_token_dim (768, matching the DiT's
    cross-attention token dimension — same as config.MODEL_DIM, used
    identically by StyleVectorProjection).

    Downsample factor controls how many tokens the 30s clip becomes. Higher
    = fewer tokens = cheaper attention but coarser temporal resolution.
    Tune this once you know the actual VAE latent frame count for a 30s
    chunk (print `input_latents.shape` once during a real training step).
"""
import torch
import torch.nn as nn
import config


class InputAudioProjection(nn.Module):
    """
    Projects encoded input-audio latents into a sequence of cross-attention
    conditioning tokens, giving the DiT access to the source audio's
    structure without resizing any pretrained weight matrix.
    """

    def __init__(
        self,
        latent_channels   : int = 64,    # stable-audio-open-1.0 VAE latent channels (model.io_channels)
        cond_token_dim    : int = config.MODEL_DIM,  # 768 — must match the DiT's cross-attn token dim
        downsample_factor : int = 8,     # tokens = T // downsample_factor; tune once T is known
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # Strided conv: each output token summarizes `downsample_factor`
        # consecutive latent frames — analogous to a ViT-style patch embed.
        self.patch_embed = nn.Conv1d(
            in_channels  = latent_channels,
            out_channels = cond_token_dim,
            kernel_size  = downsample_factor,
            stride       = downsample_factor,
        )
        self.norm = nn.LayerNorm(cond_token_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

        # Zero-init the final LayerNorm so this module outputs exactly zero
        # at step 0 — same rationale as StyleVectorProjection's zero-init
        # (see that file's comment): don't inject a random, disruptive token
        # sequence into the pretrained cross-attention before training has
        # had a chance to learn to use it usefully.
        nn.init.zeros_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, input_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_latents: (B, latent_channels, T)
        Returns:
            (B, T // downsample_factor, cond_token_dim) conditioning tokens
        """
        tokens = self.patch_embed(input_latents)   # (B, cond_token_dim, T')
        tokens = tokens.transpose(1, 2)             # (B, T', cond_token_dim)
        return self.norm(tokens)