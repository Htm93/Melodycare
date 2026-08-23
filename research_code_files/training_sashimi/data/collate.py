"""
MelodyCare DataLoader Collate Function
"""
import torch
from torch.utils.data import DataLoader
import config


def melodycare_collate_fn(batch: list[dict]) -> dict:
    return {
        "input_audio" : torch.stack([b["input_audio"]  for b in batch]),
        "target_audio": torch.stack([b["target_audio"] for b in batch]),
        "style_vector": torch.stack([b["style_vector"] for b in batch]),
        "text_prompt" : [b["text_prompt"] for b in batch],
        "song_name"   : [b["song_name"]   for b in batch],
    }


def build_dataloader(
    dataset,
    batch_size  : int  = config.BATCH_SIZE_PER_GPU,
    shuffle     : bool = True,
    num_workers : int  = 4,
    pin_memory  : bool = True,
) -> DataLoader:
    """Build DataLoader with MelodyCare collate function."""
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        collate_fn         = melodycare_collate_fn,
        prefetch_factor    = 2 if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
        drop_last          = True,
    )