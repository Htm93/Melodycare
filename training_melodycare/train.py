"""
MelodyCare — Training Entry Point (v2)

v2: Removed --mix_weight_melody/bass/drum CLI args (pre-mixed files now).

Single GPU:
    python train.py

Multi-GPU (training server, 2x RTX 4090):
    accelerate launch --config_file configs/accelerate_config.yaml train.py
"""
import argparse
import config
from configs.train_config import TrainConfig
from training.loop import train


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="MelodyCare Stable Audio 3 Fine-tuning (v2)"
    )

    # All defaults from config.py
    parser.add_argument("--csv_path",        type=str,   default=config.CSV_OUTPUT_PATH)
    parser.add_argument("--output_dir",      type=str,   default=config.OUTPUT_DIR)
    parser.add_argument("--logging_dir",     type=str,   default=config.LOGGING_DIR)
    parser.add_argument("--num_epochs",      type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size",      type=int,   default=config.BATCH_SIZE_PER_GPU)
    parser.add_argument("--grad_accum",      type=int,   default=config.GRADIENT_ACCUM_STEPS)
    parser.add_argument("--learning_rate",   type=float, default=config.LEARNING_RATE)
    parser.add_argument("--warmup_steps",    type=int,   default=config.LR_WARMUP_STEPS)
    parser.add_argument("--save_every",      type=int,   default=config.SAVE_EVERY_N_STEPS)
    parser.add_argument("--log_every",       type=int,   default=config.LOG_EVERY_N_STEPS)
    parser.add_argument("--max_checkpoints", type=int,   default=config.MAX_CHECKPOINTS)
    parser.add_argument("--seed",            type=int,   default=config.SEED)
    parser.add_argument("--resume",          type=str,   default=None,
                        help="Path to checkpoint dir to resume from")
    parser.add_argument("--no_grad_ckpt",    action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM)")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="When resuming (--resume), restore model/optimizer weights "
                             "but build a FRESH LR scheduler sized for the new num_epochs, "
                             "instead of restoring the old (possibly already fully-annealed) "
                             "scheduler state. Use this when extending a completed/near-"
                             "complete run with a larger --num_epochs.")

    args = parser.parse_args()

    return TrainConfig(
        csv_path               = args.csv_path,
        output_dir             = args.output_dir,
        logging_dir            = args.logging_dir,
        num_epochs             = args.num_epochs,
        batch_size_per_gpu     = args.batch_size,
        gradient_accum_steps   = args.grad_accum,
        learning_rate          = args.learning_rate,
        lr_warmup_steps        = args.warmup_steps,
        save_every_n_steps     = args.save_every,
        log_every_n_steps      = args.log_every,
        max_checkpoints        = args.max_checkpoints,
        seed                   = args.seed,
        resume_from            = args.resume,
        gradient_checkpointing = not args.no_grad_ckpt,
        reset_scheduler         = args.reset_scheduler,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)