import os
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from worker import _worker_init
import pandas as pd

from config import SAMPLE_RATE
from worker import _process_one   # import từ top-level module

def batch_process(input_dir, original_folder, output_dir, csv_dir,
                  sample_rate=SAMPLE_RATE, max_workers=4):

    if not os.path.exists(csv_dir):
        print(f"[ERROR] Report path not found: {csv_dir}")
        return
    failed_dir = r"Dataset\Music\processed\training_data\failed"
    failed_files = [f for f in os.listdir(failed_dir) if f.endswith(".wav")]

    # df = pd.read_csv(csv_dir)

    # Sort out failed files
    # failed_df = df[df["Overall Status"] == "FAIL"]
    # failed_files = failed_df["File Name"].tolist()

    # Strip file name to match name
    failed_files = [filename.removesuffix("_therapeutic.wav") for filename in failed_files]

    song_folders = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, f))
        and f in failed_files
        # if f == "quang_dung_bai_thanh_ca_buon_liveshow_da_khuc_cho_tinh_nhan_5"
        and os.path.exists(os.path.join(input_dir, f, "melody.wav"))
        and os.path.exists(os.path.join(input_dir, f, "drums.wav"))
        and os.path.exists(os.path.join(input_dir, f, "bass.wav"))
    ]

    if not song_folders:
        print(f"No valid song folders found in {input_dir}")
        return

    print(f"Found {len(song_folders)} songs to process")
    print("Commands: Enter = pause/resume | Ctrl+C = stop\n")

    manager     = multiprocessing.Manager()
    pause_event = manager.Event()
    pause_event.set()

    args_list = [
        (folder, original_folder, output_dir, sample_rate)
        for folder in song_folders
    ]
    batch_size = 5
    batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]

    def listen_for_pause():
        while True:
            input()
            if pause_event.is_set():
                pause_event.clear()
                tqdm.write("\n⏸  Paused — press Enter to resume...")
            else:
                pause_event.set()
                tqdm.write("\n▶  Resumed")

    threading.Thread(target=listen_for_pause, daemon=True).start()

    failed = []

    for batch_idx, batch_args in enumerate(batches):
        print(f"\n[BATCH {batch_idx + 1}/{len(batches)}] Processing {len(batch_args)} tracks...")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer = _worker_init,
            initargs = (pause_event,)
        ) as executor:
            futures = {executor.submit(_process_one, a): a[0] for a in batch_args}
            
            with tqdm(total=len(futures), desc=f"Processing Batch {batch_idx + 1}") as pbar:
                # FIX: Yields futures immediately as they finish, out of order
                for future in as_completed(futures):
                    folder, path, error = future.result()
                    if error:
                        failed.append((folder, error))
                    pbar.update(1)

    manager.shutdown()

    success = len(song_folders) - len(failed)
    print(f"\n{'='*50}")
    print(f"Batch complete: {success}/{len(song_folders)} successful")
    if failed:
        print("\nFailed:")
        for f, e in failed:
            print(f"  {os.path.basename(f)}: {e}")


if __name__ == "__main__":
    safe_workers = max(1, os.cpu_count() - 2)
    batch_process(
        input_dir       = r"Dataset\Music\processed\separated_sources",
        original_folder = r"Dataset\Raw\Tier_1+2(60-100BPM)(131)",
        output_dir      = r"Dataset\Music\processed\training_data\unsorted",
        csv_dir         = r"therapy_compliance_report.csv",
        max_workers     = safe_workers
    )