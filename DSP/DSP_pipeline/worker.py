# worker.py

from modules.pipeline import process_therapeutic_track

# Global trong worker process — được set bởi initializer
_worker_pause_event = None

def _worker_init(pause_event):
    """
    Called once when each worker process starts.
    Sets the global pause_event for this worker process.
    
    ProcessPoolExecutor's initializer runs BEFORE any task is submitted,
    so pause_event is guaranteed to be set when _process_one runs.
    """
    global _worker_pause_event
    _worker_pause_event = pause_event


def _process_one(args):
    """
    Top-level wrapper for processing a single song folder.
    pause_event is set via _worker_init, not passed through args.
    
    Args:
        args: tuple of (song_folder, original_folder, output_dir, sample_rate)
              — no longer needs pause_event in args
    """
    song_folder, original_folder, output_dir, sample_rate = args

    # _worker_pause_event guaranteed to exist — set by _worker_init
    if _worker_pause_event is not None:
        _worker_pause_event.wait()

    try:
        output_path = process_therapeutic_track(
            song_folder, original_folder, output_dir, sample_rate
        )
        return song_folder, output_path, None
    except Exception as e:
        return song_folder, None, str(e)