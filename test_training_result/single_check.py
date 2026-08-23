"""
Quick single-file compliance check, using the same logic as Check_result.py's
batch_check() but for one file at a time (no _therapeutic.wav naming
requirement, no multiprocessing overhead).

Usage:
    python /path/to/training_output/result_data_check/single_check.py /path/to/training_output/check_inference/str35_polished.wav
"""

import sys
from Check_result import check_file, print_report, export_results_to_csv

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python single_check.py /path/to/output.wav")
        sys.exit(1)

    filepath = sys.argv[1]
    result = check_file(filepath, sample_rate=48000)
    print_report(result)

    # Optional: also drop a one-row CSV, same format as batch runs
    export_results_to_csv([result], output_filename="single_check_report.csv")