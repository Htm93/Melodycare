import os
import shutil
import pandas as pd
import config

def organize_dataset():
    report_path = r"therapy_compliance_report.csv"
    if not os.path.exists(report_path):
        print(f"[ERROR] Không tìm thấy file báo cáo: {report_path}")
        return

    df = pd.read_csv(report_path)
    
    # Tạo thư mục chứa riêng PASS và FAIL
    pass_dir = r"Dataset\Music\processed\training_data\pass"
    fail_dir = r"Dataset\Music\processed\training_data\failed"
    os.makedirs(pass_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    moved_count = {"PASS": 0, "FAIL": 0}

    for _, row in df.iterrows():
        filename = row["File Name"]
        status = row["Overall Status"]
        
        # Đường dẫn nguồn của file target đã xử lý DSP
        src_path = os.path.join(r"Dataset\Music\processed\training_data\unsorted", filename)
        
        if not os.path.exists(src_path):
            print(f"[WARNING] Could not found: {filename}")
            continue
            
        if status == "PASS":
            dst_path = os.path.join(pass_dir, filename)
            shutil.move(src_path, dst_path)
            moved_count["PASS"] += 1
        elif status == "FAIL":
            dst_path = os.path.join(fail_dir, filename)
            shutil.move(src_path, dst_path)
            moved_count["FAIL"] += 1

    print(f"[DONE] Finished sorting:")
    print(f"  - Move to PASS: {moved_count['PASS']} files")
    print(f"  - Move to FAIL: {moved_count['FAIL']} files")

if __name__ == "__main__":
    organize_dataset()