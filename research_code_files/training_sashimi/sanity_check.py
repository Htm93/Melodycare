import os
import csv
import torch

def sanity_check(csv_path="/path/to/dataset/melodycare_training_manifest.csv"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Không tìm thấy file manifest: {csv_path}")
        return False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    print(f"[INFO] Đang kiểm tra {len(reader)} mẫu trong manifest...")
    
    missing_input = 0
    missing_target = 0
    missing_style = 0

    for i, row in enumerate(reader):
        if not os.path.exists(row["input_audio"]):
            missing_input += 1
            if missing_input <= 3:
                print(f"  [MISSING INPUT] {row['input_audio']}")
        
        if not os.path.exists(row["target_audio"]):
            missing_target += 1
            if missing_target <= 3:
                print(f"  [MISSING TARGET] {row['target_audio']}")
                
        if not os.path.exists(row["style_vector_pt"]):
            missing_style += 1
            if missing_style <= 3:
                print(f"  [MISSING STYLE] {row['style_vector_pt']}")

    # Kiểm tra style vector load được không
    style_pt = reader[0]["style_vector_pt"]
    if os.path.exists(style_pt):
        try:
            vec = torch.load(style_pt, map_location="cpu")
            print(f"[OK] Style vector load successfully | Shape: {vec.shape} | Norm: {vec.norm():.4f}")
        except Exception as e:
            print(f"[ERROR] Cannot load style vector: {e}")
            return False

    print("=" * 50)
    print(f"Check result:")
    print(f"  - Missing inputs:  {missing_input}")
    print(f"  - Missing targets: {missing_target}")
    print(f"  - Missing styles:  {missing_style}")
    print("=" * 50)

    if missing_input == 0 and missing_target == 0 and missing_style == 0:
        print("[SUCCESS] Data correct, ready to train!")
        return True
    else:
        print("[WARNING] missing file dir, check config.py and regenerate CSV.")
        return False

if __name__ == "__main__":
    sanity_check()