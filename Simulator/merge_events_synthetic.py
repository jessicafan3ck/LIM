import os
import glob
import pandas as pd

def merge_all(base_dir=".", events_subdir="Events", synthetic_subdir="Synthetic", merged_subdir="Merged"):
    events_dir = os.path.join(base_dir, events_subdir)
    synth_dir = os.path.join(base_dir, synthetic_subdir)
    merged_dir = os.path.join(base_dir, merged_subdir)
    os.makedirs(merged_dir, exist_ok=True)

    # Index synthetic files by their stem (without extension)
    synth_index = {}
    for spath in glob.glob(os.path.join(synth_dir, "*.csv")):
        sbase = os.path.splitext(os.path.basename(spath))[0]
        synth_index[sbase] = spath

    # Loop through all event files
    for epath in glob.glob(os.path.join(events_dir, "*.xlsx")) + glob.glob(os.path.join(events_dir, "*.xls")):
        try:
            ev = pd.read_excel(epath, sheet_name=0)
        except Exception as e:
            print(f"[WARN] Could not read {epath}: {e}")
            continue

        base = os.path.splitext(os.path.basename(epath))[0]
        # Find matching synthetic file (assumes same base name + "_synthetic.csv")
        sbase = f"{base}_synthetic"
        spath = synth_index.get(sbase)
        if not spath:
            print(f"[WARN] No synthetic file for {base}")
            continue

        try:
            syn = pd.read_csv(spath)
        except Exception as e:
            print(f"[WARN] Could not read {spath}: {e}")
            continue

        # Merge on match_id + event_id
        merged = ev.merge(syn, on=["match_id","event_id"], how="left")

        out_csv = os.path.join(merged_dir, f"{base}_merged.csv")
        merged.to_csv(out_csv, index=False)
        print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    merge_all(base_dir="/Users/jessicafan/LIM")
