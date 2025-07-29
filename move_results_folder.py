import os
import shutil

def move_folders_by_partial_name(source_dir, target_dir, partial_name):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    total_checked = 0
    total_moved = 0
    total_skipped = 0

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        
        if os.path.isdir(folder_path):
            total_checked += 1

            if partial_name in folder_name:
                dest_path = os.path.join(target_dir, folder_name)
                
                if not os.path.exists(dest_path):
                    shutil.move(folder_path, dest_path)
                    print(f"Moved: {folder_name} -> {target_dir}")
                    total_moved += 1
                else:
                    print(f"Skipped (already exists in target): {folder_name}")
                    total_skipped += 1

    # Summary
    print("\n--- Summary ---")
    print(f"Total folders checked: {total_checked}")
    print(f"Total folders matched and moved: {total_moved}")
    print(f"Total skipped (already existed): {total_skipped}")

source = "./results/"
target = "./results_feature_ablation/"
partial = "powerset"  # <- only move folders that contain this string

move_folders_by_partial_name(source, target, partial)
