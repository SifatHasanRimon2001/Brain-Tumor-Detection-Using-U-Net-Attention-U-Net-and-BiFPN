import os
from src.utils.visualize import visualize_models
def process_index(index, img_files, mask_files, img_dir, mask_dir, device, img_size):
    img_name = img_files[index]
    mask_name = mask_files[index]
    img_path = os.path.join(img_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)
    save_path = os.path.join("results", f"compare_idx{index}.png")
    print(f"[INFO] Visualizing index {index}: {img_name}")
    visualize_models(
        image_path=img_path,
        mask_path=mask_path,
        models_root="runs",
        device=device,
        img_size=img_size,
        save_path=save_path,
    )
def main():
    data_root = "brisc2025"
    device = "cuda"
    img_size = 256
    img_dir = os.path.join(data_root, "segmentation_task", "test", "images")
    mask_dir = os.path.join(data_root, "segmentation_task", "test", "masks")
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))
    max_index = min(859, len(img_files) - 1)
    os.makedirs("results", exist_ok=True)
    index = int(input(f"Enter an index (0 to {max_index}): "))
    if index < 0 or index > max_index:
        print(f"[ERROR] Index out of range! Please enter between 0 and {max_index}.")
        return
    process_index(index, img_files, mask_files, img_dir, mask_dir, device, img_size)
if __name__ == "__main__":
    main()