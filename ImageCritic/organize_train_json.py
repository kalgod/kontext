import os
import json


DATASET_ROOT = "./"
OPEN_DATASET_DIR = os.path.join(DATASET_ROOT, "Critic-10K")

OUTPUT_JSONL = os.path.join(DATASET_ROOT, "train_metadata.jsonl")

def main():
    if not os.path.isdir(OPEN_DATASET_DIR):
        raise FileNotFoundError(f"找不到目录: {OPEN_DATASET_DIR}")

    sample_dirs = sorted(
        d for d in os.listdir(OPEN_DATASET_DIR)
        if os.path.isdir(os.path.join(OPEN_DATASET_DIR, d))
    )

    print(f"发现样本数量: {len(sample_dirs)}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for idx, dirname in enumerate(sample_dirs):
            sample_dir = os.path.join(OPEN_DATASET_DIR, dirname)

            rel_base = os.path.join("open_dataset", dirname)

            product_path = os.path.join(rel_base, "product.png")
            degraded_path = os.path.join(rel_base, "generated_with_degradation.png")
            target_path = os.path.join(rel_base, "generated.png")
            mask_path = os.path.join(rel_base, "mask.png")
            shape_path = os.path.join(rel_base, "image_shape.txt")
            prompt_path = os.path.join(sample_dir, "prompt.txt")

            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as pf:
                    caption = pf.read().strip()
            else:
                caption = ""
                print(f"[WARN] {prompt_path} 不存在，caption 置为空。")

            record = {
                "A_image": product_path,
                "B_image": degraded_path,
                "target": target_path,
                "mask": mask_path,
                "image_shape": shape_path,
                "caption": caption,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0 or idx == len(sample_dirs) - 1:
                print(f"已处理 {idx + 1}/{len(sample_dirs)} 个样本")

    print(f"完成！元数据已写入: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
