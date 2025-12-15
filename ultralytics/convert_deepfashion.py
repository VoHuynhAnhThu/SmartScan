from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image

SPLIT_MAP = {"train", "val", "test", "query", "gallery"}


def find_bbox_file(anno_dir: Path) -> Path | None:
    for name in ["list_bbox_consumer2shop.txt", "list_bbox_inshop.txt", "list_bbox.txt"]:
        p = anno_dir / name
        if p.exists():
            return p
    found = sorted(anno_dir.glob("list_bbox*.txt"))
    return found[0] if found else None


def find_partition_file(eval_dir: Path) -> Path | None:
    p = eval_dir / "list_eval_partition.txt"
    if p.exists():
        return p
    found = sorted([x for x in eval_dir.glob("*.txt") if "partition" in x.name.lower()])
    return found[0] if found else None


def parse_bbox_file(bbox_file: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    mapping: dict[str, list[tuple[int, int, int, int]]] = {}
    with open(bbox_file, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[2:]

        print(f"\n[DEBUG] parse_bbox_file: Tổng {len(lines)} dòng")
        print("[DEBUG] 3 dòng đầu tiên:")
        for i in range(min(3, len(lines))):
            print(f"  Line {i}: {lines[i].strip()}")

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if idx == 0:
                print("\n[DEBUG] Parse dòng đầu tiên chi tiết:")
                print(f"  parts = {parts}")
                print(f"  len(parts) = {len(parts)}")
                if len(parts) >= 4:
                    print(f"  parts[-4:] (bbox coords) = {parts[-4:]}")
                print(f"  parts[0] (image path) = {parts[0]}")

            if len(parts) < 5:
                if idx < 3:
                    print(f"[WARN] Dòng {idx}: len(parts)={len(parts)} < 5, bỏ qua")
                continue

            try:
                x1, y1, x2, y2 = map(int, parts[-4:])
            except Exception as e:
                if idx < 3:
                    print(f"[ERROR] Dòng {idx}: Không parse được bbox: {e}")
                continue

            img_rel = parts[0]

            if idx < 3:
                print(f"  → Key: '{img_rel}', bbox: ({x1}, {y1}, {x2}, {y2})")

            mapping.setdefault(img_rel, []).append((x1, y1, x2, y2))

    return mapping


def parse_partition_file(part_file: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(part_file, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[2:]

        print(f"\n[DEBUG] parse_partition_file: Tổng {len(lines)} dòng")
        print("[DEBUG] 3 dòng đầu tiên:")
        for i in range(min(3, len(lines))):
            print(f"  Line {i}: {lines[i].strip()}")

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            if idx == 0:
                print("\n[DEBUG] Parse dòng đầu tiên chi tiết:")
                print(f"  parts = {parts}")
                print(f"  len(parts) = {len(parts)}")
                print(f"  parts[0] (image path) = {parts[0]}")
                print(f"  parts[-1] = {parts[-1]}")

            if len(parts) < 2:
                continue

            img_rel = parts[0]

            split = None
            for part in reversed(parts):
                if part.lower() in SPLIT_MAP:
                    split = part.lower()
                    break

            if idx < 3:
                print(f"  → Key: '{img_rel}', split: '{split}'")

            if split:
                mapping[img_rel] = split

    return mapping


def ensure_link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _candidate_image_roots(root: Path) -> list[Path]:
    names = ["", "img_highres", "img", "images", "image", "Img", "Images"]
    cands = [root / n for n in names]

    parent = root.parent
    if parent:
        for n in ["img_highres", "img", "images", "image", "Img", "Images"]:
            cands.append(parent / n)

    unique = []
    seen = set()
    for c in cands:
        cs = str(c.resolve() if c.exists() else c)
        if cs not in seen:
            seen.add(cs)
            unique.append(c)
    return unique


def _normalize_rel_path(rel_path: str) -> str:
    rp = rel_path.replace("\\", "/").lstrip("./").lstrip("/")
    return rp


def _strip_known_prefixes(rp: str) -> list[str]:
    variants = [rp]
    parts = rp.split("/")
    if parts and parts[0].lower() in {"img_highres", "img", "images", "image"}:
        stripped = "/".join(parts[1:]) if len(parts) > 1 else ""
        if stripped:
            variants.append(stripped)
    return list(dict.fromkeys(variants))


def _resolve_img_path(base_dir: Path, rel_path: str) -> Path | None:
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p

    rp_norm = _normalize_rel_path(rel_path)
    candidates = _strip_known_prefixes(rp_norm)

    for cand in candidates:
        p1 = base_dir / cand
        if p1.exists():
            return p1
    return None


def _score_candidates(root: Path, rel_paths: list[str]) -> list[tuple[Path, int]]:
    cands = _candidate_image_roots(root)
    sample = rel_paths[: min(200, len(rel_paths))]
    scores: list[tuple[Path, int]] = []
    for base in cands:
        hits = 0
        for rp in sample:
            if _resolve_img_path(base, rp):
                hits += 1
        scores.append((base, hits))
    scores.sort(key=lambda x: x[1], reverse=True)
    print("[i] Candidate base_dir hits:")
    for base, hits in scores:
        print(f"    - {base} -> {hits}")
    return scores


def _pick_base_dir(dataset_root: Path, rel_paths: list[str]) -> Path | None:
    scores = _score_candidates(dataset_root, rel_paths)
    return scores[0][0] if scores and scores[0][1] > 0 else None


def convert_deepfashion_cts_to_yolo(
    dataset_root: Path,
    save_dir: Path,
    class_name: str = "clothes",
    images_root: Path | None = None,
):
    dataset_root = dataset_root.resolve()
    save_dir = save_dir.resolve()

    anno_dir = dataset_root / "Anno"
    eval_dir = dataset_root / "Eval"
    if not anno_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy Anno: {anno_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy Eval: {eval_dir}")

    bbox_file = find_bbox_file(anno_dir)
    part_file = find_partition_file(eval_dir)
    if not bbox_file:
        raise FileNotFoundError(f"Không tìm thấy file bbox trong {anno_dir}")
    if not part_file:
        raise FileNotFoundError(f"Không tìm thấy file partition trong {eval_dir}")

    print(f"[i] BBox file: {bbox_file}")
    print(f"[i] Partition file: {part_file}")

    bbox_map = parse_bbox_file(bbox_file)
    split_map = parse_partition_file(part_file)

    print("\n[DEBUG] ===== SUMMARY =====")
    print(f"[DEBUG] bbox_map có {len(bbox_map)} entries")
    print("[DEBUG] 5 key đầu tiên của bbox_map:")
    for i, key in enumerate(list(bbox_map.keys())[:5]):
        print(f"  {i}: '{key}' -> {len(bbox_map[key])} bbox(es)")

    print(f"\n[DEBUG] split_map có {len(split_map)} entries")
    print("[DEBUG] 5 key đầu tiên của split_map:")
    for i, key in enumerate(list(split_map.keys())[:5]):
        print(f"  {i}: '{key}' -> '{split_map[key]}'")

    # Kiểm tra matching
    print("\n[DEBUG] ===== KEY MATCHING CHECK =====")
    sample_keys = list(bbox_map.keys())[:10]
    matches = 0
    for key in sample_keys:
        if key in split_map:
            matches += 1
            print(f"  ✓ '{key}' → có trong cả 2")
        else:
            print(f"  ✗ '{key}' → KHÔNG có trong split_map")
    print(f"  → Tỉ lệ match: {matches}/{len(sample_keys)}")

    # Chọn thư mục ảnh
    if images_root:
        base_dir = Path(images_root).resolve()
        if not base_dir.exists():
            raise FileNotFoundError(f"--images-root không tồn tại: {base_dir}")
    else:
        base_dir = _pick_base_dir(dataset_root, list(bbox_map.keys()))
        if base_dir is None:
            raise FileNotFoundError(
                "Không tìm thấy thư mục ảnh tương ứng. Kiểm tra lại:\n"
                f"- dataset_root: {dataset_root}\n"
                "- Đảm bảo file ảnh theo mẫu: dataset_root/<rel_path_trong_TXT>\n"
                "- Hoặc chỉ định trực tiếp bằng --images-root trỏ đến 'img' hay 'img_highres'."
            )
    print(f"\n[i] Chọn thư mục ảnh gốc: {base_dir}")

    kept, skipped_missing_img, skipped_no_bbox, skipped_no_split = 0, 0, 0, 0
    splits_count: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    missing_samples: list[str] = []

    print("\n[DEBUG] ===== PROCESSING IMAGES =====")
    processed = 0

    for rel_path, boxes in bbox_map.items():
        processed += 1

        # LOG: 3 entry đầu tiên
        if processed <= 3:
            print(f"\n[DEBUG] Entry #{processed}:")
            print(f"  rel_path: '{rel_path}'")
            print(f"  boxes: {boxes}")

        split = split_map.get(rel_path)

        if processed <= 3:
            print(f"  split: {split}")

        if split is None:
            skipped_no_split += 1
            if processed <= 3:
                print("  → Bỏ qua (không có split)")
            continue

        if split in {"query", "gallery"}:
            split = "test"
            if processed <= 3:
                print(f"  → Đổi split thành: {split}")

        src_img = _resolve_img_path(base_dir, rel_path)

        if processed <= 3:
            print(f"  Tìm ảnh: {src_img}")

        if not src_img:
            skipped_missing_img += 1
            if len(missing_samples) < 5:
                missing_samples.append(f"{base_dir} + {rel_path}")
            if processed <= 3:
                print("  → Bỏ qua (không tìm thấy ảnh)")
            continue

        try:
            with Image.open(src_img) as im:
                w, h = im.size
        except Exception as e:
            skipped_missing_img += 1
            if len(missing_samples) < 5:
                missing_samples.append(str(src_img))
            if processed <= 3:
                print(f"  → Bỏ qua (lỗi mở ảnh: {e})")
            continue

        # YOLO labels
        rel_out = Path(_normalize_rel_path(rel_path))
        lbl_out = save_dir / "labels" / split / rel_out.with_suffix(".txt")
        lbl_out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            xc = (x1 + x2) / 2.0 / w
            yc = (y1 + y2) / 2.0 / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            skipped_no_bbox += 1
            if processed <= 3:
                print("  → Bỏ qua (không có bbox hợp lệ)")
            continue

        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        dst_img = save_dir / "images" / split / rel_out
        ensure_link_or_copy(src_img, dst_img)

        kept += 1
        splits_count[split] = splits_count.get(split, 0) + 1

        if processed <= 3:
            print(f"  → ✓ Lưu thành công vào {split}")

    # data.yaml
    yaml_path = save_dir / "deepfashion.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {save_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            f"  0: {class_name}\n"
        )

    print("\n[✓] Hoàn tất!")
    print(f"  - Giữ: {kept} ảnh")
    print(f"  - Bỏ qua (không có split): {skipped_no_split}")
    print(f"  - Bỏ qua (thiếu ảnh): {skipped_missing_img}")
    print(f"  - Bỏ qua (thiếu bbox): {skipped_no_bbox}")
    print(f"[i] Phân bố: {splits_count}")
    print(f"[i] data.yaml: {yaml_path}")
    if missing_samples:
        print("[i] Ví dụ ảnh không tìm thấy (tối đa 5):")
        for s in missing_samples:
            print(f"    - {s}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert DeepFashion (Consumer-to-Shop/In-shop) -> YOLO detection")
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=False,
        default=Path("datasets/consumer_to_shop_Clothes_Retrieval_Benchmark"),
        help="Thư mục gốc chứa Anno/, Eval/ và (có thể) ảnh",
    )
    ap.add_argument(
        "--images-root",
        type=Path,
        required=False,
        help="Chỉ rõ thư mục ảnh nếu nằm ngoài dataset-root (vd: ...\\img_highres)",
    )
    ap.add_argument(
        "--save-dir",
        type=Path,
        required=False,
        default=Path("datasets/deepfashion_yolo"),
        help="Thư mục xuất images/, labels/, deepfashion.yaml",
    )
    ap.add_argument("--class-name", type=str, default="clothes")
    args = ap.parse_args()

    convert_deepfashion_cts_to_yolo(
        args.dataset_root,
        args.save_dir,
        class_name=args.class_name,
        images_root=args.images_root,
    )
