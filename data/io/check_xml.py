"""Audit PCB_DATASET board-level splits and export board/split/class counts."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from convert_voc_to_yolo import CLASS_NAMES, IMG_EXTENSIONS, board_id_from_filename, discover_records


SPLITS = ("train", "val", "test")


def load_assignment(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assignment = {}
    duplicates = defaultdict(set)
    if all(isinstance(manifest.get(split), list) for split in SPLITS):
        for split in SPLITS:
            for board_id in manifest[split]:
                board_id = str(board_id)
                if board_id in assignment:
                    duplicates[board_id].update((assignment[board_id], split))
                assignment[board_id] = split
    else:
        legacy = manifest.get("assignment", {})
        for board_id, split in legacy.items():
            if split not in SPLITS:
                raise ValueError(f"Unknown split {split!r} for board {board_id}")
            assignment[str(board_id)] = split
    return assignment, duplicates


def audit_dataset(dataset_root, manifest_path, yolo_root, csv_path):
    records = discover_records(dataset_root)
    assignment, duplicate_assignments = load_assignment(manifest_path)
    source_boards = defaultdict(list)
    for record in records:
        source_boards[record["board_id"]].append(record)

    observed_splits = defaultdict(set)
    yolo_root = Path(yolo_root)
    for split in SPLITS:
        image_dir = yolo_root / "images" / split
        if image_dir.is_dir():
            for image_path in image_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in IMG_EXTENSIONS:
                    board_id = board_id_from_filename(image_path.name)
                    observed_splits[board_id].add(split)

    board_ids = sorted(set(source_boards) | set(assignment) | set(observed_splits))
    leaked_splits = {}
    for board_id in board_ids:
        splits = set(observed_splits[board_id])
        if board_id in assignment:
            splits.add(assignment[board_id])
        splits.update(duplicate_assignments[board_id])
        if len(splits) > 1:
            leaked_splits[board_id] = sorted(splits)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("board_id", "split", "assigned", "class_name", "image_count", "box_count", "leakage", "leaked_splits"),
        )
        writer.writeheader()
        for board_id in board_ids:
            for split in SPLITS:
                selected = assignment.get(board_id) == split
                split_records = source_boards.get(board_id, []) if selected else []
                for class_id, class_name in enumerate(CLASS_NAMES):
                    class_records = [
                        record for record in split_records
                        if any(box[0] == class_id for box in record["boxes"])
                    ]
                    box_count = sum(
                        box[0] == class_id
                        for record in split_records
                        for box in record["boxes"]
                    )
                    writer.writerow({
                        "board_id": board_id,
                        "split": split,
                        "assigned": selected,
                        "class_name": class_name,
                        "image_count": len(class_records),
                        "box_count": box_count,
                        "leakage": board_id in leaked_splits,
                        "leaked_splits": ";".join(leaked_splits.get(board_id, [])),
                    })

    return {
        "board_count": len(source_boards),
        "assignment_count": len(assignment),
        "leaks": leaked_splits,
        "csv_path": csv_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("PCB_DATASET"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/board_split.json"))
    parser.add_argument("--yolo", type=Path, default=Path("dataset/pcb_boards_yolo"))
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.csv"))
    args = parser.parse_args()

    report = audit_dataset(args.source, args.manifest, args.yolo, args.output)
    if report["leaks"]:
        print(f"Rò rỉ dữ liệu: CÓ - {report['leaks']}")
    else:
        print("Rò rỉ dữ liệu: KHÔNG phát hiện board xuất hiện ở nhiều tập.")
    print(f"Đã xuất {report['csv_path']} ({report['board_count']} boards).")


if __name__ == "__main__":
    main()
