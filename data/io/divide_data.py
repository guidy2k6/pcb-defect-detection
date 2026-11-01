"""Split PCB_DATASET by board ID, create YOLO labels, and audit the result."""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from check_xml import audit_dataset, load_assignment
from convert_voc_to_yolo import CLASS_NAMES, convert_record, discover_records


SPLIT_SIZES = {"train": 6, "val": 2, "test": 2}


def summarize_boards(records):
    boards = defaultdict(lambda: {"boxes": 0, "classes": Counter(), "records": []})
    for record in records:
        board = boards[record["board_id"]]
        board["records"].append(record)
        board["boxes"] += len(record["boxes"])
        board["classes"].update(box[0] for box in record["boxes"])
    if len(boards) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"Expected exactly 10 unique board IDs, found {len(boards)}: {sorted(boards)}")
    return dict(boards)


def _box_totals(assignment, boards):
    return {
        split: sum(boards[board]["boxes"] for board, assigned in assignment.items() if assigned == split)
        for split in SPLIT_SIZES
    }


def _score(assignment, boards):
    totals = _box_totals(assignment, boards)
    per_board = [totals[split] / count for split, count in SPLIT_SIZES.items()]
    ratio = max(per_board) / min(per_board) if min(per_board) else float("inf")
    mean = sum(per_board) / len(per_board)
    variance = sum((value - mean) ** 2 for value in per_board) / len(per_board)
    return ratio, variance


def _has_all_classes(split, assignment, boards):
    present = set()
    for board_id, assigned in assignment.items():
        if assigned == split:
            present.update(boards[board_id]["classes"])
    return present == set(range(len(CLASS_NAMES)))


def optimize_split(boards, max_box_ratio):
    board_ids = sorted(boards)
    best_assignment, best_score = None, None
    for train in itertools.combinations(board_ids, 6):
        remaining = sorted(set(board_ids) - set(train))
        for val in itertools.combinations(remaining, 2):
            test = set(remaining) - set(val)
            assignment = {board: "train" for board in train}
            assignment.update({board: "val" for board in val})
            assignment.update({board: "test" for board in test})
            if not all(_has_all_classes(split, assignment, boards) for split in SPLIT_SIZES):
                continue
            score = _score(assignment, boards)
            if best_score is None or score < best_score:
                best_assignment, best_score = assignment, score

    if best_assignment is None:
        class_boards = {
            CLASS_NAMES[class_id]: sorted(board for board in boards if boards[board]["classes"][class_id])
            for class_id in range(len(CLASS_NAMES))
        }
        raise ValueError(f"Cannot make 6/2/2 splits with all six classes. Boards by class: {class_boards}")
    if best_score[0] > max_box_ratio:
        raise ValueError(
            f"Best valid split has max/min boxes-per-board ratio {best_score[0]:.2f}, "
            f"above the limit {max_box_ratio:.2f}. Box totals: {_box_totals(best_assignment, boards)}"
        )
    return best_assignment


def _record_swaps(initial, target, boards, ratio):
    current = dict(initial)
    swaps = []
    while any(current[board] != target[board] for board in current):
        board = next(board for board in sorted(current) if current[board] != target[board])
        source, destination = current[board], target[board]
        partner = next(
            (other for other in sorted(current) if current[other] == destination and target[other] == source),
            None,
        )
        if partner is None:
            partner = next(other for other in sorted(current) if current[other] == destination)
        swaps.append({
            "boards": [board, partner],
            "reason": (
                f"Move board {board} to {destination} and board {partner} to {source}; "
                f"the selected split covers all six classes and has box ratio {ratio:.2f}."
            ),
        })
        current[board], current[partner] = current[partner], current[board]
    return swaps


def _write_data_yaml(path, output_root):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except yaml.YAMLError:
        config = {}
    config.update({
        "path": Path(__import__("os").path.relpath(output_root.resolve(), path.parent.resolve())).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    })
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("PCB_DATASET"), help="Contains images/ and Annotations/ class folders")
    parser.add_argument("--output", type=Path, default=Path("dataset/pcb_boards_yolo"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_board.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/board_split.json"))
    parser.add_argument("--audit", type=Path, default=Path("results/data_audit.csv"))
    parser.add_argument("--max-box-ratio", type=float, default=1.5)
    parser.add_argument("--rebuild-split", action="store_true")
    args = parser.parse_args()

    source, output = args.source.resolve(), args.output.resolve()
    manifest_path = args.manifest.resolve()
    records = discover_records(source)
    boards = summarize_boards(records)
    board_ids = sorted(boards)

    if manifest_path.exists() and not args.rebuild_split:
        assignment, duplicates = load_assignment(manifest_path)
        if duplicates:
            raise ValueError(f"Boards occur in multiple manifest splits: {dict(duplicates)}")
        if set(assignment) != set(board_ids):
            raise ValueError(f"Manifest board IDs differ from source; use --rebuild-split to replace it")
        if Counter(assignment.values()) != Counter(SPLIT_SIZES):
            raise ValueError("Manifest must contain exactly 6 train, 2 val, and 2 test boards")
        if not all(_has_all_classes(split, assignment, boards) for split in SPLIT_SIZES):
            raise ValueError("The fixed manifest does not give every split all six classes")
        if _score(assignment, boards)[0] > args.max_box_ratio:
            raise ValueError("The fixed manifest exceeds --max-box-ratio; review it or use --rebuild-split")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        assignment = optimize_split(boards, args.max_box_ratio)
        initial = {
            board: "train" if index < 6 else "val" if index < 8 else "test"
            for index, board in enumerate(board_ids)
        }
        manifest = {
            split: sorted(board for board, assigned in assignment.items() if assigned == split)
            for split in SPLIT_SIZES
        }
        manifest["swaps"] = _record_swaps(initial, assignment, boards, _score(assignment, boards)[0])
        manifest["selection_reason"] = "Exhaustive 6/2/2 search; every split contains all six classes and box counts are balanced."
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for split in SPLIT_SIZES:
        records_for_split = [record for record in records if assignment[record["board_id"]] == split]
        for record in records_for_split:
            convert_record(record, output / "images" / split, output / "labels" / split)

    _write_data_yaml(args.config.resolve(), output)
    manifest["stats"] = {
        "box_counts": _box_totals(assignment, boards),
        "boxes_per_board": {
            split: _box_totals(assignment, boards)[split] / count
            for split, count in SPLIT_SIZES.items()
        },
        "max_min_boxes_per_board_ratio": _score(assignment, boards)[0],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = audit_dataset(source, manifest_path, output, args.audit.resolve())
    print(f"Split board IDs saved to {manifest_path}")
    for split, count in SPLIT_SIZES.items():
        print(f"{split}: {manifest[split]} ({count} boards)")
    print(f"Board leakage: {'YES' if report['leaks'] else 'NO'}; audit: {report['csv_path']}")


if __name__ == "__main__":
    main()