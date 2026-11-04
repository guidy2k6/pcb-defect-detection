"""Read PCB_DATASET Pascal VOC annotations and convert them to YOLO labels."""

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_NAMES = ["Missing Hole", "Mouse Bite", "Open Circuit", "Short", "Spur", "Spurious Copper"]
CLASS_IDS = {
    "missinghole": 0,
    "mousebite": 1,
    "opencircuit": 2,
    "short": 3,
    "spur": 4,
    "spuriouscopper": 5,
}
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def board_id_from_filename(filename):
    """The board ID is the filename token before its first underscore."""
    stem = Path(filename).stem
    if "_" not in stem:
        raise ValueError(f"Cannot extract board ID from filename (expected boardId_...): {filename}")
    return stem.split("_", 1)[0]


def _named_child(root, name):
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    raise FileNotFoundError(f"Missing {name}/ directory under {root}")


def _class_directories(root):
    result = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        key = normalize_name(path.name)
        if key in CLASS_IDS:
            result[key] = path
    missing = set(CLASS_IDS) - set(result)
    if missing:
        missing_names = [CLASS_NAMES[CLASS_IDS[key]] for key in sorted(missing)]
        raise ValueError(f"Missing class directories under {root}: {missing_names}")
    return result


def _image_for(root, xml_path, image_dir):
    filename = root.findtext("filename")
    candidate_name = Path(filename).name if filename else f"{xml_path.stem}"
    candidate = image_dir / candidate_name
    if candidate.is_file():
        return candidate
    for extension in IMG_EXTENSIONS:
        candidate = image_dir / f"{xml_path.stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching image found for annotation: {xml_path}")


def discover_records(dataset_root):
    """Read all class-folder XMLs; each record represents one image and its boxes."""
    dataset_root = Path(dataset_root)
    images_root = _named_child(dataset_root, "images")
    annotations_root = _named_child(dataset_root, "Annotations")
    image_dirs = _class_directories(images_root)
    annotation_dirs = _class_directories(annotations_root)
    records = []
    seen_images = set()

    for class_key, annotation_dir in annotation_dirs.items():
        image_dir = image_dirs[class_key]
        for xml_path in sorted(annotation_dir.rglob("*.xml")):
            root = ET.parse(xml_path).getroot()
            image_path = _image_for(root, xml_path, image_dir)
            if image_path in seen_images:
                raise ValueError(f"More than one XML refers to image: {image_path}")
            seen_images.add(image_path)
            width_text = root.findtext("size/width")
            height_text = root.findtext("size/height")
            if not width_text or not height_text:
                raise ValueError(f"Missing image width/height in {xml_path}")
            width, height = int(width_text), int(height_text)
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size in {xml_path}")

            boxes = []
            for obj in root.findall("object"):
                class_name = normalize_name(obj.findtext("name") or "")
                if class_name not in CLASS_IDS:
                    raise ValueError(f"Unknown class '{obj.findtext('name')}' in {xml_path}")
                bounds = obj.find("bndbox")
                if bounds is None:
                    raise ValueError(f"Missing bndbox in {xml_path}")
                coordinates = [bounds.findtext(key) for key in ("xmin", "ymin", "xmax", "ymax")]
                if any(value is None for value in coordinates):
                    raise ValueError(f"Incomplete bndbox in {xml_path}")
                xmin, ymin, xmax, ymax = map(float, coordinates)
                xmin, xmax = max(0.0, xmin), min(float(width), xmax)
                ymin, ymax = max(0.0, ymin), min(float(height), ymax)
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError(f"Invalid bounding box in {xml_path}")
                boxes.append((
                    CLASS_IDS[class_name],
                    ((xmin + xmax) / 2) / width,
                    ((ymin + ymax) / 2) / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ))

            records.append({
                "board_id": board_id_from_filename(image_path.name),
                "image_path": image_path,
                "xml_path": xml_path,
                "boxes": boxes,
            })

    if not records:
        raise ValueError(f"No annotated images found under {annotations_root}")
    return records


def convert_record(record, output_images, output_labels):
    """Copy an image and write its YOLO class/normalized-box rows."""
    output_images, output_labels = Path(output_images), Path(output_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    image_path = record["image_path"]
    image_out = output_images / image_path.name
    label_out = output_labels / f"{image_path.stem}.txt"
    if image_out.exists() and image_out.resolve() != image_path.resolve():
        raise FileExistsError(f"Output filename collision: {image_out.name}")
    shutil.copy2(image_path, image_out)
    label_out.write_text(
        "".join(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for class_id, x, y, w, h in record["boxes"]),
        encoding="utf-8",
    )