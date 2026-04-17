"""Prepare COCO image subsets for PyWatermark training and evaluation."""

from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from config import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for COCO preparation."""

    parser = argparse.ArgumentParser(
        description="Download or split COCO images into train/val/test folders for PyWatermark.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing extracted image directory. If provided, download is skipped.",
    )
    parser.add_argument(
        "--download-split",
        type=str,
        choices=tuple(DEFAULT_CONFIG.dataset_prep.coco_image_urls.keys()),
        default=DEFAULT_CONFIG.dataset_prep.coco_download_split,
        help="Official COCO image split zip to download when --source-dir is not provided.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_CONFIG.paths.raw_data_dir,
        help="Directory where raw COCO zips and extracted images are stored.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CONFIG.paths.data_root,
        help="Directory where train/val/test folders are created.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on the number of source images before splitting.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=DEFAULT_CONFIG.dataset_prep.train_count,
        help="Number of images to place in the train split.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=DEFAULT_CONFIG.dataset_prep.val_count,
        help="Number of images to place in the validation split.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=DEFAULT_CONFIG.dataset_prep.test_count,
        help="Number of images to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.runtime.random_seed,
        help="Random seed used for shuffling image paths.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into train/val/test instead of moving them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing train/val/test directories under --output-root before writing new splits.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> Path:
    """Download a file to disk if it does not already exist."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using existing archive: {destination.resolve()}")
        return destination

    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    print(f"Saved archive to: {destination.resolve()}")
    return destination


def extract_zip(archive_path: Path, extract_root: Path) -> Path:
    """Extract a zip archive and return the extracted image directory."""

    with zipfile.ZipFile(archive_path) as archive:
        top_level_names = sorted({name.split("/", 1)[0] for name in archive.namelist() if name and not name.endswith("/")})
        if not top_level_names:
            raise ValueError(f"Archive contained no files: {archive_path}")
        if len(top_level_names) != 1:
            raise ValueError(f"Expected a single top-level directory in {archive_path}, found {top_level_names}.")
        output_dir = extract_root / top_level_names[0]
        if output_dir.exists():
            existing_files = [path for path in output_dir.rglob("*") if path.is_file()]
            if existing_files:
                print(f"Using existing extracted directory: {output_dir.resolve()}")
                return output_dir

            print(f"Existing extracted directory is empty, re-extracting: {output_dir.resolve()}")
            shutil.rmtree(output_dir)

        print(f"Extracting {archive_path.name} into {extract_root.resolve()}")
        archive.extractall(extract_root)
    return output_dir


def collect_image_paths(source_dir: Path) -> list[Path]:
    """Collect image paths recursively from a directory."""

    supported_extensions = set(DEFAULT_CONFIG.data.extensions)
    candidate_root = source_dir
    image_paths = [
        path
        for path in candidate_root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]
    if not image_paths:
        child_directories = [path for path in source_dir.iterdir() if path.is_dir()]
        if len(child_directories) == 1:
            candidate_root = child_directories[0]
            image_paths = [
                path
                for path in candidate_root.rglob("*")
                if path.is_file() and path.suffix.lower() in supported_extensions
            ]
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(f"No supported images were found in: {source_dir}")
    return image_paths


def choose_images(image_paths: list[Path], max_images: int | None, seed: int) -> list[Path]:
    """Shuffle image paths reproducibly and apply an optional cap."""

    shuffled = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if max_images is not None:
        return shuffled[:max_images]
    return shuffled


def ensure_clean_split_directories(output_root: Path, force: bool) -> dict[str, Path]:
    """Create train/val/test directories, optionally clearing prior contents."""

    split_dirs = {
        "train": output_root / "train",
        "val": output_root / "val",
        "test": output_root / "test",
    }
    for split_dir in split_dirs.values():
        if split_dir.exists():
            if not force:
                raise FileExistsError(
                    f"Output split already exists: {split_dir}. Use --force to replace existing splits."
                )
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
    return split_dirs


def assign_splits(image_paths: list[Path], train_count: int, val_count: int, test_count: int) -> dict[str, list[Path]]:
    """Split image paths into train, validation, and test groups."""

    total_requested = train_count + val_count + test_count
    if total_requested <= 0:
        raise ValueError("At least one of train_count, val_count, or test_count must be positive.")
    if total_requested > len(image_paths):
        raise ValueError(
            f"Requested {total_requested} images across splits, but only found {len(image_paths)} source images."
        )

    train_end = train_count
    val_end = train_count + val_count
    return {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:total_requested],
    }


def write_split(split_dir: Path, image_paths: Iterable[Path], copy_files: bool) -> None:
    """Copy or move images into a dataset split directory."""

    for index, image_path in enumerate(image_paths):
        destination = split_dir / f"{index:06d}{image_path.suffix.lower()}"
        if copy_files:
            shutil.copy2(image_path, destination)
        else:
            shutil.move(str(image_path), destination)


def resolve_source_dir(args: argparse.Namespace) -> Path:
    """Resolve or download the image directory used as the source split pool."""

    if args.source_dir is not None:
        source_dir = args.source_dir.expanduser()
        if not source_dir.exists():
            raise FileNotFoundError(f"Provided source directory does not exist: {source_dir}")
        return source_dir

    raw_dir = args.raw_dir.expanduser()
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = DEFAULT_CONFIG.dataset_prep.coco_image_urls[args.download_split]
    archive_path = download_file(url, raw_dir / Path(url).name)
    return extract_zip(archive_path, raw_dir)


def main() -> None:
    """Program entry point."""

    args = parse_args()
    source_dir = resolve_source_dir(args)
    image_paths = collect_image_paths(source_dir)
    chosen_images = choose_images(image_paths, args.max_images, args.seed)
    split_mapping = assign_splits(chosen_images, args.train_count, args.val_count, args.test_count)
    split_dirs = ensure_clean_split_directories(args.output_root.expanduser(), force=args.force)
    copy_files = args.copy or args.source_dir is not None

    for split_name, split_paths in split_mapping.items():
        write_split(split_dirs[split_name], split_paths, copy_files=copy_files)

    print(f"Prepared dataset under: {args.output_root.expanduser().resolve()}")
    print(f"Source directory: {source_dir.resolve()}")
    print(f"Transfer mode: {'copy' if copy_files else 'move'}")
    for split_name, split_paths in split_mapping.items():
        print(f"{split_name}: {len(split_paths)} images")


if __name__ == "__main__":
    main()
