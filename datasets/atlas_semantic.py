from pathlib import Path
from typing import Union, Optional
from torch.utils.data import DataLoader
import torch
from torchvision import tv_tensors
from PIL import Image
from io import BytesIO
import zipfile
import numpy as np
import re

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


class ATLASSemantic(LightningDataModule):
    """Data module for ATLAS dataset with zip-based storage.
    
    Expects structure: atlas.zip -> atlas/ -> train/, val/, test/
    Each split contains: procedure_*/ -> video_*/ -> clip_*/ -> images/, machine_masks/
    """

    def __init__(
        self,
        path: str,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 2,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        clip_length: int = 1,
        clip_stride: int = 1,
        val_clip_length: int = 1,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.val_clip_length = max(1, int(val_clip_length))

    @staticmethod
    def target_parser(target, **kwargs):
        """Parse target mask into individual object masks and labels.
        
        Background (class 0) is preserved as a class but skipped from instance extraction.
        This allows proper handling in loss functions that support ignore_idx.
        """
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            # if cls_id == 0:  # Skip background instances
            #     continue

            masks.append(target[0] == label_id)
            #labels.append(cls_id - 1)  # Adjust to 0-indexed
            labels.append(cls_id)  # Do not adjust to 0-indexed

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        self.train_dataset = ATLASZipDataset(
            zip_path=Path(self.path),
            split="train",
            transforms=self.transforms,
            target_parser=self.target_parser,
            check_empty_targets=self.check_empty_targets,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
        )

        self.val_dataset = ATLASZipDataset(
            zip_path=Path(self.path),
            split="val",
            transforms=None,
            target_parser=self.target_parser,
            check_empty_targets=self.check_empty_targets,
            clip_length=self.val_clip_length,
            clip_stride=1,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class ATLASZipDataset(torch.utils.data.Dataset):
    """Zip-based dataset loader for ATLAS with nested structure.
    
    Zip structure:
        atlas.zip
        └── atlas/
            ├── train/
            │   ├── procedure_001/
            │   │   ├── video_001/
            │   │   │   ├── clip_001/
            │   │   │   │   ├── images/
            │   │   │   │   │   └── frame_*.jpg
            │   │   │   │   └── machine_masks/
            │   │   │   │       └── frame_*.png
            ...
            ├── val/
            ...
            └── test/
                ...
    """

    def __init__(
        self,
        zip_path: Path,
        split: str,
        img_suffix: str = ".jpg",
        target_suffix: str = ".png",
        target_parser: Optional[callable] = None,
        transforms: Optional[callable] = None,
        check_empty_targets: bool = True,
        clip_length: int = 1,
        clip_stride: int = 1,
    ):
        self.zip_path = zip_path
        self.split = split
        self.img_suffix = img_suffix
        self.target_suffix = target_suffix
        self.target_parser = target_parser
        self.transforms = transforms
        self.check_empty_targets = check_empty_targets
        self.clip_length = max(1, int(clip_length))
        self.clip_stride = max(1, int(clip_stride))

        self.imgs = []
        self.targets = []
        self.sequences = []
        
        # Per-worker zip file handles
        self.zip = {}

        self._build_index()

    def _build_index(self):
        """Build index of images and corresponding masks from zip file."""
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            filenames = zf.namelist()

        # Find all image and mask files for this split
        img_mask_pairs = {}

        for filename in filenames:
            # Check if file is in the split directory
            if f"/{self.split}/" not in filename:
                continue

            # Check if it's an image
            if filename.endswith(self.img_suffix) and "/images/" in filename:
                # Extract relative path and construct mask path
                parts = filename.split(f"/{self.split}/")[1]
                img_relpath = parts.split("/images/")[1]
                base_path = parts.split("/images/")[0]

                mask_relpath = img_relpath.replace(self.img_suffix, self.target_suffix)
                mask_filename = f"{self.split}/{base_path}/machine_masks/{mask_relpath}"
                
                # Reconstruct full path
                prefix = filename.split(f"/{self.split}/")[0]
                mask_full_path = f"{prefix}/{mask_filename}"

                if mask_full_path in filenames:
                    if self.check_empty_targets:
                        # Verify mask is not empty
                        with zipfile.ZipFile(self.zip_path, "r") as zf:
                            try:
                                with zf.open(mask_full_path) as f:
                                    mask_img = Image.open(BytesIO(f.read()))
                                    if mask_img.getextrema() == (0, 0):
                                        continue
                            except Exception:
                                continue

                    img_mask_pairs[filename] = mask_full_path

        # Sort and build lists
        for img_path in sorted(img_mask_pairs.keys()):
            self.imgs.append(img_path)
            self.targets.append(img_mask_pairs[img_path])

        if self.split == "train" and self.clip_length > 1:
            clip_to_indices = {}
            for i, img_path in enumerate(self.imgs):
                clip_key = self._clip_key_from_path(img_path)
                clip_to_indices.setdefault(clip_key, []).append(i)

            for indices in clip_to_indices.values():
                if len(indices) < self.clip_length:
                    continue
                for start in range(0, len(indices) - self.clip_length + 1, self.clip_stride):
                    self.sequences.append(indices[start : start + self.clip_length])

    def __len__(self):
        if self.split == "train" and self.clip_length > 1:
            return len(self.sequences)
        return len(self.imgs)

    def _get_worker_id(self):
        """Get current worker id for multi-worker data loading."""
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info else 0

    def _get_zip_file(self):
        """Get or create zip file handle for current worker."""
        worker_id = self._get_worker_id()
        if worker_id not in self.zip:
            self.zip[worker_id] = zipfile.ZipFile(self.zip_path, "r")
        return self.zip[worker_id]

    def __getitem__(self, index: int):
        if self.split == "train" and self.clip_length > 1:
            return self._get_sequence_item(index)

        img_path = self.imgs[index]
        mask_path = self.targets[index]

        procedure_id = self._procedure_id_from_path(img_path)

        # Load image and mask from zip using persistent worker handle
        zf = self._get_zip_file()
        with zf.open(img_path) as f:
            img_array = np.array(Image.open(BytesIO(f.read())).convert("RGB"))
            img = tv_tensors.Image(torch.from_numpy(img_array).permute(2, 0, 1))

        with zf.open(mask_path) as f:
            mask_array = np.array(Image.open(BytesIO(f.read())))
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
            # Apply class remapping
            mask_array = remap_mask(mask_array, mapping)
            mask = tv_tensors.Mask(torch.from_numpy(mask_array).long())

        # Parse masks
        masks, labels, is_crowd = self.target_parser(target=[mask])

        target_dict = {
            "masks": tv_tensors.Mask(
                torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:]))
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd),
            "procedure": torch.tensor(procedure_id, dtype=torch.long),
            "frame_index": torch.tensor(self._frame_index_from_path(img_path), dtype=torch.long),
        }

        if self.transforms is not None:
            img, target_dict = self.transforms(img, target_dict)

        return img, target_dict

    def _get_sequence_item(self, index: int):
        sequence_indices = self.sequences[index]
        seed = torch.randint(0, 2**31 - 1, (1,)).item()

        imgs = []
        targets = []
        for t, sample_idx in enumerate(sequence_indices):
            img_path = self.imgs[sample_idx]
            mask_path = self.targets[sample_idx]
            procedure_id = self._procedure_id_from_path(img_path)

            zf = self._get_zip_file()
            with zf.open(img_path) as f:
                img_array = np.array(Image.open(BytesIO(f.read())).convert("RGB"))
                img = tv_tensors.Image(torch.from_numpy(img_array).permute(2, 0, 1))

            with zf.open(mask_path) as f:
                mask_array = np.array(Image.open(BytesIO(f.read())))
                if mask_array.ndim == 3:
                    mask_array = mask_array[0]
                mask_array = remap_mask(mask_array, mapping)
                mask = tv_tensors.Mask(torch.from_numpy(mask_array).long())

            masks, labels, is_crowd = self.target_parser(target=[mask])
            target_dict = {
                "masks": tv_tensors.Mask(
                    torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:]))
                ),
                "labels": torch.tensor(labels, dtype=torch.long),
                "is_crowd": torch.tensor(is_crowd),
                "procedure": torch.tensor(procedure_id, dtype=torch.long),
                "frame_index": torch.tensor(self._frame_index_from_path(img_path), dtype=torch.long),
                "time_index": torch.tensor(t, dtype=torch.long),
            }

            if self.transforms is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    img, target_dict = self.transforms(img, target_dict)

            imgs.append(img)
            targets.append(target_dict)

        return torch.stack(imgs), targets

    def _clip_key_from_path(self, img_path: str) -> str:
        rel_path = img_path.split(f"/{self.split}/", 1)[1]
        return rel_path.split("/images/", 1)[0]

    def _frame_index_from_path(self, img_path: str) -> int:
        frame_name = Path(img_path).stem
        matches = re.findall(r"\d+", frame_name)
        if matches:
            return int(matches[-1])
        return 0

    def _procedure_id_from_path(self, img_path: str) -> int:
        try:
            rel_path = img_path.split(f"/{self.split}/", 1)[1]
        except IndexError:
            raise ValueError(
                f"Unable to parse split '{self.split}' from path: {img_path}"
            )

        procedure_folder = rel_path.split("/", 1)[0]
        if procedure_folder in PROCEDURE_NAME_TO_ID:
            return PROCEDURE_NAME_TO_ID[procedure_folder]

        if procedure_folder.startswith("procedure_"):
            suffix = procedure_folder.split("procedure_", 1)[1]
            if suffix.isdigit():
                return int(suffix) - 1

        raise ValueError(
            "Unknown procedure folder: "
            f"{procedure_folder}. Expected one of {list(PROCEDURE_NAME_TO_ID.keys())} "
            "or a procedure_### pattern."
        )
    


mapped_class_names  = {
    0:  "Background",
    1:  "Tools/camera",
    2:  "Vein",
    3:  "Artery",  
    4:  "Nerve",
    5:  "Small intestine",
    6:  "Colon/rectum",
    7:  "Abdominal wall",
    8:  "Diaphragm",
    9:  "Fat",
    10: "Liver",
    11: "Bile/lymph Duct",
    12: "Gallbladder",
    13: "Hepatic ligament",
    14: "Cystic plate",
    15: "Stomach",
    16: "Spleen",
    17: "Uterus",
    18: "Ovary",
    19: "Oviduct",
    20: "Prostate",
    21: "Urethra",
    22: "Ligated plexus",
    23: "Seminal vesicles",
    24: "Non anatomical",
    25: "Bladder",
    26: "Lung",
    27: "Airway (bronchus/trachea)",
    28: "Esophagus",
    29: "Pericardium",
}

PROCEDURE_NAMES = [
    "adrenalectomy",
    "appendectomy",
    "cholecystectomy",
    "colectomy",
    "esophagectomy",
    "gastric_surgery",
    "gastrojejunostomy",
    "hemicolectomy",
    "lar",
    "liver_resection",
    "rarp",
    "rectopexy",
    "sigmoidcolectomy",
    "splenectomy",
]

PROCEDURE_NAME_TO_ID = {name: idx for idx, name in enumerate(PROCEDURE_NAMES)}

mapping = {
    0:  0,  # Background"
    1:  1,  # Tools/camera
    2:  2,  # Vein
    3:  3,  # Artery
    4:  4,  # Nerve
    5:  5,  # Small intestine
    6:  6,  # Colon/rectum
    7:  7,  # Abdominal wall
    8:  8,  # Diaphragm
    9:  9,  # Omentum
    10: 3,  # Aorta => Artery
    11: 2,  # Vena cava => Vein
    12: 10, # Liver
    13: 11, # Cystic duct => Bile/lymph Duct
    14: 12, # Gallbladder
    15: 2,  # Hepatic vein => Vein
    16: 13, # Hepatic ligament
    17: 14, # Cystic plate
    18: 15, # Stomach
    19: 11, # Ductus choledochus => Bile/lymph Duct
    20: 9,  # Mesenterium => Fat
    21: 11, # Ductus hepaticus => Bile/lymph Duct
    22: 16, # Spleen
    23: 17, # Uterus
    24: 18, # Ovary
    25: 19, # Oviduct
    26: 20, # Prostate
    27: 21, # Urethra
    28: 22, # Ligated plexus
    29: 23, # Seminal vesicles
    30: 24, # Catheter => Non anatomical
    31: 25, # Bladder
    32: 0,  # Kidney => Background
    33: 26, # Lung
    34: 27, # Airway (bronchus/trachea)
    35: 28, # Esophagus
    36: 29, # Pericardium
    37: 2,  # V azygos => Vein
    38: 11, # Thoracic duct => Bile/lymph Duct
    39: 4,  # Nerves => Nerve
    40: 0,  # Ureter => Background
    41: 24, # Non anatomical structures => Non anatomical
    42: 0,  # Excluded frames => Background
    43: 0,  # Mesocolon => Background
    44: 0,  # Adrenal Gland => Background
    45: 0,  # Pancreas => Background
    46: 0,  # Duodenum => Background
}

def remap_mask(mask: np.ndarray, mapping: dict, default_value=0) -> np.ndarray:
    """
    Remap class labels in a 2D segmentation mask using a lookup table (LUT).

    Pixel-wise labels in `mask` are converted according to `mapping` using
    fast NumPy indexing. Labels not present in `mapping` are set to
    `default_value`.

    Parameters
    ----------
    mask : np.ndarray
        2D array of shape (H, W) containing integer class labels.
    mapping : dict
        Mapping from original labels to new labels.
    default_value : int, optional
        Value assigned to unmapped labels (default: 0).

    Returns
    -------
    np.ndarray
        Remapped mask of shape (H, W).
    """

    if mask.ndim != 2:
        raise ValueError("Mask must be 2D (H, W)")

    mask = mask.astype(np.int32)

    # LUT size based on maximum possible label
    max_label = max(mapping.keys())  # safer than mask.max()

    # Fill LUT with default value
    lut = np.full(max_label + 1, default_value, dtype=np.int32)

    # Assign mappings
    for k, v in mapping.items():
        lut[k] = v

    # Apply mapping (vectorized, extremely fast)
    return lut[mask]

