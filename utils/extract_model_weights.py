import torch
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_model_weights(ckpt_path: str, output_path: str = None, exclude_class_head: bool = False):
    """
    Extract model weights from an ATLAS Lightning checkpoint and save as .pth file.
    
    Args:
        ckpt_path: Path to the .ckpt checkpoint file
        output_path: Path for the output .pth file (optional, defaults to same name)
        exclude_class_head: If True, excludes class head weights from the output
    """
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict from Lightning checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logging.info("Found 'state_dict' key in checkpoint")
    else:
        state_dict = checkpoint
        logging.info("Using checkpoint directly as state_dict")
    
    # Remove Lightning-specific prefixes (e.g., "._orig_mod")
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    
    # Optionally exclude class head
    if exclude_class_head:
        original_len = len(state_dict)
        state_dict = {
            k: v for k, v in state_dict.items()
            if "class_head" not in k and "class_predictor" not in k
        }
        logging.info(f"Excluded class head ({original_len - len(state_dict)} keys removed)")
    
    # Set output path
    if output_path is None:
        output_path = ckpt_path.parent / f"{ckpt_path.stem}.pth"
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .pth file
    logging.info(f"Saving {len(state_dict)} keys to: {output_path}")
    torch.save(state_dict, output_path)
    
    logging.info("✅ Successfully extracted model weights")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract model weights from ATLAS Lightning checkpoint"
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to the .ckpt checkpoint file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for the .pth file (default: same directory as input)"
    )
    parser.add_argument(
        "--exclude-class-head",
        action="store_true",
        help="Exclude class head weights from output"
    )
    
    args = parser.parse_args()
    
    extract_model_weights(
        args.ckpt_path,
        args.output,
        args.exclude_class_head
    )


if __name__ == "__main__":
    main()
