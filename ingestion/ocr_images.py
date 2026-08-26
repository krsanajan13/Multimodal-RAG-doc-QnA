"""
ingestion/ocr_images.py

Performs OCR on extracted images using pytesseract and Pillow.
Returns OCR text and basic layout info for each image.
"""
from pathlib import Path
from typing import List, Dict
import pytesseract
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr_images")


def ocr_single_image(image_path: str, lang: str = "eng", psm: int = 3) -> Dict:
    """
    OCR a single image and return text and basic details.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(path)
    # pytesseract options
    config = f"--psm {psm}"
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.DICT)
        return {
            "image": str(path.resolve()),
            "text": text.strip(),
            "ocr_data": data,
            "width": img.width,
            "height": img.height
        }
    except Exception as e:
        logger.exception("OCR failed for %s: %s", image_path, e)
        return {
            "image": str(path.resolve()),
            "text": "",
            "ocr_data": {},
            "width": img.width if hasattr(img, "width") else None,
            "height": img.height if hasattr(img, "height") else None,
            "error": str(e)
        }


def ocr_images(image_paths: List[str], lang: str = "eng") -> List[Dict]:
    """
    OCR multiple images. Returns list of dicts with OCR text and metadata.
    """
    results = []
    for p in image_paths:
        res = ocr_single_image(p, lang=lang)
        results.append(res)
    return results


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python ocr_images.py image1.png image2.jpg ...")
        sys.exit(1)
    outs = ocr_images(sys.argv[1:])
    print(json.dumps(outs[:2], indent=2))
