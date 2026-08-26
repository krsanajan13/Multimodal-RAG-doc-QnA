"""
ingestion/extract_images.py

Extracts embedded images from the PDF and writes them to a given output directory.
Returns metadata for each image (page, xref, filename, bbox).
"""
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
import logging
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_images")


def extract_images(pdf_path: str, out_dir: str) -> List[Dict]:
    """
    Extract images from PDF and save them as PNG files.

    Args:
        pdf_path: path to source PDF
        out_dir: directory to write images to (will be created)

    Returns:
        List of metadata dicts:
        [
          {
            "page": int,
            "xref": int,
            "name": "p{page}_img{xref}.png",
            "path": "/abs/path/..",
            "width": int,
            "height": int,
            "ext": "png"
          }, ...
        ]
    """
    src = Path(pdf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(src))
    images_meta = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        logger.info("Page %s has %s images", page_index + 1, len(image_list))
        for img_idx, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_ext = base_image.get("ext", "png")
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # convert to RGB if necessary
                if image.mode in ("RGBA", "LA"):
                    image = image.convert("RGB")
                fname = f"p{page_index+1}_img{xref}.{img_ext}"
                out_path = out / fname
                image.save(out_path, format=img_ext.upper())
                images_meta.append({
                    "page": page_index + 1,
                    "xref": xref,
                    "name": fname,
                    "path": str(out_path.resolve()),
                    "width": image.width,
                    "height": image.height,
                    "ext": img_ext
                })
            except Exception as e:
                logger.exception("Failed to save image xref %s on page %s: %s", xref, page_index+1, e)
    doc.close()
    return images_meta


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python extract_images.py path/to/doc.pdf out/image/dir")
        sys.exit(1)
    meta = extract_images(sys.argv[1], sys.argv[2])
    print(json.dumps(meta[:5], indent=2))
