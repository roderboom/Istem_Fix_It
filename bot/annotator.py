"""
annotator.py — Draws numbered dots on the image for each component.

Dots identify the physical parts the user will need to touch.
They are completely independent of the repair steps.
"""

import io
import logging
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

DOT_RADIUS    = 14
OUTLINE_WIDTH = 3
WHITE         = (255, 255, 255, 245)
DARK          = (20,  20,  20,  220)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def annotate_image(image_bytes: bytes, analysis: dict) -> bytes:
    """
    Draw a numbered white dot for each component in analysis["components"].
    Returns JPEG bytes.
    """
    components = analysis.get("components", [])

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)  # apply EXIF rotation before placing dots
    img = img.convert("RGBA")

    # Scale up small images so dots look proportional
    min_dim = 700
    w, h = img.size
    if min(w, h) < min_dim:
        scale = min_dim / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    W, H = img.size

    dot_r     = max(DOT_RADIUS, W // 55)
    outline_r = dot_r + OUTLINE_WIDTH
    font      = _load_font(max(12, dot_r - 2))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    for comp in components:
        pt  = comp.get("point", {"x": 50, "y": 50})
        num = comp.get("id", 1)
        cx  = int(pt["x"] / 100 * W)
        cy  = int(pt["y"] / 100 * H)

        # Dark outline ring
        draw.ellipse(
            [cx - outline_r, cy - outline_r, cx + outline_r, cy + outline_r],
            fill=DARK,
        )
        # White filled dot
        draw.ellipse(
            [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
            fill=WHITE,
        )
        # Component number
        text = str(num)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, cy - th // 2 - 1),
            text,
            fill=(20, 20, 20, 255),
            font=font,
        )

    img = Image.alpha_composite(img, layer)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()
