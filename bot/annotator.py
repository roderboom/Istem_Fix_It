"""
annotator.py — Draws annotation overlays on repair images.

Two annotation types:
  components — numbered dots (for specific small parts)
  areas      — semi-transparent colored boxes (for regions/zones)

Combined max: 6 annotations.
"""

import io
import logging
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps

logger = logging.getLogger(__name__)

# ── Dot style ─────────────────────────────────────────────────────────────────
DOT_RADIUS     = 16          # base radius, scales with image size
OUTLINE_WIDTH  = 3

# Dot colors: white fill, dark shadow ring, colored accent ring
DOT_FILL       = (255, 255, 255, 250)
DOT_SHADOW     = (0,   0,   0,   180)
DOT_ACCENT     = (255, 80,  60,  230)   # red-orange ring between shadow and fill

# ── Area style ────────────────────────────────────────────────────────────────
AREA_COLORS = [
    (255, 160,  40),   # amber
    ( 80, 180, 255),   # sky blue
    (120, 220,  80),   # green
    (220,  80, 220),   # purple
    ( 80, 220, 200),   # teal
    (255, 220,  60),   # yellow
]
AREA_FILL_ALPHA    = 38     # ~15% opacity fill
AREA_BORDER_ALPHA  = 200    # solid-ish border
AREA_BORDER_WIDTH  = 3


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


def annotate_image(image_bytes: bytes, analysis: dict, photo_index: int = 1) -> bytes:
    """Draw annotations for the given photo_index (1-based).
    Filters components and areas to only those matching photo_index.
    If only one photo was analyzed (all photo fields = 1), photo_index=1 draws everything.
    """
    components = [c for c in analysis.get("components", []) if c.get("photo", 1) == photo_index]
    areas      = [a for a in analysis.get("areas",      []) if a.get("photo", 1) == photo_index]

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")

    # Scale up small images so annotations look proportional
    min_dim = 800
    w, h = img.size
    if min(w, h) < min_dim:
        scale = min_dim / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    W, H = img.size

    # ── Draw areas (bottom layer) ─────────────────────────────────────────────
    if areas:
        area_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        area_draw  = ImageDraw.Draw(area_layer)

        for i, area in enumerate(areas):
            color = AREA_COLORS[i % len(AREA_COLORS)]
            x1 = int(area["x1"] / 100 * W)
            y1 = int(area["y1"] / 100 * H)
            x2 = int(area["x2"] / 100 * W)
            y2 = int(area["y2"] / 100 * H)

            # Semi-transparent fill
            fill_color   = (*color, AREA_FILL_ALPHA)
            border_color = (*color, AREA_BORDER_ALPHA)

            area_draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=border_color,
                                 width=AREA_BORDER_WIDTH)

            # Area label in top-left corner of box
            label      = area.get("name", f"Area {area.get('id', i+1)}")[:24]
            font_size  = max(12, min(18, (x2 - x1) // 8))
            font       = _load_font(font_size)
            pad        = 6
            bbox       = area_draw.textbbox((0, 0), label, font=font)
            tw, th     = bbox[2] - bbox[0], bbox[3] - bbox[1]
            lx, ly     = x1 + pad, y1 + pad
            # Background pill for readability
            area_draw.rectangle(
                [lx - 3, ly - 2, lx + tw + 3, ly + th + 2],
                fill=(*color, 210),
            )
            area_draw.text((lx, ly), label, fill=(255, 255, 255, 255), font=font)

        img = Image.alpha_composite(img, area_layer)

    # ── Draw dots (top layer) ─────────────────────────────────────────────────
    if components:
        dot_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dot_draw  = ImageDraw.Draw(dot_layer)

        dot_r     = max(DOT_RADIUS, W // 50)
        shadow_r  = dot_r + OUTLINE_WIDTH + 2
        accent_r  = dot_r + OUTLINE_WIDTH
        font      = _load_font(max(13, dot_r - 1))

        for comp in components:
            pt  = comp.get("point", {"x": 50, "y": 50})
            num = comp.get("id", 1)
            cx  = int(pt["x"] / 100 * W)
            cy  = int(pt["y"] / 100 * H)

            # Layer 1: dark shadow (outermost)
            dot_draw.ellipse(
                [cx - shadow_r, cy - shadow_r, cx + shadow_r, cy + shadow_r],
                fill=DOT_SHADOW,
            )
            # Layer 2: colored accent ring
            dot_draw.ellipse(
                [cx - accent_r, cy - accent_r, cx + accent_r, cy + accent_r],
                fill=DOT_ACCENT,
            )
            # Layer 3: white fill
            dot_draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=DOT_FILL,
            )
            # Number centered
            text = str(num)
            bbox = dot_draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            dot_draw.text(
                (cx - tw // 2, cy - th // 2 - 1),
                text,
                fill=(30, 30, 30, 255),
                font=font,
            )

        img = Image.alpha_composite(img, dot_layer)

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=93, optimize=True)
    return buf.getvalue()
