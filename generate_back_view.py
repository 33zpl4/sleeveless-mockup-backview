#!/usr/bin/env python3
"""
Generate a back-view mockup from a front-view sleeveless shirt TIFF template.

Reads a layered TIFF (e.g., Yellow Images sleeveless shirt mockup), and produces
a back-view version by:
  1. Mirroring the shirt horizontally
  2. Raising the neckline (back necklines are higher than front)
  3. Removing front-specific details
  4. Creating a clean design placeholder layer for the back
  5. Preserving the original layer structure and quality

Usage:
    python generate_back_view.py input.tif [output.tif]

If output path is omitted, writes to <input_basename>_back_view.tif
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def load_layered_tiff(path: str) -> list[dict]:
    """Load a layered TIFF and return a list of layer dicts.

    Each dict contains:
      - 'name': str (page description or 'Layer N')
      - 'data': np.ndarray (RGBA or RGB)
      - 'shape': tuple
      - 'dtype': numpy dtype
    """
    layers = []
    with tifffile.TiffFile(path) as tif:
        for i, page in enumerate(tif.pages):
            data = page.asarray()
            desc = ""
            if page.description:
                desc = page.description
            elif hasattr(page, "tags") and "ImageDescription" in page.tags:
                desc = page.tags["ImageDescription"].value

            layers.append({
                "name": desc if desc else f"Layer {i}",
                "data": data,
                "shape": data.shape,
                "dtype": data.dtype,
                "page_index": i,
            })
    return layers


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    """Convert an image array to RGBA if it isn't already."""
    if img.ndim == 2:
        # Grayscale -> RGBA
        rgba = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
        return rgba
    if img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=img.dtype)
        return np.concatenate([img, alpha], axis=-1)
    if img.shape[2] == 4:
        return img
    return img


def detect_shirt_mask(img_rgba: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Detect the shirt region using alpha channel and color analysis.

    Returns a binary mask where the shirt is white (255) and background is black (0).
    """
    h, w = img_rgba.shape[:2]

    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
        # If alpha provides good separation, use it
        if alpha.min() < 200 and alpha.max() > 50:
            mask = (alpha > threshold).astype(np.uint8) * 255
            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            return mask

    # Fallback: use color-based detection (find the largest non-background region)
    gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Sample corners to estimate background color
    corner_size = max(h, w) // 20
    corners = [
        gray[:corner_size, :corner_size],
        gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size],
        gray[-corner_size:, -corner_size:],
    ]
    bg_value = int(np.median(np.concatenate([c.flatten() for c in corners])))

    diff = np.abs(gray.astype(np.int16) - bg_value)
    mask = (diff > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def find_neckline_region(shirt_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the neckline region (top-center concavity) in the shirt mask.

    Returns (top_y, bottom_y, left_x, right_x) bounding the neckline opening.
    """
    h, w = shirt_mask.shape[:2]

    # Find the topmost shirt pixels across columns
    top_profile = np.full(w, h, dtype=np.int32)
    for x in range(w):
        col = shirt_mask[:, x]
        nonzero = np.nonzero(col)[0]
        if len(nonzero) > 0:
            top_profile[x] = nonzero[0]

    # The shirt occupies some horizontal range
    shirt_cols = np.where(top_profile < h)[0]
    if len(shirt_cols) == 0:
        return (0, h // 4, w // 4, 3 * w // 4)

    shirt_left = shirt_cols[0]
    shirt_right = shirt_cols[-1]
    shirt_width = shirt_right - shirt_left

    # Focus on the center third for the neckline
    center_left = shirt_left + shirt_width // 3
    center_right = shirt_right - shirt_width // 3
    center_profile = top_profile[center_left:center_right]

    if len(center_profile) == 0:
        return (0, h // 4, w // 4, 3 * w // 4)

    # The neckline is the concave dip in the top profile
    # Find where the profile dips below the shoulder line
    shoulder_y = np.min(top_profile[shirt_cols])
    neckline_bottom = np.max(center_profile)

    # Find the lateral extent of the neckline opening
    neck_threshold = shoulder_y + (neckline_bottom - shoulder_y) * 0.3
    neck_cols = np.where(top_profile > neck_threshold)[0]
    # Filter to center region only
    neck_cols = neck_cols[
        (neck_cols >= center_left - shirt_width // 6)
        & (neck_cols <= center_right + shirt_width // 6)
    ]

    if len(neck_cols) == 0:
        neck_left = center_left
        neck_right = center_right
    else:
        neck_left = neck_cols[0]
        neck_right = neck_cols[-1]

    return (int(shoulder_y), int(neckline_bottom), int(neck_left), int(neck_right))


def raise_neckline(
    img_rgba: np.ndarray,
    shirt_mask: np.ndarray,
    fill_ratio: float = 0.55,
) -> np.ndarray:
    """Raise the neckline to approximate a back-view neckline.

    The back neckline is typically higher (less deep) than the front.
    We fill in the lower portion of the neckline opening with fabric color,
    matched from surrounding shirt pixels.

    Args:
        img_rgba: The RGBA image to modify.
        shirt_mask: Binary mask of the shirt region.
        fill_ratio: How much of the neckline depth to fill (0.0 = none, 1.0 = all).

    Returns:
        Modified RGBA image with raised neckline.
    """
    result = img_rgba.copy()
    h, w = result.shape[:2]
    top_y, bottom_y, left_x, right_x = find_neckline_region(shirt_mask)

    neck_depth = bottom_y - top_y
    if neck_depth < 5:
        return result  # No significant neckline to adjust

    # The new neckline will be higher by fill_ratio of the depth
    fill_height = int(neck_depth * fill_ratio)
    new_bottom_y = bottom_y - fill_height

    # Sample fabric color from the shirt area just beside/below the neckline
    sample_margin = max(10, (right_x - left_x) // 8)
    sample_regions = []

    # Sample from left side of neckline
    sl = max(0, left_x - sample_margin * 2)
    sr = max(0, left_x - sample_margin // 2)
    st = max(0, top_y)
    sb = min(h, bottom_y)
    if sr > sl and sb > st:
        region = result[st:sb, sl:sr]
        mask_region = shirt_mask[st:sb, sl:sr]
        valid = region[mask_region > 0]
        if len(valid) > 0:
            sample_regions.append(valid)

    # Sample from right side of neckline
    sl2 = min(w, right_x + sample_margin // 2)
    sr2 = min(w, right_x + sample_margin * 2)
    if sr2 > sl2 and sb > st:
        region = result[st:sb, sl2:sr2]
        mask_region = shirt_mask[st:sb, sl2:sr2]
        valid = region[mask_region > 0]
        if len(valid) > 0:
            sample_regions.append(valid)

    # Sample from below the neckline
    bl_t = min(h, bottom_y)
    bl_b = min(h, bottom_y + sample_margin * 2)
    bl_l = left_x
    bl_r = right_x
    if bl_b > bl_t and bl_r > bl_l:
        region = result[bl_t:bl_b, bl_l:bl_r]
        mask_region = shirt_mask[bl_t:bl_b, bl_l:bl_r]
        valid = region[mask_region > 0]
        if len(valid) > 0:
            sample_regions.append(valid)

    if not sample_regions:
        fabric_color = np.array([200, 200, 200, 255], dtype=result.dtype)
    else:
        all_samples = np.concatenate(sample_regions, axis=0)
        fabric_color = np.median(all_samples, axis=0).astype(result.dtype)

    # Create a fill mask for the neckline area we want to raise
    neck_center_x = (left_x + right_x) // 2
    neck_width = right_x - left_x

    for y in range(new_bottom_y, bottom_y):
        # Create an elliptical fill that's wider at the bottom, narrower at top
        progress = (y - new_bottom_y) / max(1, (bottom_y - new_bottom_y))
        # The fill width narrows as we go up (toward the new neckline)
        # Use a curve that creates a natural neckline shape
        width_factor = progress**0.6
        half_width = int(neck_width * 0.45 * width_factor)

        for x in range(neck_center_x - half_width, neck_center_x + half_width):
            if 0 <= x < w and shirt_mask[y, x] == 0:
                # Only fill if this pixel is not already part of the shirt
                result[y, x] = fabric_color

    # Smooth the transition zone with Gaussian blur on the filled area
    fill_zone_top = max(0, new_bottom_y - 5)
    fill_zone_bottom = min(h, bottom_y + 5)
    fill_zone_left = max(0, left_x - 10)
    fill_zone_right = min(w, right_x + 10)

    zone = result[fill_zone_top:fill_zone_bottom, fill_zone_left:fill_zone_right].copy()
    for ch in range(min(3, zone.shape[2])):
        zone[:, :, ch] = gaussian_filter(zone[:, :, ch].astype(np.float64), sigma=2.0).astype(
            zone.dtype
        )
    # Blend smoothed zone back, only where we filled
    for y in range(fill_zone_top, fill_zone_bottom):
        for x in range(fill_zone_left, fill_zone_right):
            if new_bottom_y <= y < bottom_y:
                ly = y - fill_zone_top
                lx = x - fill_zone_left
                if 0 <= ly < zone.shape[0] and 0 <= lx < zone.shape[1]:
                    # Blend factor: stronger smoothing at the edges of the fill
                    result[y, x, :3] = zone[ly, lx, :3]
                    result[y, x, 3] = 255  # Fully opaque in filled area

    return result


def mirror_horizontal(img: np.ndarray) -> np.ndarray:
    """Mirror image horizontally (left-right flip)."""
    return np.fliplr(img).copy()


def create_design_placeholder(
    shirt_mask: np.ndarray,
    shrink_factor: float = 0.3,
) -> np.ndarray:
    """Create a transparent design placeholder layer sized for the shirt back.

    The placeholder is a rectangular region centered on the shirt's torso area,
    inset from the edges by shrink_factor.

    Returns an RGBA image with the placeholder area marked at low opacity.
    """
    h, w = shirt_mask.shape[:2]
    placeholder = np.zeros((h, w, 4), dtype=np.uint8)

    # Find the shirt bounding box
    rows = np.any(shirt_mask > 0, axis=1)
    cols = np.any(shirt_mask > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return placeholder

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    shirt_h = rmax - rmin
    shirt_w = cmax - cmin

    # The design area is a centered rectangle inset from the shirt edges
    margin_top = int(shirt_h * (shrink_factor + 0.05))  # Extra top margin for neckline
    margin_bottom = int(shirt_h * shrink_factor)
    margin_left = int(shirt_w * shrink_factor)
    margin_right = int(shirt_w * shrink_factor)

    design_top = rmin + margin_top
    design_bottom = rmax - margin_bottom
    design_left = cmin + margin_left
    design_right = cmax - margin_right

    if design_bottom <= design_top or design_right <= design_left:
        return placeholder

    # Mark the design area with a semi-transparent indicator
    # This is a guide layer — actual designs get composited here
    placeholder[design_top:design_bottom, design_left:design_right] = [128, 128, 255, 30]

    # Mask out any pixels outside the shirt
    outside_shirt = shirt_mask == 0
    placeholder[outside_shirt] = [0, 0, 0, 0]

    return placeholder


def classify_layers(layers: list[dict]) -> dict[str, list[int]]:
    """Attempt to classify layers by their role in the mockup.

    Returns a dict mapping role names to lists of layer indices:
      - 'background': Opaque full-coverage layers (likely background)
      - 'shirt_base': Main shirt fabric layer(s)
      - 'effects': Shadow/highlight overlay layers
      - 'design': Design placeholder / smart object layers
      - 'unknown': Unclassified layers
    """
    classified = {
        "background": [],
        "shirt_base": [],
        "effects": [],
        "design": [],
        "unknown": [],
    }

    for i, layer in enumerate(layers):
        data = layer["data"]
        name_lower = layer["name"].lower()

        # Check name-based hints
        if any(kw in name_lower for kw in ["background", "bg", "backdrop"]):
            classified["background"].append(i)
            continue
        if any(kw in name_lower for kw in ["shadow", "highlight", "light", "shade", "glare"]):
            classified["effects"].append(i)
            continue
        if any(kw in name_lower for kw in ["design", "smart", "placeholder", "artwork", "print"]):
            classified["design"].append(i)
            continue
        if any(kw in name_lower for kw in ["shirt", "fabric", "base", "body", "garment"]):
            classified["shirt_base"].append(i)
            continue

        # Heuristic classification based on pixel analysis
        rgba = ensure_rgba(data)
        alpha = rgba[:, :, 3]
        h, w = alpha.shape

        total_pixels = h * w
        opaque_pixels = np.sum(alpha > 250)
        transparent_pixels = np.sum(alpha < 5)
        semi_transparent = total_pixels - opaque_pixels - transparent_pixels

        opaque_ratio = opaque_pixels / total_pixels
        transparent_ratio = transparent_pixels / total_pixels

        if opaque_ratio > 0.95:
            # Mostly opaque — likely background or base shirt layer
            # Check if it has uniform color (background) or varied (shirt)
            rgb_std = np.std(rgba[:, :, :3].astype(np.float64))
            if rgb_std < 15:
                classified["background"].append(i)
            else:
                classified["shirt_base"].append(i)
        elif transparent_ratio > 0.6 and semi_transparent / total_pixels > 0.05:
            # Mostly transparent with some semi-transparent — likely effects
            classified["effects"].append(i)
        elif transparent_ratio > 0.7:
            # Mostly transparent — could be design placeholder
            classified["design"].append(i)
        else:
            classified["unknown"].append(i)

    return classified


def process_layer_for_back_view(
    layer_data: np.ndarray,
    role: str,
    shirt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Process a single layer for back-view conversion.

    Args:
        layer_data: The layer's pixel data.
        role: The classified role of this layer.
        shirt_mask: Binary mask of the shirt region (for neckline work).

    Returns:
        Processed layer data.
    """
    rgba = ensure_rgba(layer_data)

    if role == "background":
        # Background stays as-is (or mirror for consistency)
        return layer_data

    if role == "design":
        # Clear the front design — back gets a fresh placeholder
        cleared = rgba.copy()
        cleared[:, :, 3] = 0  # Make fully transparent
        return cleared

    # Mirror for shirt_base, effects, and unknown layers
    result = mirror_horizontal(rgba)

    if role == "shirt_base" and shirt_mask is not None:
        result = raise_neckline(result, shirt_mask, fill_ratio=0.55)

    return result


def generate_back_view(input_path: str, output_path: str) -> None:
    """Main pipeline: load front-view TIFF, generate back-view, save."""
    print(f"Loading: {input_path}")
    layers = load_layered_tiff(input_path)
    print(f"Found {len(layers)} layer(s)")

    for i, layer in enumerate(layers):
        print(f"  Layer {i}: '{layer['name']}' shape={layer['shape']} dtype={layer['dtype']}")

    # Classify layers
    roles = classify_layers(layers)
    print("\nLayer classification:")
    for role, indices in roles.items():
        if indices:
            names = [layers[i]["name"] for i in indices]
            print(f"  {role}: {names}")

    # Build a role lookup: index -> role
    role_map = {}
    for role, indices in roles.items():
        for idx in indices:
            role_map[idx] = role

    # Detect shirt mask from the primary shirt layer or composite
    shirt_layer_indices = roles["shirt_base"] + roles["unknown"]
    if shirt_layer_indices:
        primary_idx = shirt_layer_indices[0]
        primary_rgba = ensure_rgba(layers[primary_idx]["data"])
    else:
        # Fallback: use the first layer
        primary_rgba = ensure_rgba(layers[0]["data"])
        role_map.setdefault(0, "shirt_base")

    shirt_mask = detect_shirt_mask(primary_rgba)
    print(f"\nShirt mask coverage: {np.sum(shirt_mask > 0) / shirt_mask.size:.1%}")

    # Process each layer
    processed_layers = []
    for i, layer in enumerate(layers):
        role = role_map.get(i, "unknown")
        print(f"Processing layer {i} ('{layer['name']}') as '{role}'...")
        processed = process_layer_for_back_view(layer["data"], role, shirt_mask)
        processed_layers.append(processed)

    # Create the design placeholder layer
    mirrored_mask = mirror_horizontal(shirt_mask)
    # Re-detect from the processed shirt to account for raised neckline
    if shirt_layer_indices:
        processed_shirt = ensure_rgba(processed_layers[shirt_layer_indices[0]])
        final_mask = detect_shirt_mask(processed_shirt)
    else:
        final_mask = mirrored_mask

    design_placeholder = create_design_placeholder(final_mask)
    print(f"\nDesign placeholder created: {design_placeholder.shape}")

    # Assemble output layers with names
    # Insert design placeholder after the shirt base layer
    output_entries: list[tuple[str, np.ndarray]] = []
    design_inserted = False

    for i, proc in enumerate(processed_layers):
        role = role_map.get(i, "unknown")
        original_name = layers[i]["name"]

        if role == "background":
            name = original_name
        elif role == "design":
            # Original front design layer is cleared; skip it in favor of new placeholder
            name = f"{original_name} (cleared)"
        else:
            name = f"{original_name} (back view)"

        output_entries.append((name, ensure_rgba(proc)))

        if role == "shirt_base" and not design_inserted:
            output_entries.append(("Design Placeholder (back)", design_placeholder))
            design_inserted = True

    if not design_inserted:
        # Insert before the last layer if no shirt_base was found
        insert_pos = max(0, len(output_entries) - 1)
        output_entries.insert(insert_pos, ("Design Placeholder (back)", design_placeholder))

    # Write the layered TIFF
    print(f"\nWriting {len(output_entries)} layers to: {output_path}")
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif_out:
        for name, layer_data in output_entries:
            tif_out.write(
                layer_data,
                photometric="rgb",
                extrasamples=[tifffile.EXTRASAMPLE.UNASSALPHA],
                compression="lzw",
                description=name,
                metadata={"Name": name},
            )

    output_size = Path(output_path).stat().st_size
    print(f"\nDone! Output size: {output_size / (1024 * 1024):.1f} MB")
    print(f"Layers written: {len(output_entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a back-view mockup from a front-view sleeveless shirt TIFF.",
    )
    parser.add_argument("input", help="Path to the input front-view layered TIFF file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path for the output back-view TIFF (default: <input>_back_view.tif)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_back_view")

    generate_back_view(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
