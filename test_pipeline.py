#!/usr/bin/env python3
"""
Test the back-view generation pipeline using synthetic shirt images.

Creates a multi-layer TIFF that simulates a front-view sleeveless shirt mockup,
runs the pipeline, and validates the output.
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tifffile

from generate_back_view import (
    classify_layers,
    create_design_placeholder,
    detect_shirt_mask,
    ensure_rgba,
    find_neckline_region,
    generate_back_view,
    load_layered_tiff,
    mirror_horizontal,
    raise_neckline,
)


def create_synthetic_shirt(width: int = 2000, height: int = 2400) -> dict[str, np.ndarray]:
    """Create a synthetic sleeveless shirt mockup with multiple layers.

    Returns a dict of layer_name -> RGBA numpy array.
    """
    layers = {}

    # Layer 0: Background (solid light gray, fully opaque)
    bg = np.full((height, width, 4), [230, 230, 230, 255], dtype=np.uint8)
    layers["Background"] = bg

    # Layer 1: Shirt base (the main fabric shape with a front neckline)
    shirt = np.zeros((height, width, 4), dtype=np.uint8)

    # Draw the shirt body as a filled polygon
    cx, cy = width // 2, height // 2
    shirt_top = int(height * 0.12)
    shirt_bottom = int(height * 0.85)
    shirt_left = int(width * 0.2)
    shirt_right = int(width * 0.8)

    # Shoulder width
    shoulder_left = int(width * 0.15)
    shoulder_right = int(width * 0.85)
    shoulder_y = int(height * 0.15)

    # Armhole curves
    armhole_bottom_y = int(height * 0.38)

    # Front neckline (scoop neck, goes fairly deep)
    neck_top_y = int(height * 0.14)
    neck_bottom_y = int(height * 0.3)  # Deep scoop
    neck_left = int(width * 0.38)
    neck_right = int(width * 0.62)

    # Build the shirt body contour
    # Left shoulder -> neck left -> neck curve -> neck right -> right shoulder
    # -> right armhole -> right body -> bottom -> left body -> left armhole -> close

    pts = []

    # Left shoulder
    pts.append([shoulder_left, shoulder_y])

    # Left side of neck
    pts.append([neck_left, neck_top_y])

    # Front neckline curve (scoop neck)
    for t in np.linspace(0, np.pi, 20):
        x = int(cx + (neck_left - cx) * np.cos(t))
        y = int(neck_top_y + (neck_bottom_y - neck_top_y) * np.sin(t))
        pts.append([x, y])

    # Right side of neck
    pts.append([neck_right, neck_top_y])

    # Right shoulder
    pts.append([shoulder_right, shoulder_y])

    # Right armhole (concave curve inward)
    for t in np.linspace(0, np.pi, 15):
        x = int(shoulder_right - (shoulder_right - shirt_right) * (1 - np.cos(t)) * 0.5)
        y = int(shoulder_y + (armhole_bottom_y - shoulder_y) * t / np.pi)
        pts.append([x, y])

    # Right body
    pts.append([shirt_right, armhole_bottom_y])
    pts.append([int(shirt_right + width * 0.02), shirt_bottom])

    # Bottom
    pts.append([int(shirt_left - width * 0.02), shirt_bottom])

    # Left body
    pts.append([shirt_left, armhole_bottom_y])

    # Left armhole
    for t in np.linspace(np.pi, 0, 15):
        x = int(shoulder_left + (shirt_left - shoulder_left) * (1 - np.cos(t)) * 0.5)
        y = int(shoulder_y + (armhole_bottom_y - shoulder_y) * (np.pi - t) / np.pi)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32)

    # Fill the shirt with a fabric-like color
    # Use separate contiguous arrays for OpenCV compatibility
    fabric_color = [180, 180, 185]
    shirt_rgb = np.ascontiguousarray(shirt[:, :, :3])
    cv2.fillPoly(shirt_rgb, [pts], fabric_color)
    shirt[:, :, :3] = shirt_rgb

    alpha_ch = np.ascontiguousarray(shirt[:, :, 3])
    alpha_3ch = cv2.cvtColor(alpha_ch, cv2.COLOR_GRAY2BGR)
    cv2.fillPoly(alpha_3ch, [pts], [255, 255, 255])
    shirt[:, :, 3] = alpha_3ch[:, :, 0]

    # Add subtle fabric texture via noise
    noise = np.random.normal(0, 3, shirt[:, :, :3].shape).astype(np.int16)
    shirt_rgb = np.clip(shirt[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    shirt[:, :, :3] = shirt_rgb
    # Only apply texture where shirt exists
    shirt[:, :, :3][shirt[:, :, 3] == 0] = 0

    layers["Shirt Base"] = shirt

    # Layer 2: Shadow/highlight effects
    effects = np.zeros((height, width, 4), dtype=np.uint8)

    # Add some shadow along the sides of the shirt
    for i in range(20):
        offset = i * 2
        alpha_val = max(0, 40 - i * 2)
        # Left shadow
        cv2.line(
            effects,
            (shirt_left + offset, armhole_bottom_y),
            (shirt_left + offset - 10, shirt_bottom),
            [0, 0, 0, alpha_val],
            3,
        )
        # Right shadow
        cv2.line(
            effects,
            (shirt_right - offset, armhole_bottom_y),
            (shirt_right - offset + 10, shirt_bottom),
            [0, 0, 0, alpha_val],
            3,
        )

    layers["Shadows"] = effects

    # Layer 3: Design placeholder (mostly transparent, with a faint guide rectangle)
    design = np.zeros((height, width, 4), dtype=np.uint8)
    design_top = int(height * 0.3)
    design_bottom = int(height * 0.65)
    design_left = int(width * 0.3)
    design_right = int(width * 0.7)
    design[design_top:design_bottom, design_left:design_right] = [200, 200, 255, 20]
    # Mask to shirt area
    design[shirt[:, :, 3] == 0] = [0, 0, 0, 0]

    layers["Design Placeholder"] = design

    return layers


def save_synthetic_tiff(layers: dict[str, np.ndarray], path: str) -> None:
    """Save synthetic layers as a multi-page TIFF."""
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for name, data in layers.items():
            rgba = ensure_rgba(data)
            tif.write(
                rgba,
                photometric="rgb",
                extrasamples=[tifffile.EXTRASAMPLE.UNASSALPHA],
                compression="lzw",
                description=name,
            )


def test_ensure_rgba():
    """Test RGBA conversion for various input formats."""
    # Grayscale
    gray = np.zeros((10, 10), dtype=np.uint8)
    result = ensure_rgba(gray)
    assert result.shape == (10, 10, 4), f"Expected (10,10,4), got {result.shape}"

    # RGB
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    result = ensure_rgba(rgb)
    assert result.shape == (10, 10, 4), f"Expected (10,10,4), got {result.shape}"
    assert np.all(result[:, :, 3] == 255), "Alpha should be 255 for RGB input"

    # RGBA passthrough
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    result = ensure_rgba(rgba)
    assert result.shape == (10, 10, 4)

    print("  ensure_rgba: PASSED")


def test_mirror():
    """Test horizontal mirroring."""
    img = np.zeros((10, 20, 4), dtype=np.uint8)
    img[:, 0, 0] = 255  # Red on left edge

    mirrored = mirror_horizontal(img)
    assert np.all(mirrored[:, -1, 0] == 255), "Red should move to right edge"
    assert np.all(mirrored[:, 0, 0] == 0), "Left edge should be black"

    print("  mirror_horizontal: PASSED")


def test_shirt_mask_detection():
    """Test shirt mask detection on synthetic data."""
    layers = create_synthetic_shirt(800, 1000)
    shirt = layers["Shirt Base"]

    mask = detect_shirt_mask(shirt)
    assert mask.shape == (1000, 800), f"Unexpected mask shape: {mask.shape}"

    coverage = np.sum(mask > 0) / mask.size
    assert 0.1 < coverage < 0.9, f"Unexpected coverage: {coverage:.1%}"

    print(f"  detect_shirt_mask: PASSED (coverage={coverage:.1%})")


def test_neckline_detection():
    """Test neckline region detection."""
    layers = create_synthetic_shirt(800, 1000)
    shirt = layers["Shirt Base"]
    mask = detect_shirt_mask(shirt)

    top_y, bottom_y, left_x, right_x = find_neckline_region(mask)
    assert top_y < bottom_y, f"Neckline top ({top_y}) should be above bottom ({bottom_y})"
    assert left_x < right_x, f"Neckline left ({left_x}) should be left of right ({right_x})"
    # Neckline should be in the upper portion
    assert top_y < 1000 * 0.3, f"Neckline top ({top_y}) should be in upper 30% of height"

    print(f"  find_neckline_region: PASSED (y={top_y}-{bottom_y}, x={left_x}-{right_x})")


def test_raise_neckline():
    """Test neckline raising fills in pixels."""
    layers = create_synthetic_shirt(800, 1000)
    shirt = layers["Shirt Base"]
    mask = detect_shirt_mask(shirt)

    original_coverage = np.sum(mask > 0)
    raised = raise_neckline(shirt, mask, fill_ratio=0.55)
    raised_mask = detect_shirt_mask(raised)
    new_coverage = np.sum(raised_mask > 0)

    assert new_coverage >= original_coverage, (
        f"Raised neckline coverage ({new_coverage}) should be >= original ({original_coverage})"
    )

    print(f"  raise_neckline: PASSED (coverage delta: +{new_coverage - original_coverage} px)")


def test_design_placeholder():
    """Test design placeholder creation."""
    layers = create_synthetic_shirt(800, 1000)
    shirt = layers["Shirt Base"]
    mask = detect_shirt_mask(shirt)

    placeholder = create_design_placeholder(mask)
    assert placeholder.shape == (1000, 800, 4)
    has_content = np.sum(placeholder[:, :, 3] > 0)
    assert has_content > 0, "Placeholder should have some non-transparent pixels"

    # Placeholder should be within shirt bounds
    outside_shirt = mask == 0
    assert np.all(placeholder[outside_shirt, 3] == 0), (
        "Placeholder should not extend outside shirt"
    )

    print(f"  create_design_placeholder: PASSED ({has_content} active pixels)")


def test_layer_classification():
    """Test automatic layer classification."""
    layers = create_synthetic_shirt(400, 500)
    layer_list = [
        {"name": name, "data": data, "shape": data.shape, "dtype": data.dtype, "page_index": i}
        for i, (name, data) in enumerate(layers.items())
    ]

    roles = classify_layers(layer_list)
    assert 0 in roles["background"], "Background should be classified"
    assert 1 in roles["shirt_base"], "Shirt base should be classified"
    assert 2 in roles["effects"], "Shadows should be classified as effects"
    assert 3 in roles["design"], "Design placeholder should be classified as design"

    print("  classify_layers: PASSED")


def test_full_pipeline():
    """End-to-end test: create synthetic TIFF -> run pipeline -> validate output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = str(Path(tmpdir) / "front_view.tif")
        output_path = str(Path(tmpdir) / "back_view.tif")

        # Create and save synthetic mockup
        layers = create_synthetic_shirt(1200, 1500)
        save_synthetic_tiff(layers, input_path)

        input_size = Path(input_path).stat().st_size
        print(f"  Synthetic input: {input_size / 1024:.0f} KB, {len(layers)} layers")

        # Run the pipeline
        generate_back_view(input_path, output_path)

        # Validate output
        assert Path(output_path).exists(), "Output file should exist"
        output_size = Path(output_path).stat().st_size
        assert output_size > 0, "Output file should not be empty"

        # Read back and check structure
        output_layers = load_layered_tiff(output_path)
        # We expect original 4 layers + 1 design placeholder = 5
        assert len(output_layers) >= len(layers), (
            f"Output should have at least {len(layers)} layers, got {len(output_layers)}"
        )

        # Check that the design placeholder layer exists
        has_placeholder = any(
            "placeholder" in l["name"].lower() and "back" in l["name"].lower()
            for l in output_layers
        )
        assert has_placeholder, "Output should contain a back design placeholder layer"

        # Check dimensions match
        for ol in output_layers:
            assert ol["shape"][0] == 1500, f"Height should be 1500, got {ol['shape'][0]}"
            assert ol["shape"][1] == 1200, f"Width should be 1200, got {ol['shape'][1]}"

        print(f"  Full pipeline: PASSED")
        print(f"    Output: {output_size / 1024:.0f} KB, {len(output_layers)} layers")
        for i, ol in enumerate(output_layers):
            print(f"    Layer {i}: '{ol['name']}'")


def main():
    print("=" * 60)
    print("Back-View Generation Pipeline Tests")
    print("=" * 60)

    tests = [
        ("ensure_rgba", test_ensure_rgba),
        ("mirror_horizontal", test_mirror),
        ("detect_shirt_mask", test_shirt_mask_detection),
        ("find_neckline_region", test_neckline_detection),
        ("raise_neckline", test_raise_neckline),
        ("create_design_placeholder", test_design_placeholder),
        ("classify_layers", test_layer_classification),
        ("full_pipeline", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
