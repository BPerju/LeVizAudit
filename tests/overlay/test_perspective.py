import pytest

from vizaudit.overlay.perspective import apply_homography, compute_homography, invert_homography


def test_axis_aligned_corners_behave_like_identity_plus_translation():
    # corners already form a 100x50 axis-aligned rectangle -> canonical dims derive to the
    # same 100x50, so canonical (u, v) should map to pixel (u + 10, v + 20) almost exactly.
    corners = [(10, 20), (110, 20), (110, 70), (10, 70)]
    h = compute_homography(corners)
    x, y = apply_homography(h, (50, 25))
    assert x == pytest.approx(60, abs=1e-6)
    assert y == pytest.approx(45, abs=1e-6)


def test_corners_map_back_to_themselves():
    corners = [(0, 0), (200, 0), (200, 100), (0, 100)]
    h = compute_homography(corners)
    canonical_corners = [(0, 0), (200, 0), (200, 100), (0, 100)]
    for canonical, pixel in zip(canonical_corners, corners):
        result = apply_homography(h, canonical)
        assert result == pytest.approx(pixel, abs=1e-6)


def test_skewed_quadrilateral_center_lands_inside_bounding_box():
    # A genuinely skewed (non-rectangular-looking) quadrilateral, as a tilted camera would
    # produce. The canonical center should still map inside the pixel quadrilateral's
    # bounding box -- a loose but meaningful sanity check without hand-deriving the exact
    # projective center.
    corners = [(50, 50), (250, 60), (240, 200), (40, 190)]
    h = compute_homography(corners)
    width, height = 200.0, 145.0  # roughly matches the corners' own scale
    x, y = apply_homography(h, (width / 2, height / 2))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    assert min(xs) <= x <= max(xs)
    assert min(ys) <= y <= max(ys)


def test_rejects_wrong_number_of_corners():
    with pytest.raises(ValueError):
        compute_homography([(0, 0), (1, 0), (1, 1)])


def test_explicit_aspect_ratio_changes_canonical_scale():
    corners = [(0, 0), (200, 0), (200, 100), (0, 100)]
    h_default = compute_homography(corners)
    h_square = compute_homography(corners, aspect_ratio=1.0)
    # Same corners, different aspect_ratio -> different canonical scale -> a fixed canonical
    # point maps to a different pixel point.
    assert apply_homography(h_default, (50, 50)) != apply_homography(h_square, (50, 50))


def test_invert_homography_round_trips():
    corners = [(50, 50), (250, 60), (240, 200), (40, 190)]
    h = compute_homography(corners)
    inv = invert_homography(h)
    canonical_point = (37.5, 22.0)
    pixel_point = apply_homography(h, canonical_point)
    back = apply_homography(inv, pixel_point)
    assert back == pytest.approx(canonical_point, abs=1e-6)


def test_invert_homography_maps_pixel_corners_back_to_canonical_corners():
    corners = [(0, 0), (200, 0), (200, 100), (0, 100)]
    h = compute_homography(corners)
    inv = invert_homography(h)
    canonical_corners = [(0, 0), (200, 0), (200, 100), (0, 100)]
    for pixel, canonical in zip(corners, canonical_corners):
        assert apply_homography(inv, pixel) == pytest.approx(canonical, abs=1e-6)
