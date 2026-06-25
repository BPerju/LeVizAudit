import http.server
from pathlib import Path
from unittest.mock import MagicMock, patch

from vizaudit.overlay.calibrate import (
    _STATIC_DIR,
    _STATIC_HTML,
    _STATIC_SIBLINGS,
    serve_and_open,
    write_calibration_html,
)


def test_static_html_links_external_css_and_js():
    # calibrate.html itself is just markup now -- styles/logic live in the sibling
    # calibrate.css/calibrate.js files it links by relative path (see _STATIC_SIBLINGS).
    html = _STATIC_HTML.read_text()
    assert '<link rel="stylesheet" href="calibrate.css">' in html
    assert '<script src="calibrate.js"></script>' in html
    assert "<video" in html
    assert "<canvas" in html


def test_static_js_has_camera_access():
    js = (_STATIC_DIR / "calibrate.js").read_text()
    assert "getUserMedia" in js


def test_static_html_labels_radial_distribution_as_beta():
    # "radial" has had many more iterative correctness fixes than "grid"/"random" (see
    # CLAUDE.md) and isn't supported at all yet for combined (union) multi-arm sampling --
    # the dropdown option says so explicitly rather than presenting all three as equally
    # mature/supported.
    html = _STATIC_HTML.read_text()
    assert 'value="radial"' in html
    assert "Radial (beta)" in html


def test_static_html_has_objects_card():
    # The Objects card (named objects, each assigned onto a subset of the sample-point cloud
    # by selecting points with the general "select" tool and then clicking Assign/Unassign --
    # stacking order is derived automatically from assignment order, not a manual field -- plus
    # an episode-preview stepper) -- see CLAUDE.md's Objects-layer notes. A behavioral smoke
    # test of the actual JS logic lives in an ad-hoc Node harness during development (this
    # project's established convention for calibrate.js, per CLAUDE.md), not in this Python
    # suite; this just confirms the markup calibrate.js's getElementById calls depend on
    # actually ships.
    html = _STATIC_HTML.read_text()
    for element_id in [
        "objectChips", "objectColorInput", "objectNameInput",
        "addObjectBtn", "removeObjectBtn", "assignSelectedBtn", "unassignSelectedBtn",
        "previewModeSelect", "episodeStepper", "episodePrevBtn", "episodeNextBtn", "episodeLabel",
    ]:
        assert f'id="{element_id}"' in html, f"missing #{element_id} in calibrate.html"
    assert "objectLevelInput" not in html, "level must be automatic, not a manual input field"


def test_static_html_has_behavioral_randomization_knobs():
    # The behavioral knobs (sequencing/episode_targets/co-location/level strategy) are REAL
    # config fields the live overlay consumes (session.py/pattern.py), not preview-only -- see
    # CLAUDE.md. Animate/play is the one pure-preview addition alongside them.
    html = _STATIC_HTML.read_text()
    for element_id in [
        "objectSequencingSelect", "objectSeedInput", "episodeTargetsSelect",
        "coLocationSelect", "stackLevelStrategySelect", "stackLevelSeedInput",
        "episodePlayBtn", "collisionWarning", "sampleCountLabel",
    ]:
        assert f'id="{element_id}"' in html, f"missing #{element_id} in calibrate.html"
    # "stratified" and "phase" sequencing modes were both removed before ever shipping in a
    # release -- "shuffled" already covers stratified's use case at least as well (a fixed
    # stride risks resonating with the underlying point grid's own periodic structure, which
    # true randomness can't), and "phase" conflated object-pair decorrelation with the
    # separate, scene-level co_location concern (see CLAUDE.md). `jitter_px`/`presence`, two
    # other earlier per-object knobs, were removed after a direct report that they didn't pull
    # their weight against the rest of this feature's complexity.
    assert "stratified" not in html.lower()
    assert "phase offset" not in html.lower()
    assert "objectJitterInput" not in html
    assert "objectPresenceInput" not in html


def test_write_calibration_html_copies_to_output(tmp_path):
    output_path = tmp_path / "calibration.html"
    result = write_calibration_html(output_path)
    assert result == output_path
    assert output_path.read_text() == _STATIC_HTML.read_text()


def test_write_calibration_html_copies_sibling_css_and_js(tmp_path):
    output_path = tmp_path / "calibration.html"
    write_calibration_html(output_path)
    for name in _STATIC_SIBLINGS:
        assert (output_path.parent / name).read_text() == (_STATIC_DIR / name).read_text()


def test_write_calibration_html_creates_parent_dirs(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "calibration.html"
    write_calibration_html(output_path)
    assert output_path.exists()
    for name in _STATIC_SIBLINGS:
        assert (output_path.parent / name).exists()


def test_serve_and_open_opens_browser_to_localhost_url(tmp_path):
    output_path = tmp_path / "calibration.html"
    output_path.write_text("<html></html>")
    save_to = tmp_path / "saved.yaml"

    mock_server = MagicMock()
    mock_server.server_port = 12345

    with (
        patch("vizaudit.overlay.calibrate.http.server.HTTPServer", return_value=mock_server),
        patch("vizaudit.overlay.calibrate.webbrowser.open") as mock_open,
    ):
        serve_and_open(output_path, save_to)

    mock_open.assert_called_once_with("http://127.0.0.1:12345/calibration.html")
    mock_server.serve_forever.assert_called_once()
    mock_server.shutdown.assert_called_once()


def test_serve_and_open_shuts_down_cleanly_on_keyboard_interrupt(tmp_path):
    output_path = tmp_path / "calibration.html"
    output_path.write_text("<html></html>")
    save_to = tmp_path / "saved.yaml"

    mock_server = MagicMock()
    mock_server.server_port = 1
    mock_server.serve_forever.side_effect = KeyboardInterrupt

    with (
        patch("vizaudit.overlay.calibrate.http.server.HTTPServer", return_value=mock_server),
        patch("vizaudit.overlay.calibrate.webbrowser.open"),
    ):
        serve_and_open(output_path, save_to)  # must not raise

    mock_server.shutdown.assert_called_once()


def test_save_endpoint_writes_request_body_to_save_to_path(tmp_path):
    import http.client
    import threading

    from vizaudit.overlay.calibrate import _make_handler

    save_to = tmp_path / "nested" / "vizaudit_calibration.yaml"
    handler = _make_handler(str(tmp_path), save_to)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/save", body=b"surface_calibration:\n  corners: []\n")
        response = conn.getresponse()
        assert response.status == 200
        response.read()
        conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert save_to.read_text() == "surface_calibration:\n  corners: []\n"


def test_save_endpoint_unknown_path_404s():
    import http.client
    import threading

    from vizaudit.overlay.calibrate import _make_handler

    handler = _make_handler(".", Path("/tmp/unused.yaml"))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("POST", "/not-save", body=b"x")
        response = conn.getresponse()
        assert response.status == 404
        response.read()
        conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
