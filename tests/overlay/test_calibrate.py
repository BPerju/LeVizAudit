import http.server
from pathlib import Path
from unittest.mock import MagicMock, patch

from vizaudit.overlay.calibrate import _STATIC_HTML, serve_and_open, write_calibration_html


def test_static_html_exists_and_has_camera_access():
    html = _STATIC_HTML.read_text()
    assert "getUserMedia" in html
    assert "<video" in html
    assert "<canvas" in html


def test_write_calibration_html_copies_to_output(tmp_path):
    output_path = tmp_path / "calibration.html"
    result = write_calibration_html(output_path)
    assert result == output_path
    assert output_path.read_text() == _STATIC_HTML.read_text()


def test_write_calibration_html_creates_parent_dirs(tmp_path):
    output_path = tmp_path / "nested" / "dir" / "calibration.html"
    write_calibration_html(output_path)
    assert output_path.exists()


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
