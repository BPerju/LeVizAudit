"""Standalone camera-calibration tool -- connects directly to a camera in the browser.

No dataset, no recording session, no Rerun connection: calibration has to be possible
*before* you've recorded anything, so it can't depend on a dataset already existing. The
HTML page (`static/calibrate.html`) uses the browser's own `getUserMedia()` camera API to
show a live feed and let the operator capture a frame and click on it -- this module's only
job is to hand the operator that self-contained file. Editing the resulting config and
running the overlay against it are separate, later steps (`vizaudit-overlay --config ...`).
"""

from __future__ import annotations

import argparse
import http.server
import logging
import shutil
import webbrowser
from pathlib import Path

_STATIC_HTML = Path(__file__).parent / "static" / "calibrate.html"

# Shared convention with `overlay/cli.py`: vizaudit-calibrate's "Save to file..." button
# writes here by default, and vizaudit-overlay's --config reads from here by default -- so
# running the two tools in sequence from the same directory needs no --config/path
# bookkeeping at all. Just a filename (resolved relative to cwd), not an absolute path, so
# "the repo" in practice means "wherever you ran these commands from."
DEFAULT_SAVE_PATH = "vizaudit_calibration.yaml"


def write_calibration_html(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_STATIC_HTML, output_path)
    return output_path


def _make_handler(directory: str, save_to: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=directory, **kwargs)  # type: ignore[arg-type]

        def do_POST(self) -> None:  # noqa: N802 (http.server's naming convention)
            if self.path != "/save":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            save_to.write_bytes(body)
            logging.getLogger(__name__).info("Saved calibration config to %s", save_to)
            response = f"Saved to {save_to}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return Handler


def serve_and_open(output_path: Path, save_to: Path) -> None:
    """Serves `output_path`'s directory on localhost and opens it in the system's default
    browser, then blocks until Ctrl-C.

    `getUserMedia()` requires a "secure context" -- some browsers refuse camera access on
    plain `file://` pages (and a VS Code/editor HTML preview pane typically blocks camera
    access outright, regardless of origin). `http://127.0.0.1` is universally treated as
    secure, and `webbrowser.open()` launches the real system browser rather than whatever
    preview pane might otherwise be showing the file -- fixing both at once.

    Also handles `POST /save` (the page's "Save to file..." button) by writing the request
    body to `save_to` directly -- this is what makes saving land in a predictable, reusable
    location instead of the browser's generic downloads folder, with no save dialog.
    """
    handler = _make_handler(str(output_path.parent), save_to)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_port}/{output_path.name}"
    logging.getLogger(__name__).info(
        "Serving %s at %s (Ctrl-C to stop) -- 'Save to file...' writes to %s", output_path, url, save_to
    )
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Writes the standalone camera-calibration HTML page to --output, serves "
        "it on localhost, and opens it in the default browser. Connect a camera and pick "
        "pixel coordinates -- no other vizaudit process needs to be running."
    )
    parser.add_argument("--output", default="calibration.html", help="Output HTML path.")
    parser.add_argument(
        "--save-to",
        default=DEFAULT_SAVE_PATH,
        help=f"Where the page's 'Save to file...' button writes the config snippet "
        f"(default: {DEFAULT_SAVE_PATH!r} -- the same path vizaudit-overlay's --config "
        f"defaults to, so the two tools chain together with no extra flags).",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Just write the file; don't start a local server or open a browser (the "
        "page's 'Save to file...' button falls back to a native save dialog or a plain "
        "browser download in this mode, since there's no server to POST to).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    output_path = write_calibration_html(Path(args.output))
    if args.no_serve:
        logging.getLogger(__name__).info(
            "Wrote calibration page to %s -- open it in a browser to connect a camera and "
            "pick coordinates. Some browsers block camera access on file:// pages; rerun "
            "without --no-serve if so.",
            output_path,
        )
        return
    serve_and_open(output_path, Path(args.save_to))


if __name__ == "__main__":
    main()
