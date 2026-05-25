"""
Launch the UrbEm downscaling desktop UI (PyWebView + Flask).
"""

import sys
import threading
import time
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import webview
from UrbEm_Visualizer.ui.backend import app

HOST = "127.0.0.1"
PORT = 5010


def main():
    def start_server():
        app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(0.5)

    webview.create_window(
        "UrbEm Downscaling Tool",
        f"http://{HOST}:{PORT}",
        width=1200,
        height=820,
        resizable=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
