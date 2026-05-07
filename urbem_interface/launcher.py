"""
UrbEm Interface launcher - starts the desktop UI with PyWebView.
"""

import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import webview
from urbem_interface.ui.backend import app


def main():
    def start_server():
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

    server = threading.Thread(target=start_server)
    server.daemon = True
    server.start()

    time.sleep(0.5)

    window = webview.create_window(
        "UrbEm Interface",
        "http://127.0.0.1:5000",
        width=1200,
        height=800,
        resizable=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
