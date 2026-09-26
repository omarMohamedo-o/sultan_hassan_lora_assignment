"""System and console utilities."""

import sys


def configure_utf8_streams() -> None:
    """Safely configure stdout and stderr to use UTF-8 on Windows consoles."""
    if sys.platform == "win32":
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name, None)
            reconfig = getattr(stream, "reconfigure", None)
            if callable(reconfig):
                try:
                    reconfig(encoding="utf-8")
                except Exception:
                    pass
