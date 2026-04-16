"""Entry point for `python -m hawi_gui` and `uv run hawi_gui`."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue.*")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hawi_gui",
        description="Hawi GUI (PyQt6)",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="",
        help="Model factory name (optional; shows selection dialog if omitted)",
    )
    args = parser.parse_args()

    try:
        from PyQt6.QtWidgets import QApplication, QDialog
    except ImportError:
        print("PyQt6 is not installed. Install with: uv sync --extra gui")
        sys.exit(1)

    from hawi.models import model_registry
    from hawi_gui.app import HawiGuiApp, ModelSwitchDialog

    # Load model configs
    for cfg in [
        Path.home() / ".hawi" / "models.yaml",
        Path.cwd() / ".hawi" / "models.yaml",
        Path.cwd() / "models.yaml",
    ]:
        if cfg.exists():
            model_registry.load_config(cfg, quiet=True)

    available = model_registry.list_models()
    if not available:
        print("No model factories found. Please configure ~/.hawi/models.yaml")
        sys.exit(1)

    app = QApplication(sys.argv)
    model_name = args.model_name if args.model_name in available else ""
    if not model_name:
        dlg = ModelSwitchDialog(
            models=available,
            current_model=available[0],
            parent=None,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        selected = str(dlg.selected_model or "").strip()
        if not selected:
            sys.exit(0)
        model_name = selected

    window = HawiGuiApp(model_name=model_name)
    window.run()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
