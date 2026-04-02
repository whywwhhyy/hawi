"""Entry point for `python -m hawi_gui` and `uv run hawi_gui`."""

import warnings
warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue.*")

import argparse
import sys
import tkinter as tk
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hawi_gui",
        description="Hawi GUI (tkinter)",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="",
        help="Model factory name (optional; shows selection dialog if omitted)",
    )
    args = parser.parse_args()

    from hawi.models import model_registry
    from hawi_gui.app import HawiGuiApp
    from hawi_gui.widgets.model_dialog import ModelSelectionDialog

    # Load model configs
    for cfg in [Path.home() / ".hawi" / "models.yaml", Path.cwd() / "models.yaml"]:
        if cfg.exists():
            model_registry.load_config(cfg, quiet=True)

    available = model_registry.list_models()
    if not available:
        print("No model factories found. Please configure ~/.hawi/models.yaml")
        sys.exit(1)

    model_name = args.model_name if args.model_name in available else None

    if model_name is None:
        root = tk.Tk()
        root.withdraw()
        dlg = ModelSelectionDialog(root, available, title="选择模型工厂", modal=True)
        root.wait_window(dlg)
        model_name = dlg.result
        if model_name is None:
            root.destroy()
            sys.exit(0)
        root.deiconify()
    else:
        root = None

    print(f"Using: {model_name}")
    HawiGuiApp(model_name=model_name, root=root).run()


if __name__ == "__main__":
    main()
