"""Entry point: python -m hawi_gui [model_name]"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue.*")

import tkinter as tk
from pathlib import Path

from hawi.models import model_registry
from hawi_gui.app import HawiGuiApp
from hawi_gui.widgets.model_dialog import ModelSelectionDialog

# Load models
user_config = Path.home() / ".hawi" / "models.yaml"
project_config = Path.cwd() / "models.yaml"
if user_config.exists():
    model_registry.load_config(user_config, quiet=True)
if project_config.exists():
    model_registry.load_config(project_config, quiet=True)

available = model_registry.list_models()
if not available:
    print("No model factories found. Please configure ~/.hawi/models.yaml")
    sys.exit(1)

# Check command-line argument
model_name = None
for arg in sys.argv[1:]:
    if arg in available:
        model_name = arg
        break

# Show selection dialog if no arg
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
app = HawiGuiApp(model_name=model_name, root=root)
app.run()
