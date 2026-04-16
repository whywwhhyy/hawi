"""PyQt6-based Hawi GUI application."""

from __future__ import annotations

import html
import json
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QColor, QFontMetrics, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .protocol import (
    CmdApplyPlugins,
    CmdClearContext,
    CmdClearQueue,
    CmdEnqueue,
    CmdInterrupt,
    CmdStop,
    CmdSwitchModel,
    PluginConfigs,
    QueueKind,
    UiAgentInterrupt,
    UiDebugInfo,
    UiError,
    UiInterrupt,
    UiModelMetadata,
    UiModelRetry,
    UiPluginsApplied,
    UiReady,
    UiRunStart,
    UiRunStop,
    UiStatusUpdate,
    UiTextDelta,
    UiThinkingDelta,
    UiToolCallStart,
    UiToolCallDelta,
    UiToolCallStop,
    UiToolResult,
)
from .scheduler_bridge import (
    PLUGIN_FILESYSTEM,
    PLUGIN_MCP,
    PLUGIN_PYTHON_INTERPRETER,
    PLUGIN_SHELL,
    PLUGIN_SKILLS,
    PLUGIN_WEB,
    SchedulerThread,
)
from .streaming_json import StreamingJsonState, render_json_tree_html
from .streaming_markdown import MarkdownOpHtmlRenderer, StreamingMarkdownOpParser


HELP_TEXT = (
    "命令: /clear(清空上下文)  /cq(清普通队列)  /chq(清高优队列)  "
    "/cuq(清紧急队列)  /ca(清所有队列)  /quit(退出)"
)


@dataclass
class PluginCatalogItem:
    key: str
    label: str
    schema: dict[str, Any]
    defaults: dict[str, Any]


@dataclass
class ChatNode:
    node_id: str
    kind: str
    html: str


@dataclass
class MarkdownStreamState:
    node_id: str
    parser: StreamingMarkdownOpParser = field(default_factory=StreamingMarkdownOpParser)
    renderer: MarkdownOpHtmlRenderer = field(default_factory=MarkdownOpHtmlRenderer)


@dataclass
class ToolBubbleState:
    node_id: str
    run_id: str
    name: str
    json_state: StreamingJsonState = field(default_factory=StreamingJsonState)
    args_tree: Any = field(default_factory=dict)
    status: str = "running"
    duration_ms: float | None = None
    result_preview: str = ""
    final_arguments: dict[str, Any] | None = None


@dataclass
class RunRenderState:
    agent: MarkdownStreamState | None = None
    thinking: MarkdownStreamState | None = None
    tool_call_ids: set[str] = field(default_factory=set)


class AutoResizeInput(QTextEdit):
    """Input box with dynamic height and Enter-to-send behavior."""

    def __init__(
        self,
        on_submit: Callable[[], None],
        min_lines: int = 2,
        max_lines: int = 8,
        on_height_changed: Callable[[int], None] | None = None,
    ):
        super().__init__()
        self._on_submit = on_submit
        self._min_lines = min_lines
        self._max_lines = max_lines
        self._on_height_changed = on_height_changed
        self.setAcceptRichText(False)
        self.setPlaceholderText("输入消息，Enter 发送，Shift+Enter 换行，Shift+Tab 切换优先级")
        self.textChanged.connect(self._adjust_height)
        self._adjust_height()

    def keyPressEvent(self, e):
        if e is not None:
            if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    super().keyPressEvent(e)
                else:
                    self._on_submit()
                return
        super().keyPressEvent(e)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        if self._on_height_changed:
            self._on_height_changed(self.height())

    def _adjust_height(self):
        metrics = QFontMetrics(self.font())
        line_height = metrics.lineSpacing()
        doc_height = int(self.document().size().height()) + 10
        min_height = line_height * self._min_lines + 14
        max_height = line_height * self._max_lines + 14
        target_height = max(min_height, min(max_height, doc_height))
        if self.height() != target_height:
            self.setFixedHeight(target_height)
            if self._on_height_changed:
                self._on_height_changed(target_height)
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
            if doc_height > max_height
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )


class PluginConfigDialog(QDialog):
    """Dialog to choose plugins and edit plugin configs from JSON schema."""

    def __init__(
        self,
        catalog: list[PluginCatalogItem],
        selected_plugins: list[str],
        plugin_configs: PluginConfigs,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("插件配置")
        self.resize(680, 560)
        self._catalog = catalog
        self._selected = set(selected_plugins)
        self._configs = {k: dict(v) for k, v in plugin_configs.items()}
        self.selected_plugins_result: list[str] = []
        self.plugin_configs_result: PluginConfigs = {}
        self._checkboxes: dict[str, QCheckBox] = {}
        self._fields: dict[str, dict[str, tuple[QWidget, dict[str, Any]]]] = {}
        self._groups: dict[str, QGroupBox] = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        info = QLabel("勾选启用插件；参数会按插件模板保存到 ./.hawi/gui_plugins.json")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(info)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        for item in self._catalog:
            group = QGroupBox(item.label, content)
            group_layout = QVBoxLayout(group)
            group_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

            checkbox = QCheckBox("启用此插件", group)
            checkbox.setChecked(item.key in self._selected)
            self._checkboxes[item.key] = checkbox
            group_layout.addWidget(checkbox)

            form = QWidget(group)
            form_layout = QFormLayout(form)
            form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            form_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            field_map: dict[str, tuple[QWidget, dict[str, Any]]] = {}
            schema_props = item.schema.get("properties", {})
            cfg_values = {**item.defaults, **self._configs.get(item.key, {})}
            for field_name, field_schema in schema_props.items():
                field_widget = self._create_field_widget(field_schema, cfg_values.get(field_name))
                field_widget.setEnabled(checkbox.isChecked())
                label = field_schema.get("title", field_name)
                form_layout.addRow(label, field_widget)
                field_map[field_name] = (field_widget, field_schema)

            self._fields[item.key] = field_map
            group_layout.addWidget(form)
            self._groups[item.key] = group

            checkbox.toggled.connect(
                lambda enabled, plugin_key=item.key: self._set_plugin_fields_enabled(plugin_key, enabled)
            )
            content_layout.addWidget(group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _create_field_widget(self, schema: dict[str, Any], value: Any) -> QWidget:
        field_type = schema.get("type", "string")
        default = schema.get("default")
        if value is None:
            value = default

        if field_type == "boolean":
            widget = QCheckBox(self)
            widget.setChecked(bool(value))
            return widget

        if field_type == "integer":
            widget = QSpinBox(self)
            widget.setRange(-1_000_000_000, 1_000_000_000)
            widget.setValue(int(value or 0))
            return widget

        if field_type == "number":
            widget = QDoubleSpinBox(self)
            widget.setRange(-1_000_000_000.0, 1_000_000_000.0)
            widget.setDecimals(6)
            widget.setValue(float(value or 0.0))
            return widget

        widget = QLineEdit(self)
        if value is not None:
            widget.setText(str(value))
        return widget

    def _set_plugin_fields_enabled(self, plugin_key: str, enabled: bool):
        for widget, _schema in self._fields.get(plugin_key, {}).values():
            widget.setEnabled(enabled)

    def _field_value(self, widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def _on_accept(self):
        selected: list[str] = []
        configs: PluginConfigs = {}

        for item in self._catalog:
            enabled = self._checkboxes[item.key].isChecked()
            if not enabled:
                continue

            selected.append(item.key)
            field_values: dict[str, Any] = {}
            for field_name, (widget, _field_schema) in self._fields.get(item.key, {}).items():
                field_values[field_name] = self._field_value(widget)

            required_fields = item.schema.get("required", [])
            for req in required_fields:
                req_value = field_values.get(req)
                if req_value is None or (isinstance(req_value, str) and not req_value.strip()):
                    QMessageBox.warning(
                        self,
                        "参数错误",
                        f"{item.label}: 参数 '{req}' 为必填项。",
                    )
                    return
            configs[item.key] = field_values

        self.selected_plugins_result = selected
        self.plugin_configs_result = configs
        self.accept()


class ModelSwitchDialog(QDialog):
    """Large dialog to choose model using provider/model two-column layout."""

    def __init__(
        self,
        models: list[str],
        current_model: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("切换模型")
        self.resize(900, 580)
        self._models = list(models)
        self._current_model = current_model
        self._provider_models: dict[str, list[str]] = {}
        for model in self._models:
            provider, _model_id = self._split_model_name(model)
            self._provider_models.setdefault(provider, []).append(model)
        self._provider_order = list(self._provider_models.keys())
        self.selected_model: str | None = None
        self._build()
        self._reload_provider_list()
        # Default focus goes to the left provider list.
        QTimer.singleShot(0, self._focus_provider_list)

    def _build(self):
        root = QVBoxLayout(self)

        info = QLabel("左侧选择 Provider，右侧选择模型。支持筛选，双击可直接确认。")
        info.setWordWrap(True)
        root.addWidget(info)

        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText("筛选模型（输入 provider 或 model_id）")
        self._filter_edit.textChanged.connect(self._reload_provider_list)
        root.addWidget(self._filter_edit)

        body = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        left.addWidget(QLabel("Provider"))
        right.addWidget(QLabel("Model"))

        self._provider_list = QListWidget(self)
        self._provider_list.installEventFilter(self)
        self._provider_list.currentRowChanged.connect(
            lambda _row: self._reload_model_list()
        )
        left.addWidget(self._provider_list, stretch=1)

        self._model_list = QListWidget(self)
        self._model_list.installEventFilter(self)
        self._model_list.itemDoubleClicked.connect(lambda _item: self._on_accept())
        self._model_list.currentRowChanged.connect(lambda _row: self._update_ok_state())
        right.addWidget(self._model_list, stretch=1)

        body.addLayout(left, stretch=1)
        body.addLayout(right, stretch=2)
        root.addLayout(body, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        root.addWidget(buttons)

    @staticmethod
    def _split_model_name(model: str) -> tuple[str, str]:
        if "/" in model:
            provider, model_id = model.split("/", 1)
            return provider, model_id
        return "default", model

    def _reload_provider_list(self):
        keyword = self._filter_edit.text().strip().lower()
        current_selected = (
            self._provider_list.currentItem().text()
            if self._provider_list.currentItem()
            else ""
        )

        self._provider_list.clear()
        for provider in self._provider_order:
            models = self._provider_models.get(provider, [])
            if keyword and not any(keyword in model.lower() for model in models):
                continue
            self._provider_list.addItem(provider)

        default_provider, _ = self._split_model_name(self._current_model)
        target_provider = current_selected or default_provider
        if target_provider:
            matched = self._provider_list.findItems(
                target_provider, Qt.MatchFlag.MatchExactly
            )
            if matched:
                self._provider_list.setCurrentItem(matched[0])

        if self._provider_list.currentItem() is None and self._provider_list.count() > 0:
            self._provider_list.setCurrentRow(0)
        self._reload_model_list()

    def _reload_model_list(self):
        keyword = self._filter_edit.text().strip().lower()
        current_selected = (
            self._model_list.currentItem().text() if self._model_list.currentItem() else ""
        )
        provider_item = self._provider_list.currentItem()

        self._model_list.clear()
        if provider_item is None:
            self._update_ok_state()
            return

        provider = provider_item.text()
        for model in self._provider_models.get(provider, []):
            if keyword and keyword not in model.lower():
                continue
            self._model_list.addItem(model)

        current_provider, _ = self._split_model_name(self._current_model)
        target_model = (
            current_selected
            or (self._current_model if provider == current_provider else "")
        )
        if target_model:
            matched = self._model_list.findItems(target_model, Qt.MatchFlag.MatchExactly)
            if matched:
                self._model_list.setCurrentItem(matched[0])

        if self._model_list.currentItem() is None and self._model_list.count() > 0:
            self._model_list.setCurrentRow(0)
        self._update_ok_state()

    def _update_ok_state(self):
        self._ok_btn.setEnabled(self._model_list.currentItem() is not None)

    def _focus_provider_list(self):
        if self._provider_list.count() > 0 and self._provider_list.currentRow() < 0:
            self._provider_list.setCurrentRow(0)
        self._provider_list.setFocus(Qt.FocusReason.OtherFocusReason)

    def _focus_model_list(self):
        if self._model_list.count() > 0 and self._model_list.currentRow() < 0:
            self._model_list.setCurrentRow(0)
        self._model_list.setFocus(Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, watched: object, event: object) -> bool:
        if (
            watched in (self._provider_list, self._model_list)
            and isinstance(event, QEvent)
            and event.type() == QEvent.Type.KeyPress
        ):
            key_event = event
            if key_event.key() == Qt.Key.Key_Right and watched is self._provider_list:
                self._focus_model_list()
                return True
            if key_event.key() == Qt.Key.Key_Left and watched is self._model_list:
                self._focus_provider_list()
                return True
        return super().eventFilter(watched, event)

    def _on_accept(self):
        current = self._model_list.currentItem()
        if current is None:
            return
        self.selected_model = current.text()
        self.accept()


class HawiGuiApp(QMainWindow):
    """PyQt6 Hawi GUI."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.ui_queue: queue.Queue = queue.Queue()
        self.cmd_queue: queue.Queue = queue.Queue()
        self._sched_thread: SchedulerThread | None = None
        self._scheduler_state = "IDLE"
        self._agent_state = "IDLE"
        self._active_run_id: str | None = None
        self._chat_follow_tail = True
        self._chat_nodes: list[ChatNode] = []
        self._chat_nodes_by_id: dict[str, int] = {}
        self._chat_node_counter = 0
        self._render_scheduled = False
        self._run_render_state: dict[str, RunRenderState] = {}
        self._tool_bubbles: dict[str, ToolBubbleState] = {}
        self._show_debug_info = True

        self._config_path = Path.cwd() / ".hawi" / "gui_plugins.json"
        persisted = self._load_plugin_state()
        self._selected_plugins = persisted["selected_plugins"]
        self._plugin_configs = persisted["plugin_configs"]
        self._catalog = self._build_plugin_catalog()

        self._build_ui()
        self._bind_shortcuts()

    # ─── Setup ────────────────────────────────────────────────────────────────

    def _build_plugin_catalog(self) -> list[PluginCatalogItem]:
        from hawi_plugins.filesystem_plugin import FileSystemPlugin
        from hawi_plugins.mcp_plugin import MCPPlugin
        from hawi_plugins.python_interpreter import PythonInterpreterPlugin
        from hawi_plugins.shell_plugin import ShellPlugin
        from hawi_plugins.skills_plugin import SkillsPlugin
        from hawi_plugins.web import WebPlugin

        entries = [
            (PLUGIN_FILESYSTEM, "FileSystemPlugin", FileSystemPlugin),
            (PLUGIN_SHELL, "ShellPlugin", ShellPlugin),
            (PLUGIN_WEB, "WebPlugin", WebPlugin),
            (PLUGIN_SKILLS, "SkillsPlugin", SkillsPlugin),
            (PLUGIN_PYTHON_INTERPRETER, "PythonInterpreterPlugin", PythonInterpreterPlugin),
            (PLUGIN_MCP, "MCPPlugin", MCPPlugin),
        ]
        catalog: list[PluginCatalogItem] = []
        for key, label, cls in entries:
            catalog.append(
                PluginCatalogItem(
                    key=key,
                    label=label,
                    schema=cls.gui_config_schema(),
                    defaults=cls.gui_default_config(),
                )
            )
        return catalog

    def _load_plugin_state(self) -> dict[str, Any]:
        default = {
            "version": 1,
            "selected_plugins": [],
            "plugin_configs": {},
        }
        if not self._config_path.exists():
            return default

        try:
            raw = json.loads(self._config_path.read_text(encoding="utf-8"))
        except Exception:
            return default

        selected = raw.get("selected_plugins", [])
        plugin_configs = raw.get("plugin_configs", {})
        if not isinstance(selected, list):
            selected = []
        if not isinstance(plugin_configs, dict):
            plugin_configs = {}

        cleaned = {
            "version": 1,
            "selected_plugins": [str(name) for name in selected],
            "plugin_configs": {
                str(name): dict(cfg) for name, cfg in plugin_configs.items() if isinstance(cfg, dict)
            },
        }
        return cleaned

    def _save_plugin_state(self):
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "selected_plugins": list(self._selected_plugins),
            "plugin_configs": {k: dict(v) for k, v in self._plugin_configs.items()},
        }
        self._config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_ui(self):
        self.setWindowTitle(f"Hawi - {self.model_name}")
        self.resize(1100, 760)
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 3, 6, 6)
        root.setSpacing(2)
        self._top_bar_height = min(
            24, max(18, int(QFontMetrics(self.font()).lineSpacing() * 1.5))
        )
        self._top_button_height = max(36, self._top_bar_height + 8)

        # Top bar
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self._status_strip = QWidget(self)
        status_layout = QHBoxLayout(self._status_strip)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(6)

        self._status_cells: dict[int, QLabel] = {}
        for column in (1, 2, 3):
            if column != 1:
                status_layout.addWidget(self._make_status_separator())
            cell = QLabel(self._status_strip)
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_layout.addWidget(cell)
            self._status_cells[column] = cell

        self._status_strip.setFixedHeight(self._top_bar_height)
        self._set_state_cell(1, "Scheduler", "IDLE")
        self._set_state_cell(2, "Agent", "IDLE")
        self._set_status_cell(3, "Queue U/H/N: 0/0/0")
        top.addWidget(self._status_strip, stretch=1)
        top.addWidget(self._make_top_separator(), alignment=Qt.AlignmentFlag.AlignVCenter)

        self._plugin_btn = QPushButton("插件配置")
        self._plugin_btn.clicked.connect(self._open_plugin_dialog)
        self._plugin_btn.setMinimumHeight(self._top_button_height)
        self._plugin_btn.setStyleSheet(self._neutral_button_style(horizontal_padding_px=10))
        top.addWidget(self._plugin_btn)

        self._model_btn = QPushButton("Model: -/-")
        self._model_btn.clicked.connect(self._show_model_switcher)
        self._model_btn.setMinimumHeight(self._top_button_height)
        self._model_btn.setStyleSheet(self._neutral_button_style(horizontal_padding_px=10))
        top.addWidget(self._model_btn)
        self._set_model_header(self.model_name)
        root.addLayout(top)

        # Chat area
        self.chat_view = QTextBrowser(self)
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setReadOnly(True)
        self.chat_view.verticalScrollBar().valueChanged.connect(self._on_chat_scroll)
        root.addWidget(self.chat_view, stretch=1)

        # Bottom controls
        priority_row = QHBoxLayout()
        priority_row.addWidget(QLabel("优先级:"))
        self._priority_group = QButtonGroup(self)
        self._priority_buttons: dict[QueueKind, QPushButton] = {}
        for key, text in (
            ("normal", "普通"),
            ("high_prio", "高优"),
            ("urgent", "紧急"),
        ):
            btn = QPushButton(text)
            btn.setCheckable(True)
            self._priority_group.addButton(btn)
            self._priority_buttons[key] = btn
            priority_row.addWidget(btn)
        priority_row.addSpacing(12)
        priority_row.addWidget(QLabel("上下文:"))
        self._clear_context_btn = QPushButton("清空")
        self._clear_context_btn.clicked.connect(self._cmd_clear_context)
        self._clear_context_btn.setMinimumWidth(72)
        self._clear_context_btn.setAutoDefault(False)
        self._clear_context_btn.setDefault(False)
        priority_row.addWidget(self._clear_context_btn)
        self._priority_buttons["high_prio"].setChecked(True)
        priority_row.addStretch(1)
        root.addLayout(priority_row)

        debug_row = QHBoxLayout()
        debug_row.setContentsMargins(0, 0, 0, 0)
        debug_row.setSpacing(6)
        debug_row.addWidget(QLabel("调试:"))
        self._debug_toggle = QCheckBox("显示 Debug 信息")
        self._debug_toggle.setChecked(self._show_debug_info)
        self._debug_toggle.toggled.connect(self._on_debug_toggle_changed)
        debug_row.addWidget(self._debug_toggle)
        debug_row.addStretch(1)
        root.addLayout(debug_row)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)
        self.input_box = AutoResizeInput(
            on_submit=self._submit_input,
            on_height_changed=self._sync_action_buttons_size,
        )
        input_row.addWidget(self.input_box, stretch=1)
        self._send_btn = QPushButton("发送")
        self._send_btn.clicked.connect(self._submit_input)
        self._send_btn.setMinimumWidth(108)
        self._send_btn.setAutoDefault(False)
        self._send_btn.setDefault(False)
        input_row.addWidget(self._send_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self._request_interrupt)
        self._stop_btn.setMinimumWidth(96)
        self._stop_btn.setAutoDefault(False)
        self._stop_btn.setDefault(False)
        input_row.addWidget(self._stop_btn)
        root.addLayout(input_row)
        self._refresh_action_buttons()
        self._sync_action_buttons_size()

        self._build_menu()

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_input_box)
        QShortcut(QKeySequence("Ctrl+Shift+L"), self, activated=self._cmd_clear_context)
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self._show_model_switcher)
        QShortcut(QKeySequence("Shift+Tab"), self, activated=self._cycle_priority)
        QShortcut(QKeySequence("Esc"), self, activated=self.input_box.clear)

    def _build_menu(self):
        menu = self.menuBar()
        model_menu = menu.addMenu("模型")
        switch_action = QAction("切换模型", self)
        switch_action.triggered.connect(self._show_model_switcher)
        model_menu.addAction(switch_action)

        plugin_menu = menu.addMenu("插件")
        plugin_action = QAction("插件配置", self)
        plugin_action.triggered.connect(self._open_plugin_dialog)
        plugin_menu.addAction(plugin_action)

        help_menu = menu.addMenu("帮助")
        help_action = QAction("显示帮助", self)
        help_action.triggered.connect(lambda: self._append_system(HELP_TEXT))
        help_menu.addAction(help_action)

    # ─── Run / lifecycle ──────────────────────────────────────────────────────

    def run(self):
        self._sched_thread = SchedulerThread(
            ui_queue=self.ui_queue,
            cmd_queue=self.cmd_queue,
            model_name=self.model_name,
            selected_plugins=list(self._selected_plugins),
            plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
        )
        self._sched_thread.start()
        self._append_system("正在启动 Scheduler ...")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_ui_queue)
        self._timer.start(50)
        self.show()
        QTimer.singleShot(0, self._sync_action_buttons_size)
        QTimer.singleShot(0, self._focus_input_box)

    def closeEvent(self, event: QCloseEvent):  # noqa: N802
        if self._sched_thread:
            self.cmd_queue.put(CmdStop())
            self._sched_thread.join(timeout=2.0)
        super().closeEvent(event)

    # ─── UI helpers ───────────────────────────────────────────────────────────

    def _append_system(self, text: str):
        self._append_meta(f"[系统] {text}")

    def _append_error(self, text: str):
        self._append_meta(f"[错误] {text}")

    def _append_debug(self, text: str):
        debug_text = f"🔎 [DEBUG] {text}"
        self._append_chat_node("debug", self._as_meta_block(debug_text))

    def _append_meta(self, text: str):
        self._append_chat_node("meta", self._as_meta_block(text))

    def _next_node_id(self, prefix: str) -> str:
        self._chat_node_counter += 1
        return f"{prefix}_{self._chat_node_counter}"

    def _append_chat_node(self, kind: str, html_fragment: str) -> str:
        node_id = self._next_node_id(kind)
        self._chat_nodes_by_id[node_id] = len(self._chat_nodes)
        self._chat_nodes.append(ChatNode(node_id=node_id, kind=kind, html=html_fragment))
        self._queue_render()
        return node_id

    def _update_chat_node(self, node_id: str, html_fragment: str):
        idx = self._chat_nodes_by_id.get(node_id)
        if idx is None or idx >= len(self._chat_nodes):
            self._append_chat_node("dynamic", html_fragment)
            return
        self._chat_nodes[idx].html = html_fragment
        self._queue_render()

    def _on_chat_scroll(self, _value: int):
        bar = self.chat_view.verticalScrollBar()
        self._chat_follow_tail = (bar.maximum() - bar.value()) <= 2

    def _on_debug_toggle_changed(self, checked: bool):
        self._show_debug_info = checked
        self._queue_render()

    def _queue_render(self):
        if self._render_scheduled:
            return
        self._render_scheduled = True
        QTimer.singleShot(16, self._render_chat)

    def _render_chat(self):
        self._render_scheduled = False
        bar = self.chat_view.verticalScrollBar()
        old_value = bar.value()
        follow_tail = self._chat_follow_tail

        visible_nodes = [
            node for node in self._chat_nodes
            if self._show_debug_info or node.kind != "debug"
        ]

        timeline_parts: list[str] = ['<div class="timeline">']
        for idx, node in enumerate(visible_nodes):
            timeline_parts.append(node.html)
            if idx != len(visible_nodes) - 1:
                timeline_parts.append('<div class="divider"></div>')
        timeline_parts.append("</div>")

        combined = "".join(timeline_parts)
        doc = (
            "<html><head>"
            f"{self._chat_style_block()}"
            "</head><body>"
            f"{combined}"
            "</body></html>"
        )
        bar.blockSignals(True)
        self.chat_view.setHtml(doc)

        bar = self.chat_view.verticalScrollBar()
        if follow_tail:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(min(old_value, bar.maximum()))
        bar.blockSignals(False)

    def _chat_style_block(self) -> str:
        return """
<style>
body {
  margin: 0;
  padding: 10px;
  background: #ffffff;
  color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}
.timeline {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.divider {
  height: 1px;
  background: #eef2f7;
}
.bubble {
  border: 1px solid #e5eaf2;
  border-radius: 10px;
  background: #ffffff;
  padding: 10px 12px;
}
.bubble-header {
  font-size: 12px;
  color: #4b5563;
  margin-bottom: 6px;
  font-weight: 600;
}
.bubble-content {
  line-height: 1.55;
  color: #111827;
}
.bubble.user {
  border-color: #dbeafe;
  background: #f6faff;
}
.bubble.agent {
  border-color: #e5e7eb;
}
.bubble.thinking {
  border-color: #e5e7eb;
  background: #f8fafc;
}
.bubble.tool {
  border-color: #dbeafe;
  background: #f8fbff;
}
.tool-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  gap: 8px;
}
.tool-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: #1e3a8a;
  font-size: 12px;
}
.tool-status {
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 11px;
  font-weight: 600;
}
.tool-status.running {
  background: #e0f2fe;
  color: #075985;
}
.tool-status.success {
  background: #dcfce7;
  color: #166534;
}
.tool-status.fail {
  background: #fee2e2;
  color: #991b1b;
}
.tool-args {
  margin: 2px 0 0 18px;
  padding: 0;
}
.tool-args li {
  margin: 2px 0;
}
.tool-result {
  margin-top: 10px;
  padding: 8px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
}
.tool-result.success {
  border-color: #bbf7d0;
  background: #f0fdf4;
}
.tool-result.fail {
  border-color: #fecaca;
  background: #fef2f2;
}
.tool-result-title {
  font-size: 12px;
  color: #4b5563;
  margin-bottom: 4px;
}
.tool-result-body {
  color: #111827;
  white-space: pre-wrap;
}
.placeholder {
  color: #6b7280;
  font-style: italic;
}
.md-content p {
  margin: 0 0 8px 0;
}
.md-content p:last-child {
  margin-bottom: 0;
}
.md-content ul,
.md-content ol {
  margin: 0 0 8px 20px;
  padding: 0;
}
.md-content p + ul,
.md-content p + ol {
  margin-top: 0;
}
.md-content li {
  margin: 0;
}
.md-content pre {
  margin: 6px 0;
  padding: 8px;
  border-radius: 8px;
  background: #0f172a;
  color: #e2e8f0;
  overflow-x: auto;
}
.md-content code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.md-content blockquote {
  margin: 6px 0;
  padding: 0 0 0 10px;
  border-left: 3px solid #d1d5db;
  color: #4b5563;
}
.md-content table.md-table {
  border-collapse: collapse;
  margin: 8px 0;
}
.md-content table.md-table th,
.md-content table.md-table td {
  border: 1px solid #e5e7eb;
  padding: 4px 8px;
}
.meta {
  color: #808080;
  font-size: 11px;
  margin: 2px 0;
}
</style>
"""
    def _get_run_state(self, run_id: str) -> RunRenderState | None:
        return self._run_render_state.get(run_id)

    def _get_or_create_run_state(self, run_id: str) -> RunRenderState:
        state = self._run_render_state.get(run_id)
        if state is not None:
            return state
        state = RunRenderState()
        self._run_render_state[run_id] = state
        return state

    def _get_or_create_agent_stream(self, state: RunRenderState) -> MarkdownStreamState:
        if state.agent is not None:
            return state.agent
        node_id = self._append_chat_node("agent", self._render_agent_bubble(""))
        state.agent = MarkdownStreamState(node_id=node_id)
        return state.agent

    def _close_agent_stream(self, state: RunRenderState):
        if state.agent is None:
            return
        self._flush_stream(state.agent, self._render_agent_bubble)
        state.agent = None

    def _apply_delta_to_stream(
        self,
        stream: MarkdownStreamState,
        delta: str,
        render_fn: Callable[[str], str],
    ):
        ops = stream.parser.feed(delta)
        if not ops:
            return
        stream.renderer.apply(ops)
        self._update_chat_node(
            stream.node_id,
            render_fn(stream.renderer.html()),
        )

    def _flush_stream(self, stream: MarkdownStreamState, render_fn: Callable[[str], str]):
        ops = stream.parser.flush()
        if ops:
            stream.renderer.apply(ops)
        self._update_chat_node(
            stream.node_id,
            render_fn(stream.renderer.html()),
        )

    def _render_user_bubble(self, queue_label: str, content: str) -> str:
        safe_label = html.escape(queue_label)
        safe_content = html.escape(content).replace("\n", "<br/>")
        return (
            '<section class=\"bubble user\">'
            f'<div class=\"bubble-header\">用户/{safe_label}</div>'
            f'<div class=\"bubble-content\">{safe_content}</div>'
            "</section>"
        )

    def _render_agent_bubble(self, content_html: str) -> str:
        body = content_html or '<span class=\"placeholder\">...</span>'
        return (
            '<section class=\"bubble agent\">'
            '<div class=\"bubble-header\">Agent</div>'
            f'<div class=\"bubble-content md-content\">{body}</div>'
            "</section>"
        )

    def _render_thinking_bubble(self, content_html: str) -> str:
        body = content_html or '<span class=\"placeholder\">...</span>'
        return (
            '<section class=\"bubble thinking\">'
            '<div class=\"bubble-header\">思考片段</div>'
            f'<div class=\"bubble-content md-content\">{body}</div>'
            "</section>"
        )

    def _render_tool_bubble(self, state: ToolBubbleState) -> str:
        status_text = {
            "running": "运行中",
            "success": "成功",
            "fail": "失败",
        }.get(state.status, "运行中")
        safe_name = html.escape(state.name or "pending")
        args_html = render_json_tree_html(state.args_tree)

        result_section = ""
        if state.result_preview or state.duration_ms is not None:
            duration = "" if state.duration_ms is None else f" · {state.duration_ms:.0f}ms"
            safe_result = html.escape(state.result_preview).replace("\n", "<br/>")
            result_body = safe_result if safe_result else "(empty)"
            result_section = (
                f'<div class=\"tool-result {state.status}\">'
                f'<div class=\"tool-result-title\">结果{duration}</div>'
                f'<div class=\"tool-result-body\">{result_body}</div>'
                "</div>"
            )

        return (
            '<section class=\"bubble tool\">'
            '<div class=\"tool-header-row\">'
            f'<span class=\"tool-name\">⚙ {safe_name}</span>'
            f'<span class=\"tool-status {state.status}\">{status_text}</span>'
            "</div>"
            f'<div class=\"bubble-content\">{args_html}</div>'
            f"{result_section}"
            "</section>"
        )

    def _get_or_create_tool_bubble(self, run_id: str, tool_call_id: str, tool_name: str) -> ToolBubbleState:
        bubble = self._tool_bubbles.get(tool_call_id)
        run_state = self._get_run_state(run_id)
        if bubble is None:
            if run_state is not None:
                self._close_agent_stream(run_state)
            node_id = self._append_chat_node("tool", "")
            bubble = ToolBubbleState(
                node_id=node_id,
                run_id=run_id,
                name=tool_name or "pending",
                args_tree={},
            )
            self._tool_bubbles[tool_call_id] = bubble
        else:
            bubble.run_id = run_id or bubble.run_id
            if tool_name:
                bubble.name = tool_name

        if run_state is not None:
            run_state.tool_call_ids.add(tool_call_id)

        self._update_chat_node(
            bubble.node_id,
            self._render_tool_bubble(bubble),
        )
        return bubble

    def _as_meta_block(self, text: str) -> str:
        safe = html.escape(text).replace("\n", "<br/>")
        return f'<p class=\"meta\">{safe}</p>'

    def _truncate(self, text: str, limit: int = 600) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]} ...[截断]"

    @staticmethod
    def _split_model_name(model_name: str) -> tuple[str, str]:
        if "/" in model_name:
            provider, model_id = model_name.split("/", 1)
            return provider, model_id
        return "default", model_name

    def _make_status_separator(self) -> QWidget:
        sep = QWidget(self._status_strip)
        sep.setFixedWidth(1)
        sep.setFixedHeight(max(12, self._top_bar_height - 6))
        sep.setStyleSheet("background-color: #D1D5DB;")
        return sep

    def _make_top_separator(self) -> QWidget:
        sep = QWidget(self)
        sep.setFixedWidth(1)
        sep.setFixedHeight(max(14, self._top_button_height - 8))
        sep.setStyleSheet("background-color: #D1D5DB;")
        return sep

    def _state_color(self, state: str) -> QColor:
        palette = {
            "IDLE": "#6B7280",
            "READY": "#2563EB",
            "RUNNING": "#0F766E",
            "INTERRUPTING": "#B45309",
            "ERROR": "#B91C1C",
        }
        return QColor(palette.get(state.upper(), "#374151"))

    def _set_status_cell(
        self,
        column: int,
        text: str,
        *,
        color: QColor | None = None,
        bold: bool = False,
    ):
        cell = self._status_cells[column]
        text_color = color or QColor("#111827")
        font_weight = "600" if bold else "500"
        cell.setText(text)
        cell.setStyleSheet(
            f"color: {text_color.name()};"
            f"padding: 0px 6px;"
            f"font-weight: {font_weight};"
            "background: transparent;"
            "border: none;"
        )

    def _set_state_cell(self, column: int, label: str, state: str):
        self._set_status_cell(
            column,
            f"{label}: {state}",
            color=self._state_color(state),
            bold=True,
        )

    def _set_model_header(self, model_name: str):
        provider, model_id = self._split_model_name(model_name)
        self._model_btn.setText(f"Model: {provider}/{model_id}")

    @staticmethod
    def _neutral_button_style(horizontal_padding_px: int = 0) -> str:
        return (
            "QPushButton { "
            f"padding: 0px {horizontal_padding_px}px; "
            "margin: 0px; "
            "border: 1px solid #D1D5DB; "
            "border-radius: 4px; "
            "background-color: #FFFFFF; "
            "color: #111827; "
            "font-weight: 500; "
            "text-align: center; "
            "} "
            "QPushButton:disabled { "
            "border: 1px solid #D1D5DB; "
            "background-color: #F3F4F6; "
            "color: #9CA3AF; "
            "}"
        )

    def _selected_queue(self) -> QueueKind:
        for key, btn in self._priority_buttons.items():
            if btn.isChecked():
                return key
        return "normal"

    def _set_selected_queue(self, queue_kind: QueueKind):
        self._priority_buttons[queue_kind].setChecked(True)

    def _cycle_priority(self):
        order: list[QueueKind] = ["normal", "high_prio", "urgent"]
        current = self._selected_queue()
        idx = order.index(current)
        self._set_selected_queue(order[(idx + 1) % len(order)])

    # ─── Commands ─────────────────────────────────────────────────────────────

    def _submit_input(self):
        text = self.input_box.toPlainText().strip()
        if not text:
            return
        self.input_box.clear()
        if text.startswith("/"):
            self._handle_command(text)
            return
        queue_kind = self._selected_queue()
        self.cmd_queue.put(CmdEnqueue(content=text, queue=queue_kind))

    def _focus_input_box(self):
        self.input_box.setFocus(Qt.FocusReason.ShortcutFocusReason)
        cursor = self.input_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.input_box.setTextCursor(cursor)

    def _request_interrupt(self):
        # TODO: support hard interrupt for long-running external tools/subprocesses.
        if self._agent_state == "IDLE":
            return
        self.cmd_queue.put(CmdInterrupt(reason="user"))
        self._append_system("已请求中断当前执行")

    def _refresh_action_buttons(self):
        running = self._agent_state != "IDLE"
        self._send_btn.setEnabled(True)
        self._send_btn.setStyleSheet(self._neutral_button_style())
        self._stop_btn.setEnabled(running)
        self._stop_btn.setStyleSheet(
            "QPushButton { "
            "padding: 0px; "
            "margin: 0px; "
            "border: 1px solid #7F1D1D; "
            "border-radius: 4px; "
            "background-color: #991B1B; "
            "color: white; "
            "font-weight: 600; "
            "text-align: center; "
            "} "
            "QPushButton:disabled { "
            "border: 1px solid #D1D5DB; "
            "background-color: #F3F4F6; "
            "color: #9CA3AF; "
            "}"
        )
        self._sync_action_buttons_size()

    def _sync_action_buttons_size(self, target_height: int | None = None):
        if not hasattr(self, "_send_btn"):
            return
        if target_height is None:
            target_height = self.input_box.height()
        target_height = max(32, int(target_height))
        self._send_btn.setFixedHeight(target_height)
        if hasattr(self, "_stop_btn"):
            self._stop_btn.setFixedHeight(target_height)

    def _handle_command(self, text: str):
        cmd = text.lower().strip()
        if cmd in ("/help", "/h"):
            self._append_system(HELP_TEXT)
        elif cmd == "/clear":
            self._cmd_clear_context()
        elif cmd == "/cq":
            self.cmd_queue.put(CmdClearQueue(queue="normal"))
            self._append_system("普通队列已清空")
        elif cmd == "/chq":
            self.cmd_queue.put(CmdClearQueue(queue="high_prio"))
            self._append_system("高优队列已清空")
        elif cmd == "/cuq":
            self.cmd_queue.put(CmdClearQueue(queue="urgent"))
            self._append_system("紧急队列已清空")
        elif cmd == "/ca":
            self.cmd_queue.put(CmdClearQueue(queue="all"))
            self._append_system("所有队列已清空")
        elif cmd in ("/quit", "/q", "/exit"):
            self.close()
        else:
            self._append_system(f"未知命令: {text}")

    def _cmd_clear_context(self):
        self.cmd_queue.put(CmdClearContext())
        self._append_system("对话上下文已清空")

    def _show_model_switcher(self):
        from hawi.models import model_registry

        models = model_registry.list_models()
        if not models:
            self._append_error("没有可用模型")
            return

        dlg = ModelSwitchDialog(
            models=models,
            current_model=self.model_name,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        model_name = str(dlg.selected_model or "").strip()
        if not model_name:
            return
        if model_name == self.model_name:
            return
        self.cmd_queue.put(CmdSwitchModel(model_name=model_name))

    def _open_plugin_dialog(self):
        if self._scheduler_state != "IDLE":
            QMessageBox.information(self, "提示", "Agent 正在运行，请在空闲状态下应用插件配置。")
            return

        dlg = PluginConfigDialog(
            catalog=self._catalog,
            selected_plugins=list(self._selected_plugins),
            plugin_configs={k: dict(v) for k, v in self._plugin_configs.items()},
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self.cmd_queue.put(
            CmdApplyPlugins(
                selected_plugins=list(dlg.selected_plugins_result),
                plugin_configs={k: dict(v) for k, v in dlg.plugin_configs_result.items()},
            )
        )

    # ─── UI queue dispatch ────────────────────────────────────────────────────

    def _poll_ui_queue(self):
        while True:
            try:
                msg = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            self._dispatch(msg)

    def _dispatch(self, msg: Any):
        if isinstance(msg, UiStatusUpdate):
            self._scheduler_state = msg.scheduler_state
            self._agent_state = msg.agent_state
            self._set_state_cell(1, "Scheduler", msg.scheduler_state)
            self._set_state_cell(2, "Agent", msg.agent_state)
            q = msg.queue_lengths
            self._set_status_cell(
                3,
                f"Queue U/H/N: {q.get('urgent', 0)}/{q.get('high_prio', 0)}/{q.get('normal', 0)}"
            )
            idle = msg.scheduler_state == "IDLE"
            self._plugin_btn.setEnabled(idle)
            self._refresh_action_buttons()

        elif isinstance(msg, UiReady):
            self.model_name = msg.model_name
            self.setWindowTitle(f"Hawi - {msg.model_name}")
            self._set_model_header(msg.model_name)
            self._selected_plugins = list(msg.selected_plugins)
            self._plugin_configs = {k: dict(v) for k, v in msg.plugin_configs.items()}
            self._append_system(f"模型已就绪: {msg.model_name}")

        elif isinstance(msg, UiRunStart):
            queue_label = {
                "normal": "普通",
                "high_prio": "高优",
                "urgent": "紧急",
            }.get(msg.queue_kind, msg.queue_kind)
            self._append_chat_node(
                "user",
                self._render_user_bubble(queue_label, msg.user_content),
            )
            self._active_run_id = msg.run_id
            self._run_render_state[msg.run_id] = RunRenderState()

        elif isinstance(msg, UiTextDelta):
            run_id = msg.run_id or self._active_run_id or ""
            if not run_id:
                return
            state = self._get_or_create_run_state(run_id)
            stream = self._get_or_create_agent_stream(state)
            self._apply_delta_to_stream(stream, msg.delta, self._render_agent_bubble)

        elif isinstance(msg, UiThinkingDelta):
            run_id = msg.run_id or self._active_run_id or ""
            if not run_id:
                return
            state = self._get_or_create_run_state(run_id)
            if state.thinking is None:
                thinking_node_id = self._append_chat_node(
                    "thinking",
                    self._render_thinking_bubble(""),
                )
                state.thinking = MarkdownStreamState(node_id=thinking_node_id)
            self._apply_delta_to_stream(state.thinking, msg.delta, self._render_thinking_bubble)

        elif isinstance(msg, UiRunStop):
            state = self._run_render_state.pop(msg.run_id, None)
            if state is not None:
                self._close_agent_stream(state)
                if state.thinking is not None:
                    self._flush_stream(state.thinking, self._render_thinking_bubble)
            self._append_meta(f"完成: {msg.stop_reason} · {msg.duration_ms / 1000:.1f}s")
            if self._active_run_id == msg.run_id:
                self._active_run_id = None

        elif isinstance(msg, UiToolCallStart):
            run_id = msg.run_id or self._active_run_id or ""
            bubble = self._get_or_create_tool_bubble(run_id, msg.tool_call_id, msg.tool_name)
            bubble.status = "running"
            if msg.tool_name:
                bubble.name = msg.tool_name
            self._update_chat_node(
                bubble.node_id,
                self._render_tool_bubble(bubble),
            )

        elif isinstance(msg, UiToolCallDelta):
            run_id = msg.run_id or self._active_run_id or ""
            bubble = self._get_or_create_tool_bubble(run_id, msg.tool_call_id, "pending")
            bubble.args_tree = bubble.json_state.feed(msg.delta)
            self._update_chat_node(
                bubble.node_id,
                self._render_tool_bubble(bubble),
            )

        elif isinstance(msg, UiToolCallStop):
            run_id = msg.run_id or self._active_run_id or ""
            bubble = self._get_or_create_tool_bubble(run_id, msg.tool_call_id, msg.tool_name)
            bubble.final_arguments = msg.arguments
            bubble.args_tree = bubble.json_state.finalize(msg.arguments)
            self._update_chat_node(
                bubble.node_id,
                self._render_tool_bubble(bubble),
            )

        elif isinstance(msg, UiToolResult):
            run_id = msg.run_id or self._active_run_id or ""
            bubble = self._get_or_create_tool_bubble(run_id, msg.tool_call_id, msg.tool_name)
            bubble.status = "success" if msg.success else "fail"
            bubble.duration_ms = msg.duration_ms
            bubble.result_preview = self._truncate(str(msg.output).strip())
            if msg.tool_name:
                bubble.name = msg.tool_name
            self._update_chat_node(
                bubble.node_id,
                self._render_tool_bubble(bubble),
            )

        elif isinstance(msg, UiInterrupt):
            self._append_system(f"执行被中断: {msg.reason}")

        elif isinstance(msg, UiAgentInterrupt):
            self._append_system(f"Agent 中断: {msg.interrupt_type}")

        elif isinstance(msg, UiModelMetadata):
            latency = "n/a" if msg.latency_ms is None else f"{msg.latency_ms:.0f}ms"
            self._append_meta(
                f"模型统计 in={msg.input_tokens} out={msg.output_tokens} "
                f"total={msg.total_tokens} latency={latency}"
            )

        elif isinstance(msg, UiModelRetry):
            self._append_system(
                f"模型重试 {msg.attempt}/{msg.max_retries}: [{msg.error_type}] {msg.error_message}"
            )

        elif isinstance(msg, UiPluginsApplied):
            if msg.success:
                self._selected_plugins = list(msg.selected_plugins)
                self._plugin_configs = {k: dict(v) for k, v in msg.plugin_configs.items()}
                self._save_plugin_state()
                self._append_system("插件配置已应用并保存")
            else:
                self._append_error(msg.message)

        elif isinstance(msg, UiError):
            self._append_error(msg.message)

        elif isinstance(msg, UiDebugInfo):
            self._append_debug(msg.message)
