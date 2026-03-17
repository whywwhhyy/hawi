"""
CLI UI utilities for terminal-based interactive interfaces.
Similar to Claude Code's selection interface.
"""

import shutil
from typing import TypeVar
from rich.console import Console
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window, FormattedTextControl
from prompt_toolkit.styles import Style

T = TypeVar("T")

console = Console()


def user_select(items: list[T], title: str = "Select an option:") -> T | None:
    """
    Display a terminal selection interface similar to Claude Code.

    Args:
        items: List of items to select from
        title: Title to display above the selector

    Returns:
        The selected item, or None if cancelled

    Example:
        >>> options = ["Option 1", "Option 2", "Option 3"]
        >>> result = user_select(options, "Choose one:")
        >>> print(f"You selected: {result}")
    """
    if not items:
        return None

    if len(items) == 1:
        return items[0]

    # Use arrow key-based selection with live rendering
    return _interactive_select(items, title)


def _interactive_select(items: list[T], title: str) -> T | None:
    """Internal function for interactive selection using prompt_toolkit."""

    # Filter state
    search_query = ""
    filtered_indices = list(range(len(items)))  # Indices into original items

    selected_index = 0  # Index into filtered_items
    scroll_offset = 0
    bindings = KeyBindings()

    # Calculate visible items count based on terminal height
    # Reserve lines for: title (2) + search (1) + top indicator (0-1) + bottom indicator (0-1) + hint (2)
    terminal_height = shutil.get_terminal_size().lines
    reserved_lines = 7  # title(2) + search(1) + hint(2) + buffer(2)
    VISIBLE_ITEMS = max(5, terminal_height - reserved_lines)  # at least 5 items

    def get_filtered_items():
        """Get list of (original_index, item) matching search query."""
        if not search_query:
            return list(range(len(items)))
        query_lower = search_query.lower()
        return [i for i, item in enumerate(items) if query_lower in str(item).lower()]

    def update_filter():
        """Update filtered list based on current search query."""
        nonlocal filtered_indices, selected_index, scroll_offset
        filtered_indices = get_filtered_items()
        selected_index = 0
        scroll_offset = 0

    @bindings.add("up")
    def move_up(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = (selected_index - 1) % len(filtered_indices)
        # Handle wrap-around: if we wrapped to end, scroll to show the end
        if selected_index == len(filtered_indices) - 1 and len(filtered_indices) > VISIBLE_ITEMS:
            scroll_offset = len(filtered_indices) - VISIBLE_ITEMS
        # Adjust scroll offset if selection goes above visible area
        elif selected_index < scroll_offset:
            scroll_offset = selected_index
        event.app.invalidate()

    @bindings.add("down")
    def move_down(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = (selected_index + 1) % len(filtered_indices)
        # Handle wrap-around: if we wrapped to 0, scroll to top
        if selected_index == 0:
            scroll_offset = 0
        # Adjust scroll offset if selection goes below visible area
        elif selected_index >= scroll_offset + VISIBLE_ITEMS:
            scroll_offset = selected_index - VISIBLE_ITEMS + 1
        event.app.invalidate()

    @bindings.add("pageup")
    def page_up(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = max(0, selected_index - VISIBLE_ITEMS)
        scroll_offset = max(0, scroll_offset - VISIBLE_ITEMS)
        event.app.invalidate()

    @bindings.add("pagedown")
    def page_down(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = min(len(filtered_indices) - 1, selected_index + VISIBLE_ITEMS)
        scroll_offset = min(max(0, len(filtered_indices) - VISIBLE_ITEMS), scroll_offset + VISIBLE_ITEMS)
        event.app.invalidate()

    @bindings.add("home")
    def go_home(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = 0
        scroll_offset = 0
        event.app.invalidate()

    @bindings.add("end")
    def go_end(event):
        nonlocal selected_index, scroll_offset
        if not filtered_indices:
            return
        selected_index = len(filtered_indices) - 1
        scroll_offset = max(0, len(filtered_indices) - VISIBLE_ITEMS)
        event.app.invalidate()

    @bindings.add("enter")
    def select_item(event):
        if filtered_indices:
            original_index = filtered_indices[selected_index]
            event.app.exit(result=items[original_index])

    @bindings.add("c-c")
    @bindings.add("c-q")
    @bindings.add("escape")
    def cancel(event):
        event.app.exit(result=None)

    @bindings.add("backspace")
    def handle_backspace(event):
        nonlocal search_query
        if search_query:
            search_query = search_query[:-1]
            update_filter()
        event.app.invalidate()

    @bindings.add("c-u")
    def clear_search(event):
        """Clear search with Ctrl+U."""
        nonlocal search_query
        if search_query:
            search_query = ""
            update_filter()
        event.app.invalidate()

    # Handle printable characters for search
    @bindings.add("<any>")
    def handle_typing(event):
        nonlocal search_query
        key = event.key_sequence[0].key
        # Only accept single printable characters
        if isinstance(key, str) and len(key) == 1 and key.isprintable():
            search_query += key
            update_filter()
            event.app.invalidate()

    def get_formatted_text():
        lines = []
        lines.append(("class:title", f"{title}\n"))

        # Show search query
        if search_query:
            lines.append(("class:search", f"  filter: {search_query}▌\n"))
        else:
            lines.append(("class:hint", "  (type to filter)\n"))

        lines.append(("", "\n"))

        if not filtered_indices:
            lines.append(("class:hint", "  (no matches)\n"))
        else:
            # Calculate visible range
            end_index = min(len(filtered_indices), scroll_offset + VISIBLE_ITEMS)

            # Show scroll indicator if not at the top
            if scroll_offset > 0:
                lines.append(("class:hint", f"  ... ({scroll_offset} more above)\n"))

            for i in range(scroll_offset, end_index):
                original_idx = filtered_indices[i]
                item = items[original_idx]
                display = str(item)
                if i == selected_index:
                    # Selected item with highlight
                    lines.append(("class:selected", f"  ▸ {display}\n"))
                else:
                    # Unselected item
                    lines.append(("class:unselected", f"    {display}\n"))

            # Show scroll indicator if not at the bottom
            if end_index < len(filtered_indices):
                remaining = len(filtered_indices) - end_index
                lines.append(("class:hint", f"  ... ({remaining} more below)\n"))

        lines.append(("class:hint", "\n↑/↓ to navigate, type to filter, Enter to select, Esc to cancel, Ctrl+U to clear"))
        return lines

    style = Style.from_dict({
        "title": "bold cyan",
        "selected": "bold green",
        "unselected": "",
        "hint": "dim italic",
        "search": "bold yellow",
    })

    layout = Layout(
        Window(
            FormattedTextControl(get_formatted_text),
            always_hide_cursor=True,
        )
    )

    app = Application(
        layout=layout,
        key_bindings=bindings,
        style=style,
        full_screen=False,
        mouse_support=True,
    )

    return app.run()


def user_select_rich(items: list[T], title: str = "Select an option:") -> T | None:
    """
    Alternative implementation using only Rich (no prompt_toolkit).
    Simpler but less interactive.
    """
    if not items:
        return None

    if len(items) == 1:
        return items[0]

    console.print(f"\n[bold cyan]{title}[/bold cyan]\n")

    for i, item in enumerate(items, 1):
        console.print(f"  [{i}] {item}")

    console.print("\n[dim italic]Enter number to select, or 'q' to cancel[/dim italic]")

    while True:
        try:
            choice = input("\n> ").strip()

            if choice.lower() in ("q", "quit", "exit"):
                return None

            index = int(choice) - 1
            if 0 <= index < len(items):
                return items[index]
            else:
                console.print(f"[red]Please enter a number between 1 and {len(items)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")
        except (KeyboardInterrupt, EOFError):
            return None


def user_confirm(message: str, default: bool = False) -> bool:
    """
    Ask for user confirmation (yes/no).

    Args:
        message: The confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if yes, False if no
    """
    default_str = "Y/n" if default else "y/N"
    console.print(f"\n[bold]{message}[/bold] [{default_str}]")

    try:
        response = input("> ").strip().lower()
        if not response:
            return default
        return response in ("y", "yes", "true", "1")
    except (KeyboardInterrupt, EOFError):
        return default


def user_input(message: str, default: str | None = None) -> str | None:
    """
    Get text input from user.

    Args:
        message: Prompt message
        default: Default value if user just presses Enter

    Returns:
        User input string, or None if cancelled
    """
    default_str = f" [{default}]" if default else ""
    console.print(f"\n[bold]{message}[/bold]{default_str}")

    try:
        response = input("> ").strip()
        if not response and default is not None:
            return default
        return response if response else None
    except (KeyboardInterrupt, EOFError):
        return None


if __name__ == "__main__":
    # Demo/test
    options = [
        "Open a file",
        "Create new project",
        "Run tests",
        "Deploy to production",
        "Settings",
    ]

    console.print("[bold]Testing user_select:[/bold]")
    result = user_select(options, "What would you like to do?")
    console.print(f"\n[green]You selected:[/green] {result}")
