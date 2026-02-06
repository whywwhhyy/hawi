"""
CLI UI utilities for terminal-based interactive interfaces.
Similar to Claude Code's selection interface.
"""

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

    selected_index = 0
    bindings = KeyBindings()

    @bindings.add("up")
    def move_up(event):
        nonlocal selected_index
        selected_index = (selected_index - 1) % len(items)
        event.app.invalidate()

    @bindings.add("down")
    def move_down(event):
        nonlocal selected_index
        selected_index = (selected_index + 1) % len(items)
        event.app.invalidate()

    @bindings.add("enter")
    def select_item(event):
        event.app.exit(result=items[selected_index])

    @bindings.add("c-c")
    @bindings.add("c-q")
    @bindings.add("escape")
    def cancel(event):
        event.app.exit(result=None)

    def get_formatted_text():
        lines = []
        lines.append(("class:title", f"{title}\n\n"))

        for i, item in enumerate(items):
            display = str(item)
            if i == selected_index:
                # Selected item with highlight
                lines.append(("class:selected", f"  ▸ {display}\n"))
            else:
                # Unselected item
                lines.append(("class:unselected", f"    {display}\n"))

        lines.append(("class:hint", "\n↑/↓ to navigate, Enter to select, Esc/Ctrl+C to cancel"))
        return lines

    style = Style.from_dict({
        "title": "bold cyan",
        "selected": "bold green",
        "unselected": "",
        "hint": "dim italic",
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
