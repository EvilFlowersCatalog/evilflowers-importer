"""
TUI module for EvilFlowers Book Import.

This module provides a Text User Interface (TUI) for the application using the rich library.
"""

import os
import sys
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn

# Set up logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("evilflowers_importer")

# Create console
console = Console()


class LogPanel:
    """A panel that displays log messages."""

    def __init__(self, max_lines: int = 200):
        """
        Initialize the log panel.

        Args:
            max_lines (int): Maximum number of lines to display in the log panel.
        """
        self.max_lines = max_lines
        self.logs: List[str] = []
        self.lock = threading.Lock()

    def add_log(self, message: str, level: str = "INFO"):
        """
        Add a log message to the panel.

        Args:
            message (str): The log message to add.
            level (str): The log level (INFO, WARNING, ERROR, etc.).
        """
        with self.lock:
            # Format the log message with timestamp and level
            timestamp = time.strftime("[%H:%M:%S]")
            formatted_message = f"{timestamp} [{level}] {message}"

            # Add the message to the logs
            self.logs.append(formatted_message)

            # Trim logs if they exceed max_lines
            if len(self.logs) > self.max_lines:
                self.logs = self.logs[-self.max_lines:]

    def get_panel(self) -> Panel:
        """
        Get the log panel.

        Returns:
            Panel: A rich Panel containing the log messages.
        """
        with self.lock:
            # Create a Text object with the logs, ensuring the most recent logs are visible
            log_text = Text("\n".join(self.logs), overflow="crop")
            return Panel(log_text, title="Logs", border_style="blue")


class LogHandler(logging.Handler):
    """A logging handler that sends logs to the LogPanel."""

    def __init__(self, log_panel: LogPanel):
        """
        Initialize the log handler.

        Args:
            log_panel (LogPanel): The log panel to send logs to.
        """
        super().__init__()
        self.log_panel = log_panel

    def emit(self, record):
        """
        Emit a log record.

        Args:
            record (LogRecord): The log record to emit.
        """
        level = record.levelname
        message = self.format(record)
        self.log_panel.add_log(message, level)


class ProgressManager:
    """A manager for rich progress bars."""

    def __init__(self):
        """Initialize the progress manager."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
        )
        self.tasks: Dict[str, TaskID] = {}

    def add_task(self, description: str, total: int) -> str:
        """
        Add a task to the progress manager.

        Args:
            description (str): The task description.
            total (int): The total number of steps in the task.

        Returns:
            str: The task ID.
        """
        # Get the task ID from the progress and ensure it's a string
        rich_task_id = self.progress.add_task(description, total=total)
        task_id = str(rich_task_id)

        # Store the original rich_task_id for use with the progress object
        self.tasks[task_id] = rich_task_id

        return task_id

    def update(self, task_id: str, advance: int = 1, **kwargs):
        """
        Update a task in the progress manager.

        Args:
            task_id (str): The task ID.
            advance (int): The number of steps to advance the task.
            **kwargs: Additional arguments to pass to the progress.update method.
        """
        # Convert task_id to string to ensure consistent handling
        task_id = str(task_id)
        if task_id in self.tasks:
            try:
                self.progress.update(self.tasks[task_id], advance=advance, **kwargs)
            except KeyError:
                logger.warning(f"Task ID {task_id} not found in progress manager")
        else:
            logger.warning(f"Task ID {task_id} not found in tasks dictionary")

    def remove_task(self, task_id: str):
        """
        Remove a task from the progress manager.

        Args:
            task_id (str): The task ID.
        """
        # Convert task_id to string to ensure consistent handling
        task_id = str(task_id)
        if task_id in self.tasks:
            try:
                self.progress.remove_task(self.tasks[task_id])
                del self.tasks[task_id]
            except KeyError:
                logger.warning(f"Task ID {task_id} not found in progress manager")
        else:
            logger.warning(f"Task ID {task_id} not found in tasks dictionary")

    def get_panel(self) -> Panel:
        """
        Get the progress panel.

        Returns:
            Panel: A rich Panel containing the progress bars.
        """
        return Panel(self.progress, title="Progress", border_style="green")


class StatusPanel:
    """A panel that displays status information."""

    def __init__(self):
        """Initialize the status panel."""
        self.status = "Initializing..."
        self.stats: Dict[str, Any] = {}

    def update_status(self, status: str):
        """
        Update the status message.

        Args:
            status (str): The new status message.
        """
        self.status = status

    def update_stats(self, key: str, value: Any):
        """
        Update a statistic.

        Args:
            key (str): The statistic key.
            value (Any): The statistic value.
        """
        self.stats[key] = value

    def get_panel(self) -> Panel:
        """
        Get the status panel.

        Returns:
            Panel: A rich Panel containing the status information.
        """
        # Create a table for the statistics
        table = Table(show_header=False, box=None)
        table.add_column("Key")
        table.add_column("Value")

        # Add the statistics to the table
        for key, value in self.stats.items():
            table.add_row(f"[bold]{key}:[/bold]", str(value))

        # Create the status text
        status_text = Text.from_markup(f"[bold]Status:[/bold] {self.status}\n\n")

        # Create a group with status text and table
        from rich.console import Group
        content = Group(status_text, table)

        # Create the panel
        return Panel(content, title="Status", border_style="yellow")


class TUIApp:
    """The main TUI application."""

    def __init__(self):
        """Initialize the TUI application."""
        # Create the layout
        self.layout = Layout()

        # Split the layout into main and logs
        self.layout.split_column(
            Layout(name="main", ratio=2),
            Layout(name="logs", ratio=1)
        )

        # Split the main layout into status and progress
        self.layout["main"].split_row(
            Layout(name="status", ratio=1),
            Layout(name="progress", ratio=2)
        )

        # Create the panels
        self.log_panel = LogPanel()
        self.progress_manager = ProgressManager()
        self.status_panel = StatusPanel()

        # Set up the log handler
        self.log_handler = LogHandler(self.log_panel)
        self.log_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(self.log_handler)

        # Set the panels in the layout
        self.layout["status"].update(self.status_panel.get_panel())
        self.layout["progress"].update(self.progress_manager.get_panel())
        self.layout["logs"].update(self.log_panel.get_panel())

        # Create the live display with keyboard support
        self.live = Live(self.layout, refresh_per_second=4)

        # Flag to indicate if the application should exit
        self.should_exit = False

        # Initialize update thread
        self.update_thread = None
        self.stop_event = threading.Event()

        # Initialize application statistics
        self.current_directory = None
        self.processed_directories = 0
        self.start_time = None
        self.total_directories = 0

        # Initialize command-line arguments
        self.input_dir = None
        self.results_file = None
        self.model_type = None
        self.model_name = None
        self.workers = None
        self.limit = None
        self.total_found_directories = 0
        self.skipped_directories = 0

        # Initialize function result
        self._func_result = None

        # Add shutdown message to status panel
        self.status_panel.update_stats("Shutdown", "Press 'q' or Ctrl+C to exit")

    def start(self):
        """Start the TUI application."""
        # Set up keyboard event handler
        import signal

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, shutting down gracefully...")
            self.should_exit = True
            self.status_panel.update_status("Shutting down...")

        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)

        # Start the live display
        self.live.start(refresh=True)

        # Start the update thread
        self.stop_event.clear()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        # Initialize status panel with some basic info
        self.status_panel.update_status("Running")
        self.update_status_stats()

    def stop(self):
        """Stop the TUI application."""
        # Stop the update thread
        if self.update_thread and self.update_thread.is_alive():
            self.stop_event.set()
            self.update_thread.join(timeout=1.0)

        self.live.stop()

    def _update_loop(self):
        """Background thread that updates the UI periodically."""
        while not self.stop_event.is_set() and not self.should_exit:
            try:
                # Update status statistics
                self.update_status_stats()

                # Update the UI
                self.update()

                # Check for keyboard input
                self._check_keyboard_input()

                # Sleep for a short time
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1.0)  # Sleep longer on error

    def _check_keyboard_input(self):
        """Check for keyboard input to handle graceful shutdown."""
        try:
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                if key == 'q':
                    logger.info("Received 'q' key, shutting down gracefully...")
                    self.should_exit = True
                    self.status_panel.update_status("Shutting down...")
        except ImportError:
            # msvcrt is not available on non-Windows platforms
            try:
                import sys, select
                # Check if there's input available
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 'q':
                        logger.info("Received 'q' key, shutting down gracefully...")
                        self.should_exit = True
                        self.status_panel.update_status("Shutting down...")
            except (ImportError, Exception) as e:
                # If we can't check for keyboard input, just continue
                pass

    def update_status_stats(self):
        """Update the status panel with current statistics."""
        try:
            # Clear existing stats
            self.status_panel.stats = {}

            # Section: Configuration
            self.status_panel.update_stats("[bold blue]Configuration[/bold blue]", "")

            # Display command-line arguments
            if self.input_dir:
                self.status_panel.update_stats("Input Directory", self.input_dir)

            if self.results_file:
                self.status_panel.update_stats("Results File", self.results_file)

            if self.model_type:
                model_info = f"{self.model_type}"
                if self.model_name:
                    model_info += f":{self.model_name}"
                self.status_panel.update_stats("Model", model_info)

            if self.workers is not None:
                self.status_panel.update_stats("Workers", self.workers if self.workers else "auto")

            # Section: Progress
            self.status_panel.update_stats("[bold green]Progress[/bold green]", "")

            # Display current directory being processed
            if self.current_directory:
                dir_name = os.path.basename(self.current_directory)
                self.status_panel.update_stats("Current Directory", dir_name)

            # Display processing statistics
            if self.total_found_directories > 0:
                # Display total directories found
                self.status_panel.update_stats("Total Directories", self.total_found_directories)

                # Display skipped directories
                if self.skipped_directories > 0:
                    self.status_panel.update_stats("Skipped", self.skipped_directories)

                # Display limit if applied
                if self.limit is not None:
                    self.status_panel.update_stats("Limit", self.limit)

                # Display processed and remaining directories
                if self.total_directories > 0:
                    self.status_panel.update_stats("Processed", f"{self.processed_directories}/{self.total_directories}")
                    remaining = self.total_directories - self.processed_directories
                    self.status_panel.update_stats("Remaining", remaining)

            # Calculate and display estimated time remaining
            if self.start_time and self.processed_directories > 0 and self.total_directories > self.processed_directories:
                elapsed_time = time.time() - self.start_time
                avg_time_per_dir = elapsed_time / self.processed_directories
                remaining_dirs = self.total_directories - self.processed_directories
                estimated_time = avg_time_per_dir * remaining_dirs

                # Format estimated time as HH:MM:SS
                hours, remainder = divmod(int(estimated_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                self.status_panel.update_stats("Est. Time Remaining", time_str)

            # Section: Tasks
            self.status_panel.update_stats("[bold yellow]Tasks[/bold yellow]", "")

            # Count active tasks
            active_tasks = len(self.progress_manager.tasks)
            self.status_panel.update_stats("Active Tasks", active_tasks)

            # Section: System
            self.status_panel.update_stats("[bold magenta]System[/bold magenta]", "")

            # Get current time
            current_time = time.strftime("%H:%M:%S")
            self.status_panel.update_stats("Current Time", current_time)
        except Exception as e:
            logger.error(f"Error updating status stats: {e}")

    def set_args(self, args, total_found_directories, skipped_directories):
        """
        Set command-line arguments and related statistics.

        Args:
            args: Command-line arguments
            total_found_directories (int): Total number of directories found
            skipped_directories (int): Number of directories skipped (already processed)
        """
        self.input_dir = args.input_dir
        self.results_file = args.results_file
        self.model_type = args.model_type
        self.model_name = args.model_name if args.model_name else ("gpt-4o" if args.model_type == "openai" else "mistral")
        self.workers = args.workers
        self.limit = args.limit
        self.total_found_directories = total_found_directories
        self.skipped_directories = skipped_directories

    def update(self):
        """Update the TUI display."""
        self.layout["status"].update(self.status_panel.get_panel())
        self.layout["progress"].update(self.progress_manager.get_panel())
        self.layout["logs"].update(self.log_panel.get_panel())

    def run_with_app(self, func: Callable, *args, **kwargs):
        """
        Run a function with the TUI application.

        Args:
            func (Callable): The function to run.
            *args: Arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The return value of the function.
        """
        result = None
        try:
            self.start()

            # Create a thread to run the function
            import threading
            func_thread = threading.Thread(target=lambda: self._run_func(func, args, kwargs))
            func_thread.daemon = True
            func_thread.start()

            # Wait for the function to complete or for the user to request exit
            while func_thread.is_alive() and not self.should_exit:
                time.sleep(0.1)

            if self.should_exit:
                logger.info("Graceful shutdown requested, stopping processing...")
                # Return a special value to indicate graceful shutdown
                result = 0
            else:
                # Function completed normally
                result = self._func_result

            return result
        finally:
            self.stop()

    def _run_func(self, func: Callable, args, kwargs):
        """Run the function and store its result."""
        self._func_result = func(*args, **kwargs)


# Create a global TUI app instance
tui_app = TUIApp()


class RichProgressBar:
    """A progress bar that uses rich."""

    def __init__(self, total: int, description: str, leave: bool = True):
        """
        Initialize the progress bar.

        Args:
            total (int): The total number of steps in the progress bar.
            description (str): The description of the progress bar.
            leave (bool): Whether to leave the progress bar after completion.
        """
        self.total = total
        self.description = description
        self.leave = leave
        self.task_id = tui_app.progress_manager.add_task(description, total=total)
        self.n = 0

    def update(self, n: int = 1):
        """
        Update the progress bar.

        Args:
            n (int): The number of steps to advance the progress bar.
        """
        self.n += n
        try:
            tui_app.progress_manager.update(self.task_id, advance=n)
        except Exception as e:
            logger.warning(f"Failed to update progress bar: {e}")

    def set_description(self, description: str):
        """
        Set the description of the progress bar.

        Args:
            description (str): The new description.
        """
        self.description = description
        try:
            tui_app.progress_manager.update(self.task_id, description=description)
        except Exception as e:
            logger.warning(f"Failed to set progress bar description: {e}")

    def close(self):
        """Close the progress bar."""
        if not self.leave:
            try:
                tui_app.progress_manager.remove_task(self.task_id)
            except Exception as e:
                logger.warning(f"Failed to close progress bar: {e}")

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()


def run_app(func: Callable, *args, **kwargs):
    """
    Run a function with the TUI application.

    Args:
        func (Callable): The function to run.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Any: The return value of the function.
    """
    return tui_app.run_with_app(func, *args, **kwargs)
