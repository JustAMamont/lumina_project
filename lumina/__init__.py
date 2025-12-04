import time
import sys
import os
import atexit
import logging
from enum import IntEnum
from typing import Optional, Any, Tuple, Dict
from contextlib import contextmanager

try:
    from . import lumina_core # type: ignore
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import Rust extension.", file=sys.stderr)
    print(f"📄 Reason: {e}", file=sys.stderr)
    print(f"💡 Hint: Run 'maturin develop --release'", file=sys.stderr)
    sys.exit(1)

class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class LuminaHandler(logging.Handler):
    """
    A custom Logging Handler that intercepts standard Python logging records
    and forwards them to the Lumina Rust engine.
    """
    def __init__(self, lumina_instance: 'Lumina'):
        super().__init__()
        self.lumina = lumina_instance
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record: logging.LogRecord):
        try:
            # Temporarily remove exc_info to prevent formatting it into the message string
            # we want to pass the raw exception to Rust for better handling.
            cached_exc_info = record.exc_info
            record.exc_info = None 
            msg = self.format(record)
            record.exc_info = cached_exc_info
            exc = record.exc_info[1] if record.exc_info else None
            
            try:
                filename = os.path.relpath(record.pathname)
            except ValueError:
                filename = record.pathname

            lineno = record.lineno

            self.lumina._push(
                level=record.levelno,
                msg=msg,
                console=True,
                file=True,
                exc=exc,
                rate_limit=0.0,
                location=(filename, lineno)
            )
        except Exception:
            self.handleError(record)

class Lumina:
    """
    Main Logger Class (Singleton).
    Serves as the Python frontend for the high-performance Rust logging engine.
    """
    _instance: Optional['Lumina'] = None
    
    def __new__(cls, name: Optional[str], path_template: str, retention_days: int, channel_capacity: int, file_buffer_size: int, 
                ui_refresh_rate_ms: int, flush_interval_ms: int,
                text_enabled: bool, db_enabled: bool, capture_caller: bool, 
                colors_enabled: bool, theme: str):
        if cls._instance is None:
            cls._instance = super(Lumina, cls).__new__(cls)
            if not name:
                if sys.argv and sys.argv[0]:
                    name = os.path.basename(sys.argv[0])
                else:
                    name = "app"
            
            cls._instance._init(name, path_template, retention_days, channel_capacity, file_buffer_size, 
                                ui_refresh_rate_ms, flush_interval_ms, 
                                text_enabled, db_enabled, capture_caller, 
                                colors_enabled, theme)
        return cls._instance

    def _init(self, name: str, path_template: str, retention_days: int, channel_capacity: int, file_buffer_size: int, 
            ui_refresh_rate_ms: int, flush_interval_ms: int, 
            text_enabled: bool, db_enabled: bool, capture_caller: bool, 
            colors_enabled: bool, theme: str):
        
        # Initialize the Rust engine
        self._engine = lumina_core.LuminaEngine(
            name,
            path_template,
            channel_capacity, 
            file_buffer_size, 
            ui_refresh_rate_ms,
            flush_interval_ms,
            text_enabled,
            db_enabled,
            capture_caller,
            retention_days,
            colors_enabled,
            theme
        ) # type: ignore
        self.path_template = path_template
        self.capture_caller = capture_caller
        atexit.register(self._shutdown)

    def _shutdown(self):
        if hasattr(self, '_engine'):
            self._engine.terminate()
            # Prevent multiple calls
            delattr(self, '_engine')

    def shutdown(self):
        """Manually stops the logging engine and flushes pending logs."""
        self._shutdown()

    def catch_crashes(self):
        """
        Registers a global exception hook to catch unhandled exceptions (crashes)
        and log them with CRITICAL level before exiting.
        """
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            self.critical("Uncaught exception detected", exc=exc_value)
            # Give a moment for the log to be written before the process terminates
            time.sleep(0.2)
            
        sys.excepthook = handle_exception

    def intercept_std_logging(self, min_level: int = logging.INFO, remove_other_handlers: bool = True):
        """
        Redirects standard Python logging (e.g., from libraries like requests) to Lumina.
        """
        root = logging.getLogger()
        root.setLevel(min_level)
        if remove_other_handlers:
            for h in root.handlers[:]:
                root.removeHandler(h)
        handler = LuminaHandler(self)
        root.addHandler(handler)

    @classmethod
    def get_logger(cls, 
                   name: Optional[str] = None, 
                   path_template: str = "logs/{date}.ldb",
                   retention_days: int = 7, 
                   channel_capacity: int = 50_000, 
                   file_buffer_size: int = 64 * 1024,
                   ui_refresh_rate_ms: int = 50,
                   flush_interval_ms: int = 100,
                   text_enabled: bool = False,
                   db_enabled: bool = True,
                   capture_caller: bool = False,
                   colors_enabled: bool = True,
                   theme: str = "dark"
                   ) -> 'Lumina':
        """
        Factory method to get or create the Lumina logger instance.
        
        Args:
            name: Application name used in logs.
            path_template: File path pattern (supports {date}).
            retention_days: How many days to keep logs.
            channel_capacity: Max logs in memory queue before dropping.
            ui_refresh_rate_ms: How often to update the console UI.
            flush_interval_ms: How often to write to disk.
            capture_caller: If True, resolves file:line for every log (slower).
            theme: "dark" or "light".
        """
        if cls._instance is not None:
             # This is a soft-reconfiguration, not ideal for a true singleton but good for tests
             pass
        return cls(name, path_template, retention_days, channel_capacity, 
                   file_buffer_size, ui_refresh_rate_ms, flush_interval_ms, 
                   text_enabled, db_enabled, capture_caller, colors_enabled, theme)

    __slots__ = ('_engine', 'path_template', 'capture_caller')

    @contextmanager
    def profile(self, name: str, console: bool = True, 
                min_duration_ms: Optional[float] = None,
                slow_threshold_ms: Optional[float] = None,
                **tags):
        """
        Context manager for performance profiling.
        Measures Wall Time, CPU Time, RAM usage, and Page Faults.
        
        Args:
            name: Name of the task.
            min_duration_ms: If set, tasks faster than this won't be logged.
            slow_threshold_ms: If task exceeds this, it is logged as WARNING.
            tags: Extra context data to attach to the log.
        """
        
        # Snapshot resources (fast syscalls)
        snap_start = self._engine.snapshot_resources()
        t_start = time.perf_counter()
        
        try:
            yield
        finally:
            t_end = time.perf_counter()
            snap_end = self._engine.snapshot_resources()
            
            # Delegate heavy calculation and formatting to Rust
            self._engine.push_profile(
                name,
                snap_start,
                snap_end,
                t_start,
                t_end,
                min_duration_ms,
                slow_threshold_ms,
                console,
                tags
            )

    def _push(self, level: int, msg: object, console: bool, file: bool, exc: Optional[BaseException], 
              rate_limit: float, location: Optional[Tuple[str, int]] = None, 
              context: Optional[Dict[str, Any]] = None):
        ts = time.time()
        self._engine.push(ts, level, str(msg), self.path_template, console, file, rate_limit, exc, context, location)

    def debug(self, msg: Any, console: bool = True, file: bool = False, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(10, msg, console, file, None, rate_limit, None, context=kwargs)

    def info(self, msg: Any, console: bool = True, file: bool = True, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(20, msg, console, file, None, rate_limit, None, context=kwargs)

    def success(self, msg: Any, console: bool = True, file: bool = True, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(25, msg, console, file, None, rate_limit, None, context=kwargs)

    def warning(self, msg: Any, console: bool = True, file: bool = True, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(30, msg, console, file, None, rate_limit, None, context=kwargs)

    def error(self, msg: Any, exc: Optional[BaseException] = None, console: bool = True, file: bool = True, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(40, msg, console, file, exc, rate_limit, None, context=kwargs)

    def critical(self, msg: Any, exc: Optional[BaseException] = None, console: bool = True, file: bool = True, rate_limit: float = 0.0, **kwargs) -> None:
        self._push(50, msg, console, file, exc, rate_limit, None, context=kwargs)