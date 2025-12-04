import argparse
import sys, os
from datetime import datetime, timedelta

try:
    from lumina import lumina_core # type: ignore
except ImportError:
    try:
        from . import lumina_core # type: ignore
    except ImportError:
        print("❌ Lumina Core not found. If installed via pip, try reinstalling.", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Lumina Binary Log Reader")
    
    parser.add_argument("file", nargs='?', help="Path to .ldb log file OR pattern (e.g. 'logs/*.ldb'). If omitted, recursively scans for logs.")
    
    # === LEVEL FLAGS (Whitelist) ===
    parser.add_argument("-d", "--debug", action="store_true", help="Show DEBUG (10)")
    parser.add_argument("-i", "--info", action="store_true", help="Show INFO (20)")
    parser.add_argument("-s", "--success", action="store_true", help="Show SUCCESS (25)")
    parser.add_argument("-w", "--warning", action="store_true", help="Show WARNING (30)")
    parser.add_argument("-e", "--error", action="store_true", help="Show ERROR (40)")
    parser.add_argument("-c", "--critical", action="store_true", help="Show CRITICAL (50)")
    
    parser.add_argument("-l", "--min-level", type=int, default=0, help="Minimum log level (if no specific flags used)")

    # Content filters
    parser.add_argument("-g", "--grep", help="Filter by text message (case insensitive)")
    parser.add_argument("-t", "--tracetype", help="Filter by exception type (e.g. ZeroDivisionError)")
    parser.add_argument("--trace", action="store_true", help="Show full tracebacks for errors/warnings")
    
    parser.add_argument("--json", action="store_true", help="Output in JSONL format for machine reading (Stdout)")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow log output (like tail -f)")
    
    # Export options
    parser.add_argument("--export-json", type=str, help="Export logs to a JSONL file (History mode only)")
    
    # Time filters
    parser.add_argument("-D", "--days", type=int, help="Show logs from last N days")
    parser.add_argument("-H", "--hours", type=int, help="Show logs from last N hours")
    parser.add_argument("-m", "--minutes", type=int, help="Show logs from last N minutes")
    
    # THEME
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "light"], help="Color theme for console output")

    args = parser.parse_args()

    target_export_dir = os.path.join("logs", "exports")
    def process_export_path(file_path):
        """
        If only a filename is provided, saves to 'logs/exports'.
        If a full path is provided, ensures directories exist.
        """
        if not file_path:
            return None
        
        # If path has separators (user provided a custom folder)
        if os.path.dirname(file_path):
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            return file_path
        
        # If just a filename
        os.makedirs(target_export_dir, exist_ok=True)
        return os.path.join(target_export_dir, file_path)
    
    args.export_json = process_export_path(args.export_json)

    # Validation
    if args.follow and args.export_json:
        print("❌ Error: Export options (--export-json) are not available in follow mode.", file=sys.stderr)
        sys.exit(1)

    target_levels = []
    if args.debug: target_levels.append(10)
    if args.info: target_levels.append(20)
    if args.success: target_levels.append(25)
    if args.warning: target_levels.append(30)
    if args.error: target_levels.append(40)
    if args.critical: target_levels.append(50)

    rust_levels = target_levels if target_levels else None

    should_colorize = (
        not args.no_color
        and sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
    )

    start_ts = None
    if args.days or args.hours or args.minutes:
        delta = timedelta(
            days=args.days or 0,
            hours=args.hours or 0,
            minutes=args.minutes or 0
        )
        start_ts = (datetime.now() - delta).timestamp()

    # Pass args.file as is.
    # If args.file == None, Rust will trigger recursive search.
    file_pattern = args.file 
    
    if not file_pattern and not args.json:
        print(f"🔍 No file specified. Scanning directory recursively...", file=sys.stderr)

    try:
        lumina_core.read_binary_log(
            file_pattern,     # <-- Pass (possibly None)
            args.min_level,   
            args.trace,       
            args.json,        
            args.follow,      
            rust_levels,      
            args.grep,        
            args.tracetype,   
            start_ts,
            args.theme,
            args.export_json, 
            not should_colorize
        )
        
        if args.export_json:
            print(f"💾 JSON exported to: {args.export_json}", file=sys.stderr)

    except Exception as e:
        if args.json:
            import json
            print(json.dumps({"lvl": "ERROR", "msg": f"Lumina CLI Error: {e}"}))
        else:
            print(f"Error reading log: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()