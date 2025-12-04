use clap::Parser;
use lumina_core::reader::{run_reader_impl, ReaderConfig};
use std::process;
use chrono::{Local, Duration};
use std::io::IsTerminal;

/// Command-line tool for reading, filtering, and exporting Lumina binary logs (.ldb).
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to .ldb log file, glob pattern (logs/*.ldb), OR directory (recursive search).
    /// If omitted, defaults to scanning the current directory.
    #[arg(value_name = "FILE_OR_DIR")]
    file: Option<String>,

    /// Show DEBUG (10) level logs.
    #[arg(short, long)]
    debug: bool,

    /// Show INFO (20) level logs.
    #[arg(short, long)]
    info: bool,

    /// Show SUCCESS (25) level logs.
    #[arg(short, long)]
    success: bool,

    /// Show WARNING (30) level logs.
    #[arg(short = 'w', long)]
    warning: bool,

    /// Show ERROR (40) level logs.
    #[arg(short, long)]
    error: bool,

    /// Show CRITICAL (50) level logs.
    #[arg(short, long)]
    critical: bool,

    /// Filter logs by minimum numeric level (e.g., 20 for INFO+).
    #[arg(short = 'l', long = "min-level", default_value_t = 0)]
    min_level: u8,

    /// Filter logs containing specific text (case-insensitive substring match).
    #[arg(short, long)]
    grep: Option<String>,

    /// Filter logs by exception type (e.g., "ValueError").
    #[arg(short = 't', long = "tracetype")]
    tracetype: Option<String>,

    /// Display full Python tracebacks for errors.
    #[arg(long)]
    trace: bool,

    /// Output logs in JSONL format to stdout (useful for piping to jq).
    #[arg(long)]
    json: bool,

    /// Live tail mode: follow new log entries as they are written (like tail -f).
    #[arg(short, long)]
    follow: bool,

    /// Export filtered results to a JSONL file.
    #[arg(long)]
    export_json: Option<String>,

    /// Filter logs from the last N days.
    #[arg(short = 'D', long)]
    days: Option<i64>,

    /// Filter logs from the last N hours.
    #[arg(short = 'H', long)]
    hours: Option<i64>,

    /// Filter logs from the last N minutes.
    #[arg(short = 'm', long)]
    minutes: Option<i64>,

    /// UI Color Theme (dark/light).
    #[arg(long, default_value = "dark")]
    theme: String,

    /// Disable colored output explicitly.
    #[arg(long)]
    no_color: bool,
}

fn main() {
    let args = Args::parse();

    // Detect if stdout is a terminal to enable/disable colors automatically
    let is_tty = std::io::stdout().is_terminal();

    let should_colorize = is_tty 
                                && !args.no_color
                                && std::env::var("NO_COLOR").is_err();

    lumina_core::utils::set_colors_enabled(should_colorize);

    // Build specific level whitelist based on flags
    let mut target_levels = Vec::new();
    if args.debug { target_levels.push(10); }
    if args.info { target_levels.push(20); }
    if args.success { target_levels.push(25); }
    if args.warning { target_levels.push(30); }
    if args.error { target_levels.push(40); }
    if args.critical { target_levels.push(50); }

    let target_levels = if target_levels.is_empty() { None } 
    else { Some(target_levels) };

    // Calculate start timestamp based on time flags
    let mut start_ts = None;
    if args.days.is_some() || args.hours.is_some() || args.minutes.is_some() {
        let now = Local::now();
        let delta = Duration::days(args.days.unwrap_or(0)) 
                  + Duration::hours(args.hours.unwrap_or(0))
                  + Duration::minutes(args.minutes.unwrap_or(0));
        let start = now - delta;
        start_ts = Some(start.timestamp() as f64);
    }

    use std::path::{Path, PathBuf};
    use std::fs;

    // Helper to resolve export paths.
    // If only a filename is given, it saves to "logs/exports/".
    // If a path is given, it ensures directories exist.
    let process_path = |p: Option<String>| -> Option<String> {
        match p {
            Some(filename) => {
                let path = Path::new(&filename);
                // If the file has no parent directory (just a filename)
                if path.parent().map_or(true, |p| p.as_os_str().is_empty()) {
                    let export_dir = "logs/exports";
                    let _ = fs::create_dir_all(export_dir); // Create dir, ignore error if exists
                    
                    // Join path
                    let mut new_path = PathBuf::from(export_dir);
                    new_path.push(filename);
                    Some(new_path.to_string_lossy().to_string())
                } else {
                    // User provided a path, ensure parent dir exists
                    if let Some(parent) = path.parent() {
                         let _ = fs::create_dir_all(parent);
                    }
                    Some(filename)
                }
            },
            None => None
        }
    };

    let final_json_path = process_path(args.export_json);

    let config = ReaderConfig {
        file_pattern: args.file,
        min_level: args.min_level,
        show_trace: args.trace,
        json_output: args.json,
        follow: args.follow,
        target_levels,
        grep: args.grep,
        trace_type: args.tracetype,
        start_ts,
        theme: Some(args.theme),
        export_json: final_json_path,
    };

    // Run the Core Reader
    if let Err(e) = run_reader_impl(config) {
        eprintln!("❌ Error: {}", e);
        process::exit(1);
    }
}