use chrono::{DateTime, Local};
use colored::*;
use regex::Regex;
use once_cell::sync::Lazy;
use std::sync::RwLock;
use crate::types::Theme;
use crossterm::terminal::size;
use std::sync::atomic::{AtomicBool, Ordering};

/// Regex to match custom style tags in the format `[text|style1+style2]`.
static STYLE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\[([^|\]]+)\|([^\]]+)\]").unwrap()
});

/// Regex to match ANSI escape codes. Used to strip colors when calculating string length.
static ANSI_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\x1b\[[0-9;]*m").unwrap()
});

/// Used for JSON export or when colors are disabled.
pub fn remove_style_tags(text: &str) -> String {
    if !text.contains('[') { return text.to_string(); }
    STYLE_REGEX.replace_all(text, "$1").to_string()
}

/// Thread-safe storage for the current UI theme (Dark/Light).
pub static CURRENT_THEME: Lazy<RwLock<Theme>> = Lazy::new(|| RwLock::new(Theme::Dark));

/// Updates the global theme.
pub fn set_theme(theme: Theme) {
    if let Ok(mut w) = CURRENT_THEME.write() {
        *w = theme;
    }
}

/// Global flag to enable/disable colored output.
pub static COLORS_ENABLED: AtomicBool = AtomicBool::new(true);

/// Sets the global color flag and configures the `colored` crate.
pub fn set_colors_enabled(val: bool) {
    COLORS_ENABLED.store(val, Ordering::Relaxed);
    colored::control::set_override(val);
}

/// High-performance time formatter.
pub fn get_time_parts<'a>(ts: f64, cached_sec: &mut i64, cached_prefix: &'a mut String) -> (&'a str, u32) {
    let secs = ts as i64;
    let nsecs = ((ts - secs as f64) * 1_000.0) as u32;
    if secs != *cached_sec {
        if let Some(dt) = DateTime::from_timestamp(secs, 0) {
            *cached_prefix = dt.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S").to_string();
            *cached_sec = secs;
        }
    }
    (cached_prefix.as_str(), nsecs)
}

/// Parses the string for `[text|style]` tags and applies ANSI colors.
pub fn fast_colorize(text: &str) -> String {
    if !text.contains('[') { return text.to_string(); }

    if !COLORS_ENABLED.load(Ordering::Relaxed) {
        return  STYLE_REGEX.replace_all(text, "$1").to_string();
    }
    
    let is_light = {
        if let Ok(r) = CURRENT_THEME.read() { *r == Theme::Light } else { false }
    };

    STYLE_REGEX.replace_all(text, |caps: &regex::Captures| {
        let content = caps.get(1).unwrap().as_str();
        let styles_str = caps.get(2).unwrap().as_str();
        let mut painted = content.normal();
        
        for style in styles_str.split(|c| c == '+' || c == ',' || c == ' ') {
            painted = match style.trim().to_lowercase().as_str() {
                "r"|"red" => painted.red(),
                "g"|"green" => painted.green(),
                "b"|"blue" => painted.blue(),
                "y"|"yellow" => if is_light { painted.yellow() } else { painted.yellow() },
                "c"|"cyan" => if is_light { painted.cyan().dimmed() } else { painted.cyan() },
                "m"|"magenta" => painted.magenta(),
                "w"|"white" => if is_light { painted.normal() } else { painted.white() },
                "bd"|"bold" => painted.bold(),
                "u"|"under" => painted.underline(),
                "d"|"dim" => painted.dimmed(),
                "hl" => painted.black().on_yellow(),
                "alert" => painted.white().on_red().bold(),
                _ => painted,
            };
        } 
        painted.to_string()
    }).to_string()
}

/// Calculates the actual visual height of the text in the terminal.
pub fn measure_text_height(text: &str) -> usize {
    let term_width = if let Ok((w, _)) = size() { w as usize } else { 80 };
    if term_width == 0 { return 1; }

    let mut lines_count = 0;
    
    for line in text.lines() {
        let clean_line = ANSI_REGEX.replace_all(line, "");
        let len = clean_line.chars().count(); 
        
        if len == 0 {
            lines_count += 1;
        } else {
            lines_count += (len + term_width - 1) / term_width;
        }
    }
    
    if lines_count == 0 { 1 } else { lines_count }
}

/// Returns the string label, colored representation, and icon for a numeric log level.
pub fn get_level_meta(level: u8) -> (&'static str, ColoredString, &'static str) {
    if !COLORS_ENABLED.load(Ordering::Relaxed) {
        return match level {
            10 => ("DEBUG", "DEBUG".normal(), "🐛"),
            20 => ("INFO", "INFO".normal(), "ℹ️ "),
            25 => ("SUCCESS", "SUCCESS".normal(), "✅"),
            30 => ("WARNING", "WARN".normal(), "⚠️ "),
            40 => ("ERROR", "ERROR".normal(), "❌"),
            50 => ("CRITICAL", "CRIT".normal(), "🔥"),
            _  => ("LOG", "LOG".normal(), "📝"),
        }; 
    }

    let is_light = {
        if let Ok(r) = CURRENT_THEME.read() { *r == Theme::Light } else { false }
    };

    match level {
        10 => {
            if is_light { ("DEBUG", "DEBUG".purple(), "🐛") }
            else { ("DEBUG", "DEBUG".dimmed(), "🐛") }
        },
        20 => ("INFO", "INFO".blue().bold(), "ℹ️ "),
        25 => ("SUCCESS", "SUCCESS".green().bold(), "✅"),
        30 => {
            if is_light { ("WARNING", "WARN".red(), "⚠️ ") }
            else { ("WARNING", "WARN".yellow().bold(), "⚠️ ") }
        },
        40 => ("ERROR", "ERROR".red().bold(), "❌"),
        50 => ("CRITICAL", "CRIT".on_red().white().bold(), "🔥"),
        _  => {
            if is_light { ("LOG", "LOG".black(), "📝") }
            else { ("LOG", "LOG".normal(), "📝") }
        },
    }
}

/// Escapes control characters to prevent terminal corruption.
pub fn sanitize_input(text: &str) -> String {
    let s = text.replace('\x1b', "^[");
    s.replace('\r', "\\r")
}