use std::collections::HashMap;
use indexmap::IndexMap; 
use serde::{Serialize, Deserialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "light" => Theme::Light,
            _ => Theme::Dark, 
        }
    }
}

pub type CallerCache = HashMap<(String, u32), String>;
pub type FileCache = HashMap<String, Vec<String>>;

/// Represents a single stack frame for traceback visualization.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash)]
pub struct TraceFrame {
    pub filename: String,
    pub lineno: u32,
    pub name: String,
}

/// Structure for storing raw trace data for serialization.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RawTraceback {
    pub exc_type: String,
    pub exc_message: String,
    pub frames: Vec<TraceFrame>,
}

/// Internal log entry passed through the channel from Python to the Writer thread.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub app_name: Arc<str>,
    pub timestamp: f64,
    pub level: u8,
    pub message: String,
    pub path_template: String,
    pub to_console: bool,
    pub to_file: bool,
    pub rate_limit: f64,
    
    // Exception handling
    pub exc_type: Option<String>,
    pub exc_message: Option<String>,
    pub exc_hash: u64,
    pub trace_frames: Option<Vec<TraceFrame>>,
    
    // Context data (structured logging)
    pub context: Option<IndexMap<String, String>>, 
    pub context_hash: u64,
    
    // Control flags
    pub signal_shutdown: bool,
}

/// Compact record structure for binary storage on disk (Bincode/LZ4).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BinaryLogRecord {
    pub ts: f64,
    pub lvl: u8,
    pub app_name: String,
    pub pid: u32,
    pub msg: String,
    pub traceback: Option<RawTraceback>, 
    pub context: Option<IndexMap<String, String>>,
    pub count: u32, // For deduplication
}

/// Structure for JSONL export / stdout.
#[derive(Serialize)]
pub struct JsonLogRecord {
    pub ts: f64,
    pub app: String,
    pub pid: u32,
    pub lvl: String,
    pub msg: String,
    pub count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<IndexMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}