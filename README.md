# Lumina ✨

***High-Performance, Structured, Binary Logging for Python (Rust Core)***

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lumina)](https://pypi.org/project/lumina/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

> ⚠️ **Status: Alpha / Experimental**
> This project is currently in early development. APIs may change, and pre-built wheels are not yet available on PyPI.

---

Lumina is a hybrid logging library for Python designed for high-throughput and low-latency applications. It achieves this by offloading all heavy lifting—serialization, compression, and I/O—to a highly parallelized Rust core, ensuring your main application thread remains unblocked.

It uses a custom binary format (`.ldb`) for efficient, structured storage and includes a powerful CLI tool for real-time and historical log analysis.

## Key Features

* 🚀 **Extremely Low Overhead:** Python calls return in microseconds by sending logs to a lock-free in-memory channel.
* ⚙️ **Multi-Core Architecture:** A pool of Rust workers processes logs in parallel, handling compression and serialization across all available CPU cores. A dedicated I/O thread ensures non-blocking disk writes.
* 📦 **Efficient Binary Format (.ldb):** Logs are serialized with `Bincode` and compressed with `LZ4` for fast writes and reads.
* 🔍 **Structured Data:** Pass keyword arguments (`kwargs`) to your logs. They are stored as structured, searchable fields, not just plain text.
* ⏱️ **Advanced Resource Profiling:** A built-in context manager tracks Wall Time, CPU Time, RAM Deltas, and even low-level OS metrics like Page Faults and Context Switches.
* 🖥️ **Powerful CLI Reader:** A standalone tool (`lumina`) to decode, filter, merge, and live-tail log files from multiple sources in chronological order.
* 🤝 **Standard Library Integration:** Can seamlessly intercept calls from Python's standard `logging` module, speeding up your entire application and its dependencies.

---

## 🚀 Quick Start (Local Development)

Since this is an alpha build, you need to compile it from source.

### Prerequisites

* Python 3.12+
* **Rust Toolchain:** You must have `cargo` and `rustc` installed. (See [rustup.rs](https://rustup.rs))

### 🐧 Linux / macOS

1. **Clone and Setup Environment:**

    ```bash
    git clone https://github.com/JustAMamont/lumina_project
    cd lumina_project
    python3 -m venv env
    source env/bin/activate
    ```

2. **Install Build Tools:**

    ```bash
    pip install -U pip
    pip install maturin
    ```

3. **Compile and Install in Editable Mode:**

    ```bash
    maturin develop --release
    ```

### 🪟 Windows (PowerShell)

1. **Clone and Setup Environment:**

    ```powershell
    git clone https://github.com/JustAMamont/lumina_project
    cd lumina_project
    python -m venv env
    .\env\Scripts\Activate
    ```

2. **Install Build Tools:**

    ```powershell
    pip install -U pip
    pip install maturin
    ```

3. **Compile and Install in Editable Mode:**

    ```powershell
    maturin develop --release
    ```

### ✅ Verification

Run the demo script to see Lumina in action:

```bash
python examples/demo.py
```

---

### ⚡ Performance Benchmark

Lumina is designed to keep your main application thread responsive. The included benchmark highlights the difference, especially under multi-threaded contention.

```bash
python examples/benchmark.py
```

You will see how Lumina maintains stable, sub-10 microsecond latencies even when multiple threads are logging heavily, while standard logging performance degrades significantly.

---

## 🧪 Running Tests

Lumina uses `pytest` for integration testing.

1. **Install Test Dependencies:**

    ```bash
    pip install pytest
    ```

2. **Run Tests:**
    **Important:** Always rebuild the Rust extension before running tests if you've changed any Rust code.

    ```bash
    maturin develop --release
    pytest -v
    ```

---

## 📦 Building for Distribution

### Building the Python Wheel (.whl)

Creates a package that can be installed on other machines of the same architecture, even those without Rust installed.

```bash
# For a specific Python version
maturin build --release --interpreter python3.12

# For cross-compilation (e.g., building for Linux on macOS/Windows)
# Requires Docker for manylinux builds or Zig for simpler cases
pip install ziglang
maturin build --release --zig
```

*Artifacts are located in `target/wheels/`.*

### Building the Standalone Reader Binary

Compiles the `lumina` CLI tool as a single, standalone executable that requires no Python or other dependencies.

```bash
cargo build --release --bin lumina
```

*The executable is located at `target/release/lumina` (Linux/Mac) or `target/release/lumina.exe` (Windows).*

---

## 🐍 Python Usage

```python
from lumina import Lumina
import time

# Initialize the logger (singleton)
lum = Lumina.get_logger(
    name="PaymentService",
    path_template="logs/{date}.ldb",
    flush_interval_ms=500,
    capture_caller=True # Shows file:line, useful for debugging
)

# Basic log
lum.info("Service started on port 8000")

# Structured log with context
lum.info("Transaction processed", user_id=42, amount=99.99, currency="USD")

# Different log levels
lum.success("Database connection established")
lum.warning("High latency detected", peer="10.0.0.5", latency="150ms")
lum.error("Payment provider API failed", provider="Stripe", status_code=503)

# Log exceptions with full traceback
try:
    result = 1 / 0
except Exception as e:
    lum.critical("Fatal error in calculation", exc=e, details="This should not happen")

# Profile a block of code
with lum.profile("Data Processing Task", source="kafka"):
    # Simulate heavy work
    time.sleep(0.1)
    _ = [i*i for i in range(10_000)]

# Explicitly shutdown (optional, done automatically on exit)
lum.shutdown()
```

---

## 🖥️ CLI Tool (`lumina`)

Use the CLI to read, filter, and export your binary logs.

### Basic Reading & Filtering

```bash
# Read all logs found in the current directory (recursive scan)
lumina

# Follow logs live (like tail -f)
lumina -f

# Filter by minimum level (e.g., only WARNING and above)
lumina --min-level 30

# Use convenient level flags
lumina -wec  # Show warnings, errors, and criticals

# Search for text anywhere in the log (case-insensitive grep)
lumina --grep "user_id=42"
```

### 📤 Exporting to JSON Lines

Export logs for external analysis (ELK, Datadog, jq, etc.).

#### Method 1: Pipe to Stdout

Ideal for chaining commands with tools like `jq`.

```bash
lumina --json --days 1 | jq '. | select(.context.amount > 100)' > large_transactions.jsonl
```

#### Method 2: Direct File Export

Convenient for saving query results directly. The path handling is smart:

1. **Filename only:** Saves to the default `logs/exports/` directory.

    ```bash
    # Resulting file: ./logs/exports/yesterday.jsonl
    lumina --days 1 --export-json yesterday.jsonl
    ```

2. **Full or Relative Path:** Respects the provided path and creates directories if needed.

    ```bash
    # Resulting file: /tmp/data/prod-errors.jsonl
    lumina -e --export-json /tmp/data/prod-errors.jsonl
    ```

---

## ⚙️ Configuration Options

Key parameters for `Lumina.get_logger()`:

| Parameter           | Default           | Description                                                                                              |
| :------------------ | :---------------- | :------------------------------------------------------------------------------------------------------- |
| `name`              | *script name*     | The name of your application, appears in logs.                                                           |
| `path_template`     | `"logs/{date}.ldb"` | File path pattern for log files. `{date}` is automatically replaced.                                     |
| `retention_days`    | `7`               | Automatically delete log files older than N days on startup.                                             |
| `channel_capacity`  | `50,000`          | The in-memory buffer size. A larger buffer handles bigger spikes in log volume before dropping messages. |
| `flush_interval_ms` | `100`             | How often the background thread forces a write to disk, even under low load.                             |
| `capture_caller`    | `False`           | If `True`, resolves the `file:line` of every log call. Adds a small performance overhead.                |
| `text_enabled`      | `False`           | If `True`, writes a duplicate, human-readable `.log` file alongside the binary `.ldb` file.              |
| `db_enabled`        | `True`            | If `False`, disables writing to the binary `.ldb` file.                                                  |

## License

This project is licensed under the MIT License.
