# Lumina

***Structured, Binary Logging for Python (Rust Backend)**

> ⚠️ **Status: Alpha / Experimental**
> This project is currently in early development. APIs may change, and pre-built wheels are not yet available on PyPI.

---

Lumina is a hybrid logging library for Python. It attempts to reduce logging overhead by offloading I/O, serialization, and compression to a background Rust thread. It uses a custom binary format to store logs efficiently and includes a CLI tool for reading and merging log files.

## Features

* **Background Worker:** Logging operations are sent to a Rust thread via a channel, minimizing blocking in the Python main thread.
* **Binary Format (.ldb):** Logs are serialized with `Bincode` and compressed with `LZ4`.
* **Structured Data:** Supports passing keyword arguments (`kwargs`) which are stored as searchable fields.
* **CLI Reader:** A dedicated tool (`lumina-read`) to decode, filter, and merge log files.
* **Resource Profiling:** Context manager to track CPU usage, RAM deltas, and Page Faults.
* **Compatibility:** Can intercept standard Python `logging` calls.

---

## 🚀 Quick Start (Local Development)

Since this is an alpha build, you need to compile it from source. Choose your OS below.

### Prerequisites

* Python 3.12+
* **Rust Toolchain:** You must have `cargo` and `rustc` installed. (See [rustup.rs](https://rustup.rs))

### 🐧 Linux / macOS

1. **Clone and Setup Environment:**

    ```bash
    git clone https://github.com/JustAMamont/lumina_project
    cd lumina
    python3 -m venv env
    source env/bin/activate
    ```

2. **Install Build Tools:**

    ```bash
    pip install -U pip
    pip install maturin patchelf ziglang
    ```

3. **Compile and Install:**

    ```bash
    maturin develop --release
    ```

### 🪟 Windows (PowerShell)

1. **Clone and Setup Environment:**

    ```powershell
    git clone https://github.com/JustAMamont/lumina.git
    cd lumina
    python -m venv env
    .\env\Scripts\Activate
    ```

2. **Install Build Tools:**

    ```powershell
    pip install -U pip
    pip install maturin
    # Note: patchelf/ziglang are usually not required for local Windows dev
    ```

3. **Compile and Install:**

    ```powershell
    maturin develop --release
    ```

### ✅ Verification

Run the demo script to ensure everything is working:

```bash
python examples/demo.py
```

---

### ⚡ Performance Benchmark

Lumina is designed to unblock your main application thread.
You can run the included benchmark to compare it against the standard `logging` module:

```bash
python examples/benchmark.py
```

## 🧪 Running Tests

Lumina uses `pytest` for integration testing.

1. **Install Test Dependencies:**

    ```bash
    pip install pytest
    ```

2. **Run Tests:**
    **Important:** Always rebuild the Rust extension before running tests if you changed any Rust code.

    ```bash
    maturin develop --release
    pytest -v
    ```

---

## 📦 Building for Distribution

### Building the Python Wheel (.whl)

Creates a package installable on other machines (even without Rust).

```bash
maturin build --release
```

*Artifacts location: `target/wheels/`*

### Building the Standalone Reader Binary

Compiles `lumina-read` as a standalone executable (no Python required).

```bash
cargo build --release --bin lumina
```

*Artifact location: `target/release/lumina` (Linux/Mac) or `target/release/lumina.exe` (Windows)*

---

## 🐍 Python Usage

```python
from lumina import Lumina

# Initialize Logger
lum = Lumina.get_logger(
    name="PaymentService",
    path_template="logs/{date}.ldb",
    flush_interval_ms=500
)

# Basic Log
lum.info("Service started")

# Structured Log (Context)
lum.info("Transaction processed", user_id=42, amount=99.99, currency="USD")

# Levels
lum.success("Database connected")
lum.warning("High latency detected", latency="150ms")
lum.error("Connection failed", ip="10.0.0.5")

# Profiling
with lum.profile("Data Processing"):
    heavy_computation()

# Flush and Close
lum.shutdown()
```

---

## 🖥️ CLI Tool (`lumina-read`)

Use the CLI to read, filter, and export binary logs.

### Reading & Filtering

```bash
# Read all logs in current directory (recursive scan)
lumina-read

# Follow logs live (tail -f)
lumina-read -f

# Filter by minimum level (e.g., only ERROR and CRITICAL)
lumina-read --min-level 40

# Search for text (grep)
lumina-read --grep "database error"
```

### 📤 Exporting to JSON Lines

There are two ways to export logs for external analysis (ELK, jq, etc).

#### Method 1: Standard Output (Pipe)

Useful for chaining commands.

```bash
lumina-read --json > output.jsonl
```

#### Method 2: Internal Export Flag

Useful for saving files directly. Note the path behavior:

1. **Filename only:** Saves to default `logs/exports/` directory.

    ```bash
    # Saves to: ./logs/exports/dump.jsonl
    lumina-read --export-json dump.jsonl
    ```

2. **Full Path:** Respects the provided path and creates directories if needed.

    ```bash
    # Saves to: /tmp/my_data/custom.jsonl
    lumina-read --export-json /tmp/my_data/custom.jsonl
    ```

---

## ⚙️ Configuration Options

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `path_template` | `"logs/{date}.ldb"` | File path pattern. |
| `retention_days` | `7` | Auto-delete logs older than N days. |
| `channel_capacity` | `50,000` | Max pending logs in memory queue. |
| `flush_interval_ms` | `100` | Frequency of disk writes. |
| `capture_caller` | `False` | Resolve filename/line number (adds overhead). |
| `text_enabled` | `False` | Write duplicate logs to a standard .log text file. |

## License

MIT License.
