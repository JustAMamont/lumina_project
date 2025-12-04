import json
import sys
import time
from lumina import Lumina, lumina_core # type: ignore

try:
    import pytest
except ImportError:
    print("❌ Pytest not installed", file=sys.stderr)
    sys.exit(1)

# === FIXTURES ===

@pytest.fixture(autouse=True)
def reset_lumina_state():
    """
    Resets the Lumina singleton state before and after each test.
    Ensures isolation between tests.
    """
    if Lumina._instance:
        try:
            Lumina._instance.shutdown()
        except:
            pass
    Lumina._instance = None
    yield
    if Lumina._instance:
        try:
            Lumina._instance.shutdown()
        except:
            pass
    Lumina._instance = None

@pytest.fixture
def log_dir(tmp_path):
    """Creates a temporary directory for test logs."""
    path = tmp_path / "logs"
    path.mkdir()
    yield path

def filter_system_logs(lines):
    """
    Filters out internal Rust worker start/stop messages.
    Returns only user-generated logs.
    """
    filtered = []
    for line in lines:
        try:
            if not line.strip(): continue
            data = json.loads(line)
            msg = data.get("msg", "")
            # Ignore worker lifecycle messages
            if "🚀 Process started" in msg or "🛑 Process finished" in msg:
                continue
            filtered.append(data)
        except Exception as e:
            print(f"Error parsing line: {line} -> {e}")
            pass
    return filtered

def run_reader(log_dir, **kwargs):
    """
    Helper to invoke the binary log reader with default test arguments.
    """
    pattern = str(log_dir / "*.ldb")
    # Default arguments matching Rust signature
    args = {
        "file_pattern_arg": pattern,
        "min_level": 0,
        "show_trace": False,
        "json_output": True,
        "target_levels": None,
        "grep": None,
        "trace_type": None,
        "start_ts": None,
    }
    # Override defaults
    args.update(kwargs)
    
    lumina_core.read_binary_log(**args)

# === TESTS ===

def test_basic_logging(log_dir, capfd):
    """Basic test: verifies log writing and reading integrity."""
    logger = Lumina.get_logger(path_template=str(log_dir / "base_{date}.ldb"), text_enabled=False)
    
    logger.info("Message 1", console=False)
    logger.error("Message 2", console=False)
    time.sleep(0.1)
    logger.shutdown()
    
    run_reader(log_dir)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    assert len(logs) == 2
    assert logs[0]['msg'] == "Message 1"
    assert logs[0]['lvl'] == "INFO"
    assert logs[1]['msg'] == "Message 2"
    assert logs[1]['lvl'] == "ERROR"

def test_structured_context(log_dir, capfd):
    """
    Verifies that kwargs are correctly stored in the 'context' JSON field.
    """
    logger = Lumina.get_logger(path_template=str(log_dir / "ctx_{date}.ldb"), text_enabled=False)
    
    # Pass structured data
    logger.info("User Action", user_id=42, role="admin", is_active=True, console=False)
    
    time.sleep(0.1)
    logger.shutdown()
    
    run_reader(log_dir)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    assert len(logs) == 1
    
    entry = logs[0]
    assert entry['msg'] == "User Action"
    assert 'context' in entry, "Context field missing in JSON"
    
    ctx = entry['context']
    # Rust stores context as HashMap<String, String>
    assert ctx['user_id'] == "42"
    assert ctx['role'] == "admin"
    assert ctx['is_active'] == "True"

def test_exception_handling(log_dir, capfd):
    """
    Verifies that exceptions passed via 'exc' are serialized with tracebacks.
    """
    logger = Lumina.get_logger(path_template=str(log_dir / "exc_{date}.ldb"), text_enabled=False)
    
    try:
        x = 1 / 0
    except ZeroDivisionError as e:
        logger.error("Calculation failed", exc=e, console=False)
        
    time.sleep(0.1)
    logger.shutdown()
    
    run_reader(log_dir)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    assert len(logs) == 1
    
    entry = logs[0]
    assert entry['msg'] == "Calculation failed"
    assert 'exc' in entry
    
    exception_text = entry['exc']
    assert "ZeroDivisionError" in exception_text
    assert "division by zero" in exception_text
    assert "Traceback" in exception_text
    # Ensure current file is mentioned in traceback
    assert "test_integration.py" in exception_text

def test_profiler(log_dir, capfd):
    """
    Verifies the 'profile' context manager generates metrics (time, ram, cpu).
    """
    logger = Lumina.get_logger(path_template=str(log_dir / "prof_{date}.ldb"), text_enabled=False)
    
    with logger.profile("HeavyTask", console=False):
        # Simulate work
        _ = [i for i in range(1000)]
        time.sleep(0.05)
        
    time.sleep(0.1)
    logger.shutdown()
    
    run_reader(log_dir)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    assert len(logs) == 1
    
    entry = logs[0]
    assert "Profile: HeavyTask" in entry['msg']
    assert 'context' in entry
    
    ctx = entry['context']
    assert 'time' in ctx
    assert 'ram' in ctx
    assert 'cpu' in ctx
    
    # Verify timing accuracy
    time_val = float(ctx['time'].replace('s', ''))
    assert time_val >= 0.05

def test_reader_filtering(log_dir, capfd):
    """
    Verifies Reader-side filtering (Rust):
    1. min_level
    2. grep (text search)
    """
    logger = Lumina.get_logger(path_template=str(log_dir / "filter_{date}.ldb"), text_enabled=False)
    
    logger.debug("Debug Msg", console=False)      # Lvl 10
    logger.info("Info Apple", console=False)      # Lvl 20
    logger.warning("Warn Banana", console=False)  # Lvl 30
    logger.error("Error Apple", console=False)    # Lvl 40
    
    time.sleep(0.2)
    logger.shutdown()
    
    # --- Case 1: Filter by Level (>= 30 Warning) ---
    run_reader(log_dir, min_level=30)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    # Expect only Warning and Error
    assert len(logs) == 2
    assert logs[0]['msg'] == "Warn Banana"
    assert logs[1]['msg'] == "Error Apple"
    
    # --- Case 2: Filter by Grep ("Apple") ---
    run_reader(log_dir, grep="Apple")
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    # Expect Info Apple and Error Apple
    assert len(logs) == 2
    for l in logs:
        assert "Apple" in l['msg']

def test_deduplication(log_dir, capfd):
    """
    Verifies deduplication of sequential identical messages.
    """
    logger = Lumina.get_logger(path_template=str(log_dir / "dedup_{date}.ldb"), text_enabled=False)
    
    for _ in range(5):
        logger.warning("Repeated", console=False)
        
    time.sleep(0.1)
    logger.shutdown()
    
    run_reader(log_dir)
    
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    
    # Should be collapsed into 1 log
    assert len(logs) == 1
    assert logs[0]['msg'] == "Repeated"
    if 'count' in logs[0]:
        assert logs[0]['count'] == 5

def test_scatter_gather_order(log_dir, capfd):
    """
    Verifies that the reader correctly merges logs from different files
    in chronological order (Scatter-Gather pattern).
    """
    # Simulate Process 1 -> File 1
    l1 = Lumina.get_logger(path_template=str(log_dir / "proc1_{date}.ldb"), text_enabled=False)
    l1.info("Step 1", console=False)
    l1.shutdown()
    Lumina._instance = None # Reset singleton to switch files
    time.sleep(0.1) # Ensure timestamp gap
    
    # Simulate Process 2 -> File 2
    l2 = Lumina.get_logger(path_template=str(log_dir / "proc2_{date}.ldb"), text_enabled=False)
    l2.info("Step 2", console=False)
    l2.shutdown()
    Lumina._instance = None
    time.sleep(0.1)
    
    # Simulate Process 3 -> File 3
    l3 = Lumina.get_logger(path_template=str(log_dir / "proc3_{date}.ldb"), text_enabled=False)
    l3.info("Step 3", console=False)
    l3.shutdown()
    
    # Read all .ldb files in the directory
    run_reader(log_dir)
    logs = filter_system_logs(capfd.readouterr().out.strip().split('\n'))
    
    # Verify strict chronological order
    assert len(logs) == 3, f"Expected 3 logs, got {len(logs)}: {logs}"
    assert logs[0]['msg'] == "Step 1"
    assert logs[1]['msg'] == "Step 2"
    assert logs[2]['msg'] == "Step 3"