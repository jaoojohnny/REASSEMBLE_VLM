"""HDF5 結構檢查。用法：python -m scripts.inspect_h5 data/raw [--expect-cameras a,b]"""

from assembly_monitor.cli import inspect_main

if __name__ == "__main__":
    raise SystemExit(inspect_main())
