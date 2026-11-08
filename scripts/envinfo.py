"""記錄執行環境與 GPU。用法：python -m scripts.envinfo --out logs"""

from assembly_monitor.cli import envinfo_main

if __name__ == "__main__":
    raise SystemExit(envinfo_main())
