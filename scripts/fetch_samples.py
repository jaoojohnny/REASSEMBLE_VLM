"""只取 1–3 個 REASSEMBLE 樣本。用法：python -m scripts.fetch_samples [--n 3] [--list] [--zip-path data.zip]"""

from assembly_monitor.cli import fetch_main

if __name__ == "__main__":
    raise SystemExit(fetch_main())
