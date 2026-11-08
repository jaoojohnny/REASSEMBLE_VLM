"""人工核對短片。用法：python -m scripts.export_check_clips [--n 5] [--pad 1.5] [--cameras hama1,hand]"""

from assembly_monitor.cli import check_clips_main

if __name__ == "__main__":
    raise SystemExit(check_clips_main())
