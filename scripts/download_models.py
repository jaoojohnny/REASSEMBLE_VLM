"""第 5 包：下載模型權重（Hugging Face）。用法：
  python -m scripts.download_models --plan      # 只列出大小與授權
  python -m scripts.download_models             # 下載 smolvlm2、qwen3vl_4b、qwen3vl_8b
"""

from assembly_monitor.cli import download_models_main

if __name__ == "__main__":
    raise SystemExit(download_models_main())
