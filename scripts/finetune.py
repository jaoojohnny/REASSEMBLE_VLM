"""第 7 包：Qwen3-VL-8B 的 QLoRA 微調。用法：
  python -m scripts.finetune plan     # 列出訓練資料（其他零件）與一個樣本的提示詞，不用 GPU
  python -m scripts.finetune smoke    # 幾段試跑：確認顯存與速度，估計正式訓練時間
  python -m scripts.finetune train    # 正式訓練（adapter 存在 models/ft/<name>/final）
"""

from assembly_monitor.cli import finetune_main

if __name__ == "__main__":
    raise SystemExit(finetune_main())
