"""第 5 包 VLM 試跑。用法：
  python -m scripts.run_vlm --check                                  # 檢查 GPU 與套件
  python -m scripts.run_vlm                                          # 開發集 × majority/smolvlm2/qwen3vl_4b/qwen3vl_8b × 1/4/8 幀
  python -m scripts.run_vlm --use qwen3vl_8b --frames 8 --limit 3    # 小樣本
  python -m scripts.run_vlm --use mock --limit 2 --name deploy_check # 不載入模型，只檢查流程
"""

from assembly_monitor.cli import run_vlm_main

if __name__ == "__main__":
    raise SystemExit(run_vlm_main())
