# Docker 環境與 GitHub repo

建立：2026-11-09。repo：私人（Private）GitHub repo `REASSEMBLE_VLM`，用 `push_github.bat` 上傳。

## repo 裡有什麼、沒有什麼

| 有（約 600 個檔、30 MB） | 沒有（留在本機，`.gitignore`／`.dockerignore`） |
|---|---|
| `src/assembly_monitor/`（全部程式）、`scripts/`、`configs/`（含凍結設定與提示詞）、`tests/` | `.env`（API key） |
| Windows 一鍵檔 `*.bat`、`requirements*.txt`、`environment.yml`、`pyproject.toml` | `data/raw/`（REASSEMBLE HDF5、REBOOT 影片）、`data/derived/`（影格）、`data/slim/` |
| `docs/*.md`（各包說明）、`README.md` | `models/`（Hugging Face 權重）、`experiments/`（結果）、`logs/` |
| `data/annotations/`（人工標註）、`data/index/`（段索引）、`data/splits/`（凍結的切分與雜湊） | `docs/*.pptx`、`docs/*.pdf`（進度報告，有資料集影格） |
| `Dockerfile`、`docker-compose.yml`、`.dockerignore` | |

上傳前 `scripts/repo_check.py` 會再檢查一次：有 API key、原始資料、模型權重、實驗結果、簡報／PDF 或超過 50 MB 的檔就停，不 commit、不上傳。

## 上傳／更新 GitHub（Windows）

1. 需要 Git for Windows（https://git-scm.com/download/win，預設選項即可；內含 GitHub 登入工具）。
2. 雙擊 `push_github.bat`：
   - 有裝 GitHub CLI（`gh`）且已登入 → 自動建立私人 repo `REASSEMBLE_VLM` 並上傳。
   - 沒有 → 它會打開 github.com/new：名稱填 `REASSEMBLE_VLM`、選 Private、**不要**勾 README／.gitignore／license，建立後把 HTTPS 網址貼回視窗。第一次上傳會跳出瀏覽器登入 GitHub。
3. 之後有改動，再雙擊一次就只上傳改過的檔案。

## Docker

映像：`python:3.11-slim`＋CUDA 12.6 版 torch 2.14.1（和本機 conda 環境相同）＋`requirements*.txt`＋本專案（可編輯模式）。
資料、模型權重、結果、`.env` 都不進映像，執行時用 volume 掛進去（`docker-compose.yml`）。

需要：Docker（Windows 用 Docker Desktop＋WSL2）；要用 GPU 還需要 NVIDIA 驅動（Linux 另需 NVIDIA Container Toolkit）。
第一次建映像約下載 8–10 GB。

```bash
docker compose build                       # 建映像 reassemble-vlm:latest
docker compose run --rm vlm                # 單元測試（不需要 GPU、資料、模型）
docker compose run --rm vlm python -m scripts.run_vlm --check   # 檢查 GPU 與套件
```

Windows 也可以雙擊 `docker_build_test.bat`（建映像 → 在容器裡跑測試 → 檢查看不看得到 GPU）。

### 在容器裡跑實驗

`.bat` 只能在 Windows 用；容器裡直接跑 `.bat` 裡面的 `python -m ...` 指令，例如：

```bash
# REBOOT 圓柱開發集（第 13 包）
docker compose run --rm vlm python -m scripts.reboot_cyl all
docker compose run --rm vlm python -m scripts.reboot_seg
docker compose run --rm vlm python -m scripts.reboot_run cam-low
docker compose run --rm vlm python -m scripts.reboot_run prepare
docker compose run --rm vlm python -m scripts.run_vlm --experiment configs/experiment_reboot_cyl_gpt.yaml \
    --models configs/model.yaml configs/model_p8.yaml configs/model_skill2.yaml configs/model_reason.yaml
docker compose run --rm vlm python -m scripts.reboot_run evaluate
```

- API key：把 `.env.example` 複製成 `.env` 填好，compose 會自動讀；不會進映像也不會進 Git。
- REASSEMBLE 原始資料：自行從 TU Wien Research Data 下載（CC BY 4.0，請引用原論文）放到 `data/raw/`。
- 模型權重：第一次用到時從 Hugging Face 下載到 `models/hf/`（掛在電腦上，下次不用重抓）。
- 想直接改程式不重建映像：在 `docker-compose.yml` 的 volumes 加 `- .:/workspace`。
