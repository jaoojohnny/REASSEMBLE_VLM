"""Build a local, label-free development review pack; never runs a model."""
import hashlib
import html
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from assembly_monitor.vlm import load_segment_frames
from assembly_monitor.reason_eval import predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/blind_review_dev30_v1'


def main():
    if OUT.exists():
        raise SystemExit('Output already exists; preserve completed reviews. Use a new version.')
    rng = random.Random(20261031)
    source = ROOT / 'experiments/p11_reason_dev/per_segment.jsonl'
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    results = json.loads((source.parent / 'results.json').read_text(encoding='utf-8'))
    thresholds = next(r['tau_lopo'] for r in results['rows'] if r['id'] == 'score_tau')
    chosen, used = [], set()
    for part in ('dsub', 'mgear', 'sqpeg2'):
        for truth in ('failure', 'success'):
            pool = sorted([r for r in records if r['part'] == part and r['truth'] == truth], key=lambda r: r['segment_id'])
            rng.shuffle(pool)
            pool.sort(key=lambda r: r['segment_id'].split('__')[0] in used)
            selected = []
            for r in pool:
                rid = r['segment_id'].split('__')[0]
                if rid not in used:
                    selected.append(r); used.add(rid)
                if len(selected) == 5:
                    break
            if len(selected) < 5:
                selected += [r for r in pool if r not in selected][:5-len(selected)]
            assert len(selected) == 5
            chosen.extend(selected)
    rng.shuffle(chosen)
    public, private = [], []
    for number, rec in enumerate(chosen, 1):
        case = f'B{number:02d}'
        exp = yaml.safe_load((ROOT / f'configs/experiment_rdev_{rec["part"]}_v3.yaml').read_text(encoding='utf-8'))['experiment']
        seg = load_segment_frames(ROOT / exp['frames_root'], exp['sampling_template'].format(n=4), rec['segment_id'], exp['cameras'])
        assert seg and not seg.n_missing and len(seg.frames) == 12
        folder = OUT / 'images' / case
        folder.mkdir(parents=True)
        frames = []
        for i, frame in enumerate(seg.frames, 1):
            im = cv2.imdecode(np.fromfile(frame.path, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert im is not None
            h, w = im.shape[:2]
            scale = min(1, 512 / max(h, w))
            small = cv2.resize(im, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
            path = folder / f'{i:02d}.jpg'
            ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY,95]); assert ok
            buf.tofile(str(path))
            frames.append({'path':path.relative_to(OUT).as_posix(), 'camera':frame.camera, 'time':frame.time_sec, 'id':i})
        public.append({'id':case, 'part':exp['task']['part'], 'frames':frames})
        private.append({'case_id':case, **rec, 'threshold_prediction':predict(rec,'score_tau',thresholds[rec['part']]),
                        'tau':thresholds[rec['part']], 'source_frames':[f.path for f in seg.frames]})
    # Answers and source paths never enter the public HTML or exported review template.
    (OUT / 'PRIVATE_answer_key.json').write_text(json.dumps(private,ensure_ascii=False,indent=2),encoding='utf-8')
    data = json.dumps(public,ensure_ascii=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>組裝影像盲評 30 段</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#172436;margin:24px}header{position:sticky;top:0;background:white;padding:14px;z-index:2;border-bottom:2px solid #ddd}button,select,textarea{font:inherit;padding:8px;margin:5px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}img{width:100%;cursor:zoom-in}figure{margin:0}figcaption{font-size:13px}textarea{width:95%;height:70px}label{display:block;margin:10px}dialog{max-width:95vw}dialog img{width:auto;max-width:90vw;max-height:85vh} .note{color:#526378}</style>
<header><b>組裝影像盲評 · 30 段</b> <button onclick="save()">匯出評分 JSON</button><span id="count"></span>
<div class="note">先只依影像判斷，勿開啟 PRIVATE_answer_key.json。時間為動作起點後秒數；每列為同一取樣位置的三台相機。點圖可放大。</div></header>
<p>這是刻意抽樣的診斷集，不能用來估計整體準確率。先填完所有案例並匯出，再核對答案。畫面只呈現原本四幀×三相機的 512px 輸入，不含額外裁切或影片，因此評估的是這組影格是否足夠。</p>
<p>成功：零件正確插入目標位置。看不到深度、只有夾爪靠近或遮擋時，不要自行補出成敗證據。可選「無法判斷」。本頁不會自動儲存，離開前請匯出。</p><main id="main"></main>
<dialog id="zoom"><button onclick="this.parentElement.close()">關閉</button><br><img id="large"></dialog>
<script>const cases=DATA;
const options=(name,values)=>`<select data-field="${name}"><option value="">尚未填寫</option>${values.map(v=>`<option>${v}</option>`).join('')}</select>`;
main.innerHTML=cases.map(c=>`<article id="${c.id}"><h2>${c.id} · ${c.part}</h2><div class="grid">${c.frames.map(f=>`<figure><img loading="lazy" src="${f.path}" alt="${c.id} ${f.camera} frame ${f.id}" onclick="large.src=this.src;zoom.showModal()"><figcaption>影格 ${f.id} · ${f.camera} · ${f.time.toFixed(2)} 秒</figcaption></figure>`).join('')}</div>
<label>影格足以判斷嗎？${options('visibility',['足夠','部分可見','不足'])}</label>
<label>你的判斷${options('judgment',['成功','失敗','無法判斷'])}</label>
<label>主要限制${options('limitation',['無明顯限制','夾爪遮擋','零件太小或模糊','孔位或接觸點不清','深度或鎖定看不出','缺少關鍵時間點','完成條件不清楚','其他'])}</label>
<label>最佳視角${options('camera',['hama1','hama2','hand','多視角合看','皆不可見'])}</label>
<label>具體證據：請寫影格編號、看見的現象；不要只寫「感覺」<textarea data-field="evidence"></textarea></label>
<label>建議補看什麼（可留空）<textarea data-field="next_view"></textarea></label></article>`).join('');
function collect(){return cases.map(c=>{let r={case_id:c.id};document.getElementById(c.id).querySelectorAll('[data-field]').forEach(e=>r[e.dataset.field]=e.value);return r})}
document.addEventListener('change',()=>{count.textContent=`已判斷 ${collect().filter(r=>r.judgment).length}/30`});
function save(){let blob=new Blob([JSON.stringify({pack:'blind_review_dev30_v1',exported_at:new Date().toISOString(),reviews:collect()},null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='blind_review_dev30_answers.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
</script></html>'''.replace('DATA',data)
    (OUT / 'index.html').write_text(page,encoding='utf-8')
    manifest={'seed':20261031,'n':30,'parts':['dsub','mgear','sqpeg2'],'per_part_per_label':5,
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selection':'stratified random; prefer distinct recordings; no selection on model correctness',
              'purpose':'diagnostic only; not an accuracy benchmark','frames':360}
    (OUT / 'PRIVATE_sampling_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT / 'README.md').write_text('# 30 段開發集盲評\n\n開啟 index.html，填完後按「匯出評分 JSON」，將下載檔保留。頁面不自動儲存，請勿填寫途中重新整理。\n\n先不要打開 PRIVATE 開頭的檔案。真值與模型結果只在 PRIVATE_answer_key.json，HTML 不含真值、預測或原始 segment_id。先匯出評分，再核對。\n\n本包僅使用開發資料；刻意平衡抽樣，不代表資料集整體分布。每例 12 張影像均來自既有取樣並縮至長邊 512，不提供新增影片。先評估影格可判讀性；後續才依記錄找完整動作影片，避免不同資訊條件混在同一次評分。\n\n核對後將錯誤分為：可見但模型看錯、取樣漏掉證據、影像本身不可見、完成條件有歧義、疑似時間或標記問題。模型自己描述的原因不能代替人工判讀。\n',encoding='utf-8')
    assert len(public)==30 and len({r['segment_id'] for r in private})==30
    assert all(r['segment_id'] not in page for r in private)
    assert len(list((OUT/'images').rglob('*.jpg')))==360
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
