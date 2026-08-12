#!/usr/bin/env python3
"""한/영 캡션 쌍 정합성 검수 (용역 SDA-4)
=========================================

영어 캡션과 그 한국어 번역이 **같은 객체를 같은 속성으로 서술하는지** 자동 검수한다.
세 층위로 나눠 보고, 층위마다 다른 실패 유형을 잡는다.

  1) 형식(form)    — 한자·가나 혼입, 한글 부재, 길이 이상, 종결어미 규칙 위반
  2) 내용(content) — 숫자·인용문구·브랜드 보존, 색상어 대응 (도메인 특화 하드 신호)
  3) 의미(meaning) — LaBSE(Apache-2.0) 다국어 임베딩의 EN↔KO 코사인 유사도

LaBSE 는 번역쌍 판별용으로 학습된 모델이라 임계값이 안정적이다.
경험적으로 정상 번역쌍 ≥0.75, 0.6~0.75 경계, <0.6 은 오역/누락 의심.

출력: 요약 통계 + 결함 목록 jsonl (--out) — 사람 검수 대상만 추려낸다.

실행:
  python caption/qc_bilingual.py --in "caps_ko*.jsonl"
  python caption/qc_bilingual.py --in caps_ko.jsonl --out defects.jsonl --no_semantic
"""

import argparse
import glob
import json
import re
from collections import Counter

_HANJA = re.compile(r"[一-鿿぀-ヿ]")
_HANGUL = re.compile(r"[가-힣]")
_NUM = re.compile(r"\d+")
_QUOTED = re.compile(r'"([^"]{2,30})"')

# 영어 색상어 → 허용 한국어 표기(하나라도 있으면 대응된 것으로 간주)
COLOR = {
    "red": ["빨간", "빨강", "붉은", "적색", "레드"],
    "blue": ["파란", "파랑", "푸른", "청색", "블루", "남색"],
    "green": ["초록", "녹색", "연두", "그린"],
    "yellow": ["노란", "노랑", "황색", "옐로"],
    "black": ["검은", "검정", "까만", "블랙"],
    "white": ["하얀", "흰", "백색", "화이트"],
    "brown": ["갈색", "밤색", "브라운"],
    "gray": ["회색", "그레이"], "grey": ["회색", "그레이"],
    "orange": ["주황", "오렌지"],
    "purple": ["보라", "퍼플"],
    "pink": ["분홍", "핑크"],
    "gold": ["금색", "골드"], "silver": ["은색", "실버"],
}


def form_defects(ko, en=None):
    """형식 결함 (모델 없이 판정)."""
    d = []
    if "번역" in ko or "：" in ko:
        d.append("meta_leak")
    if _HANJA.search(ko):
        d.append("cjk")            # 한자/가나 혼입 — Qwen2.5 에서 다발
    if not _HANGUL.search(ko):
        d.append("no_hangul")      # 번역 실패 (원문 그대로 등)
    if len(ko) < 4 or len(ko) > 200:
        d.append("length")
    if ko.rstrip().endswith(("입니다.", "있습니다.", "있다.", "이다.", "합니다.")):
        d.append("verb_ending")    # 명사구 종결 규칙 위반
    return d


def content_defects(en, ko):
    """내용 보존 결함 — 숫자·인용문구·색상은 검색 품질에 직결되므로 따로 본다."""
    d = []
    en_l, ko_l = en.lower(), ko.lower()

    miss_num = [n for n in set(_NUM.findall(en)) if n not in ko]
    if miss_num:
        d.append("num_lost")       # "3-tier", "2 reels" 등 수량 정보 소실

    for q in _QUOTED.findall(en):  # 객체에 실제로 적힌 문구는 원문 유지가 규칙
        if q not in ko:
            d.append("quote_lost")
            break

    miss_col = [c for c, ks in COLOR.items()
                if re.search(rf"\b{c}\b", en_l) and not any(k in ko_l for k in ks)]
    if miss_col:
        d.append("color_lost:" + ",".join(miss_col[:3]))
    return d


def semantic_scores(ens, kos, model="sentence-transformers/LaBSE", batch=64):
    """LaBSE EN↔KO 코사인. 반환: numpy (N,)"""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"[load] {model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModel.from_pretrained(model).eval()
    if torch.cuda.is_available():
        net = net.cuda()

    @torch.no_grad()
    def emb(texts):
        out = []
        for i in range(0, len(texts), batch):
            e = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(net.device)
            # LaBSE 는 pooler_output(CLS+tanh)이 문장 표현
            v = net(**e).pooler_output
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out).numpy()

    return (emb(ens) * emb(kos)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="jsonl glob ({caption, caption_ko})")
    ap.add_argument("--out", default=None, help="결함 건만 jsonl 로 저장 (사람 검수용)")
    ap.add_argument("--no_semantic", action="store_true", help="LaBSE 생략 (형식·내용만)")
    ap.add_argument("--sem_thr", type=float, default=0.60, help="의미 불일치 임계값")
    ap.add_argument("--n_sample", type=int, default=15, help="정성 확인용 출력 개수")
    args = ap.parse_args()

    R = []
    for f in sorted(glob.glob(args.inp)):
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            if d.get("caption") and d.get("caption_ko"):
                R.append(d)
    if not R:
        raise SystemExit(f"쌍 없음: {args.inp}")
    print(f"검수 대상 {len(R):,} 쌍\n")

    cnt = Counter()
    for r in R:
        r["_def"] = form_defects(r["caption_ko"], r["caption"]) + \
            content_defects(r["caption"], r["caption_ko"])

    if not args.no_semantic:
        import numpy as np
        s = semantic_scores([r["caption"] for r in R], [r["caption_ko"] for r in R])
        for r, v in zip(R, s):
            r["ko_sim"] = round(float(v), 4)
            if v < args.sem_thr:
                r["_def"].append("semantic")
        print(f"[의미] LaBSE cos  mean {s.mean():.3f}  med {np.median(s):.3f}  "
              f"p10 {np.percentile(s,10):.3f}  <{args.sem_thr} {(s<args.sem_thr).sum()} "
              f"({(s<args.sem_thr).mean()*100:.2f}%)")

    for r in R:
        for d in r["_def"]:
            cnt[d.split(":")[0]] += 1
    bad = [r for r in R if r["_def"]]
    print(f"[형식·내용] 결함 유형별: {dict(cnt) or '없음'}")
    print(f"[종합] 결함 {len(bad):,} / {len(R):,}  →  "
          f"**정상률 {100*(len(R)-len(bad))/len(R):.2f}%**\n")

    for r in bad[:args.n_sample]:
        print(f"  ✗ {','.join(r['_def'])}"
              + (f" (cos {r['ko_sim']})" if "ko_sim" in r else ""))
        print(f"    EN {r['caption'][:95]}")
        print(f"    KO {r['caption_ko'][:95]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in bad:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n✓ 결함 {len(bad):,}건 저장: {args.out}")


if __name__ == "__main__":
    main()
