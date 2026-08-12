#!/usr/bin/env python3
"""캡션 샤드 병합 → 인도용 단일 데이터셋 (용역 SDA-4)
=====================================================

여러 GPU·서버가 만든 샤드 jsonl 을 하나로 합치고, 인도 가능한 상태로 정리한다.

  1) uid 중복 제거 — 샤드가 겹쳐 돌면 같은 객체가 여러 파일에 생긴다
  2) 실패 레코드 제거 — `caption` 없는 건(렌더 없음·OOM 등)은 데이터가 아니다
  3) 커버리지 보고 — 목표 대비 확보율, 클래스별 최소/최대 개수
  4) 필드 정리 — 내부용 필드(`_repaired`, `_def`) 제거, 스키마 고정

**병합은 검수가 아니다.** 합친 결과에 대해 qc_bilingual.py 를 반드시 다시 돌릴 것.

실행:
  python caption/merge_captions.py --in "caps_*.jsonl" --out dataset.jsonl
  python caption/merge_captions.py --in "ko_v4_s*.jsonl" --out dataset.jsonl --require_ko
"""

import argparse
import glob
import json
from collections import Counter, defaultdict

FIELDS = ["uid", "category", "caption", "captions", "caption_ko", "n_views"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="샤드 jsonl glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require_ko", action="store_true",
                    help="caption_ko 없는 건도 제외 (한/영 쌍 데이터셋으로 인도할 때)")
    ap.add_argument("--target", type=int, default=0, help="목표 객체 수 (커버리지 보고용)")
    ap.add_argument("--min_per_class", type=int, default=0,
                    help="클래스당 최소 개수. 미달 클래스를 경고로 나열")
    args = ap.parse_args()

    files = sorted(glob.glob(args.inp))
    if not files:
        raise SystemExit(f"입력 없음: {args.inp}")

    best, stat = {}, Counter()
    for f in files:
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                stat["깨진 라인"] += 1
                continue
            stat["총 라인"] += 1
            if not d.get("caption"):
                stat["실패(캡션 없음)"] += 1
                continue
            if args.require_ko and not d.get("caption_ko"):
                stat["한국어 없음"] += 1
                continue
            uid = d.get("uid")
            if uid in best:
                stat["중복"] += 1
                # 같은 uid 가 여럿이면 정보가 더 많은 쪽(한국어 보유)을 남긴다
                if not (best[uid].get("caption_ko") is None and d.get("caption_ko")):
                    continue
            best[uid] = d

    recs = [{k: r[k] for k in FIELDS if k in r} for r in best.values()]
    recs.sort(key=lambda r: (r.get("category", ""), r["uid"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"입력 파일 {len(files)}개")
    for k, v in stat.items():
        print(f"  {k:16s} {v:8,}")
    print(f"\n✓ 저장: {args.out}  —  **고유 객체 {len(recs):,}**")
    if args.target:
        print(f"  커버리지 {100 * len(recs) / args.target:.2f}%  "
              f"(목표 {args.target:,}, 부족 {max(args.target - len(recs), 0):,})")

    by_cat = defaultdict(int)
    for r in recs:
        by_cat[r.get("category", "")] += 1
    if by_cat:
        c = sorted(by_cat.values())
        print(f"  클래스 {len(by_cat):,}개 | 클래스당 min {c[0]} / 중앙 {c[len(c) // 2]} / max {c[-1]}")
        if args.min_per_class:
            low = sorted(k for k, v in by_cat.items() if v < args.min_per_class)
            print(f"  ⚠ {args.min_per_class}개 미만 클래스 {len(low):,}개"
                  + (f": {', '.join(low[:8])}{' …' if len(low) > 8 else ''}" if low else ""))
    if any(r.get("caption_ko") for r in recs):
        n = sum(1 for r in recs if r.get("caption_ko"))
        print(f"  한국어 보유 {n:,} ({100 * n / len(recs):.2f}%)")


if __name__ == "__main__":
    main()
