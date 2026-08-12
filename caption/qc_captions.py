#!/usr/bin/env python3
"""생성 캡션 품질 점검: 통계 + 규칙 위반 + 랜덤 샘플 출력 (+Cap3D 대조)."""
import argparse
import csv
import glob
import json
import random

import numpy as np

BAD = ["is shown", "various angles", "the object is", "3d model", "background",
       "these images", "the image", "viewpoint"]
VAGUE = ["rectangular object", "shaped object", "an object with", "a red box",
         "geometric object", "abstract object"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="jsonl glob")
    ap.add_argument("--cap3d", default="/nas/integrated/captions/Cap3D_automated_Objaverse_full.csv")
    ap.add_argument("--n_sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--compare", action="store_true", help="Cap3D 캡션 나란히 출력")
    args = ap.parse_args()

    R = [json.loads(l) for f in sorted(glob.glob(args.pattern)) for l in open(f, encoding="utf-8")]
    ok = [r for r in R if r.get("caption")]
    fail = [r for r in R if not r.get("caption")]
    L = np.array([len(r["caption"].split()) for r in ok])
    bad = [r for r in ok if any(x in r["caption"].lower() for x in BAD)]
    vague = [r for r in ok if any(x in r["caption"].lower() for x in VAGUE)]
    dup = len(ok) - len({r["caption"] for r in ok})

    print(f"총 {len(R):,} | 성공 {len(ok):,} | 실패 {len(fail)} "
          f"({len(fail)/max(len(R),1)*100:.2f}%)")
    print(f"단어수: mean {L.mean():.1f} med {np.median(L):.0f} "
          f"p10 {np.percentile(L,10):.0f} p90 {np.percentile(L,90):.0f}   [Cap3D 15.1/15/6/24]")
    print(f"형식위반 {len(bad)} ({len(bad)/len(ok)*100:.2f}%) | "
          f"모호명명 {len(vague)} ({len(vague)/len(ok)*100:.2f}%) | 중복문장 {dup}")
    print(f"클래스 수 {len({r['category'] for r in ok})} | "
          f"클래스당 평균 {len(ok)/max(len({r['category'] for r in ok}),1):.1f}")
    if bad[:3]:
        print("위반 예:", [r["caption"][:70] for r in bad[:3]])
    if vague[:3]:
        print("모호 예:", [r["caption"][:70] for r in vague[:3]])

    caps = {}
    if args.compare:
        with open(args.cap3d, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    caps[row[0]] = row[1]

    print(f"\n=== 랜덤 {args.n_sample}개 ===")
    random.seed(args.seed)
    for r in random.sample(ok, min(args.n_sample, len(ok))):
        print(f"[{r['category']}]")
        print(f"  QWEN : {r['caption'][:150]}")
        if args.compare:
            print(f"  CAP3D: {caps.get(r['uid'], '(없음)')[:150]}")


if __name__ == "__main__":
    main()
