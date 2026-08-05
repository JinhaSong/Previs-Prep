#!/usr/bin/env python3
"""
검증: 우리 10000+RGB 추출 vs OpenShape released 포인트클라우드 정량 비교
=====================================================================

목적: OpenShape 는 추출 스크립트를 공개하지 않았으므로, 같은 GLB 에서 우리
스크립트(extract_tvt_pc10k_rgb)로 추출한 결과가 OpenShape released 데이터
(Uni3D test_datasets.zip 의 objaverse_lvis npy)와 동등한지 정량 검증한다.
→ 동등하면 "OpenShape/Uni3D 규격 재현 스크립트"로 용역 제공 가능.

비교 항목 (객체별):
  1) Chamfer distance (xyz, 양방향 mean NN):    기하 일치도
  2) 색상 NN-RGB MAE: 우리 각 점의 released NN 점과 RGB 차이 (색 일치도)
  3) 좌표축 규약 탐지: 6 축순열 × 8 부호 = 48개 변환 중 최소 Chamfer 변환
     (identity 가 최적이면 좌표계 동일 → flip/스왑 불필요 확정)
  4) 색상 유무 일치(released 유색인데 우리가 회색이면 색 추출 실패)

방법: 두 점군 모두 동일 정규화(centroid 제거 + max-norm) 후 비교.
추출기는 trimesh-방식(_extract_trimesh)과 open3d-방식(_extract_open3d) 각각 평가.

실행(previs env; scipy 필요):
  python tools/prep/compare_pc_openshape.py --n 10
"""

import argparse
import io
import json
import sys
import zipfile
from itertools import permutations, product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pc10k.extract import _extract_trimesh, _extract_open3d  # noqa: E402

ZIP = "/workspace/libs/Uni3D/data/test_datasets.zip"
PREFIX = "uni3d_data/objaverse_lvis/"
GLB_INDEX = "/nas/integrated/trainvaltest/glb_index.json"  # --glb_index 로 재정의


def normalize(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean(0)
    m = np.linalg.norm(x, axis=1).max()
    return x / m if m > 1e-9 else x


def chamfer(a, b, tree_b=None, tree_a=None):
    """양방향 mean NN 거리."""
    from scipy.spatial import cKDTree
    ta = tree_a or cKDTree(a)
    tb = tree_b or cKDTree(b)
    d_ab, _ = tb.query(a, k=1)
    d_ba, _ = ta.query(b, k=1)
    return float(d_ab.mean() + d_ba.mean()) / 2


def best_axis_transform(ours, ref, nsub=2048):
    """48개 축순열×부호 변환 중 Chamfer 최소 변환 탐색 (서브샘플)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    o = ours[rng.choice(len(ours), min(nsub, len(ours)), replace=False)]
    r = ref[rng.choice(len(ref), min(nsub, len(ref)), replace=False)]
    tr = cKDTree(r)
    best = (None, 1e9)
    for perm in permutations(range(3)):
        for signs in product([1, -1], repeat=3):
            t = o[:, perm] * np.array(signs)
            d, _ = tr.query(t, k=1)
            c = float(d.mean())
            if c < best[1]:
                best = ((perm, signs), c)
    return best


def color_nn_mae(ours_xyz, ours_rgb, ref_xyz, ref_rgb):
    """우리 각 점의 released NN 점과 RGB MAE."""
    from scipy.spatial import cKDTree
    idx = cKDTree(ref_xyz).query(ours_xyz, k=1)[1]
    return float(np.abs(ours_rgb - ref_rgb[idx]).mean())


def main():
    global ZIP
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="both", choices=["both", "open3d", "trimesh"])
    ap.add_argument("--out", default=None, help="객체별 지표 jsonl 저장 경로")
    ap.add_argument("--glb_index", default=None, help="uid→mesh경로 json")
    ap.add_argument("--zip", default=None, help="test_datasets.zip 경로 (기본: 내장 경로)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help="병렬 샤딩 (picks[shard::N])")
    args = ap.parse_args()
    if args.zip:
        ZIP = args.zip
    global GLB_INDEX
    if args.glb_index:
        GLB_INDEX = args.glb_index

    z = zipfile.ZipFile(ZIP)
    lines = z.read(PREFIX + "lvis_testset.txt").decode().splitlines()
    glb = json.loads(open(GLB_INDEX).read())
    # released 테스트셋 중 GLB 를 보유한 uid 만 (재현 비교 가능 대상)
    cands = []
    for ln in lines:
        p = ln.split(",")
        if p[2] in glb:
            cands.append((p[2], p[3].lstrip("/")))
    rng = np.random.default_rng(args.seed)
    picks = [cands[i] for i in rng.choice(len(cands), min(args.n, len(cands)), replace=False)]
    if args.num_shards > 1:
        picks = picks[args.shard::args.num_shards]
    print(f"released∩GLB보유: {len(cands):,} 중 {len(picks)}개 비교 "
          f"(shard {args.shard}/{args.num_shards})\n")

    methods = ([("trimesh", _extract_trimesh), ("open3d", _extract_open3d)]
               if args.method == "both" else
               [(args.method, _extract_open3d if args.method == "open3d" else _extract_trimesh)])
    fout = open(args.out, "w") if args.out else None
    rows = []
    for k, (uid, rel) in enumerate(picks):
        d = np.load(io.BytesIO(z.read(PREFIX + rel)), allow_pickle=True).item()
        ref_xyz = normalize(d["xyz"])
        ref_rgb = np.asarray(d["rgb"], np.float64)
        ref_colored = ref_rgb.std() > 0.01
        for name, fn in methods:
            try:
                xyz, rgb, _nrm = fn(glb[uid], 10000)
                xyz = normalize(xyz)
                rgb = np.asarray(rgb, np.float64)
                (tf, c_best) = best_axis_transform(xyz, ref_xyz)
                c_id = chamfer(xyz[:4096], ref_xyz[:4096])
                cmae = color_nn_mae(xyz, rgb, ref_xyz, ref_rgb)
                rec = dict(uid=uid, method=name, chamfer_id=c_id, tf=str(tf),
                           chamfer_best=c_best, rgb_mae=cmae,
                           ref_colored=bool(ref_colored), ours_colored=bool(rgb.std() > 0.01))
            except Exception as e:
                rec = dict(uid=uid, method=name, fail=e.__class__.__name__,
                           ref_colored=bool(ref_colored))
            rows.append(rec)
            if fout:
                fout.write(json.dumps(rec) + "\n")
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(picks)}]", flush=True)
            if fout:
                fout.flush()
    if fout:
        fout.close()

    # ── 통계 요약 ──
    for name, _ in methods:
        rs = [r for r in rows if r["method"] == name]
        ok = [r for r in rs if "fail" not in r]
        if not ok:
            print(f"[{name}] 전부 실패"); continue
        cb = np.array([r["chamfer_best"] for r in ok])
        ci = np.array([r["chamfer_id"] for r in ok])
        colored_pairs = [r for r in ok if r["ref_colored"] and r["ours_colored"]]
        cm = np.array([r["rgb_mae"] for r in colored_pairs]) if colored_pairs else np.array([])
        ref_c = sum(r["ref_colored"] for r in ok)
        ours_c = sum(r["ours_colored"] for r in ok)
        miss_c = sum(1 for r in ok if r["ref_colored"] and not r["ours_colored"])
        from collections import Counter
        tf_top = Counter(r["tf"] for r in ok).most_common(3)
        print(f"\n===== [{name}] n={len(rs)} 성공 {len(ok)} 실패 {len(rs)-len(ok)} =====")
        print(f"Chamfer(best): mean {cb.mean():.4f}  median {np.median(cb):.4f}  "
              f"p90 {np.percentile(cb,90):.4f}  p99 {np.percentile(cb,99):.4f}")
        print(f"Chamfer(id)  : mean {ci.mean():.4f}  median {np.median(ci):.4f}  "
              f"p90 {np.percentile(ci,90):.4f}")
        if len(cm):
            print(f"RGB-MAE(양쪽 유색 {len(cm)}쌍): mean {cm.mean():.3f}  median {np.median(cm):.3f}  "
                  f"p90 {np.percentile(cm,90):.3f}  | <0.1 비율 {(cm<0.1).mean()*100:.1f}%")
        print(f"색 보유: released {ref_c}/{len(ok)}  우리 {ours_c}/{len(ok)}  "
              f"released유색인데 우리무색 {miss_c}")
        print(f"최적변환 top3: {tf_top}")


if __name__ == "__main__":
    main()
