#!/usr/bin/env python3
"""
정성 검증: 우리 추출 vs OpenShape released 컬러 포인트클라우드 나란히 렌더
========================================================================

객체별로 [우리 추출 | released] 2뷰(정면/사면) 산점도를 한 장의 PNG 로 저장.
용역 검수 보고서용 정성 비교 자료.

  python tools/prep/render_pc_sidebyside.py --n 4 --out_dir /nas/integrated/trainvaltest/logs/pc_vis
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pc10k.extract import extract_colored_pc          # noqa: E402
from pc10k.validate.compare_openshape import normalize                    # noqa: E402

PREFIX = "uni3d_data/objaverse_lvis/"


def scatter(ax, xyz, rgb, elev=20, azim=45, title=""):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.clip(rgb, 0, 1), s=1.2, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zip", default="/nas/integrated/trainvaltest/test_datasets.zip")
    ap.add_argument("--glb_index", default="/nas/integrated/trainvaltest/glb_index.json")
    ap.add_argument("--out_dir", default="/nas/integrated/trainvaltest/logs/pc_vis")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = zipfile.ZipFile(args.zip)
    lines = z.read(PREFIX + "lvis_testset.txt").decode().splitlines()
    glb = json.loads(open(args.glb_index).read())
    # 유색 released 우선으로 후보 선정
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(lines))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i in order:
        p = lines[i].split(",")
        uid, cat, rel = p[2], p[1], p[3].lstrip("/")
        if uid not in glb:
            continue
        d = np.load(io.BytesIO(z.read(PREFIX + rel)), allow_pickle=True).item()
        ref_xyz = normalize(d["xyz"]); ref_rgb = np.asarray(d["rgb"], np.float64)
        if ref_rgb.std() < 0.02:                    # 유색 객체만 (정성 비교 의미)
            continue
        try:
            pc = extract_colored_pc(glb[uid], 10000).astype(np.float64)
        except Exception:
            continue
        xyz, rgb = normalize(pc[:, :3]), pc[:, 3:]
        fig = plt.figure(figsize=(10, 5.5))
        for k, (X, C, name) in enumerate([(xyz, rgb, "ours"), (ref_xyz, ref_rgb, "OpenShape released")]):
            for v, (el, az) in enumerate([(20, 45), (20, 135)]):
                ax = fig.add_subplot(2, 2, v * 2 + k + 1, projection="3d")
                scatter(ax, X, C, el, az, f"{name} ({cat})" if v == 0 else "")
        fig.tight_layout()
        f = out / f"cmp_{cat}_{uid[:8]}.png"
        fig.savefig(f, dpi=110); plt.close(fig)
        print("saved", f, flush=True)
        done += 1
        if done >= args.n:
            break


if __name__ == "__main__":
    main()
