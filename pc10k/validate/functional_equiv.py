#!/usr/bin/env python3
"""
기능적 동등성 검증: Uni3D-g 에 [OpenShape released vs 우리 추출] 입력 → 결과 동일한가
=================================================================================

궁극 검증: 기하/색 지표가 아니라 **다운스트림 모델이 같은 출력을 내는지**.
  1) 임베딩 코사인: cos( E(우리 pc), E(released pc) )  객체별 → mean/median/p10
  2) LVIS zero-shot 일치: 캐시된 텍스트 특징으로 top1 예측 일치율 + 양쪽 정확도
  3) yaw 180° 프레임 보정((x,y,z)→(-x,y,-z)) 적용 시 개선되는지 → 프레임 규약 확정

실행(125 ddp_test, previs env + pointnet2_ops):
  python tools/eval/functional_pc_equiv.py --n 200
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

PREFIX = "uni3d_data/objaverse_lvis/"


def normalize_xyz(x):
    x = np.asarray(x, np.float32)
    x = x - x.mean(0)
    m = np.linalg.norm(x, axis=1).max()
    return x / m if m > 1e-9 else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip", default="/nas/integrated/trainvaltest/test_datasets.zip")
    ap.add_argument("--glb_index", default="/nas/integrated/trainvaltest/glb_index.json")
    ap.add_argument("--ckpt", default="/nas/integrated/trainvaltest/uni3d_g_ckpt.pt")
    ap.add_argument("--text_feat", default="/nas/integrated/trainvaltest/eval/lvis_text_uni3d.npy")
    ap.add_argument("--uni3d_root", default="/workspace/libs/Uni3D")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="/nas/integrated/trainvaltest/logs/func_equiv.json")
    args = ap.parse_args()

    # ── 대상 선정 ──
    z = zipfile.ZipFile(args.zip)
    lines = z.read(PREFIX + "lvis_testset.txt").decode().splitlines()
    glb = json.loads(open(args.glb_index).read())
    labels = json.load(open(Path(args.uni3d_root) / "data" / "labels.json"))["objaverse_lvis_openshape"]
    lab2idx = {l: i for i, l in enumerate(labels)}
    cands = [(p[2], p[1], p[3].lstrip("/")) for p in (l.split(",") for l in lines) if p[2] in glb]
    rng = np.random.default_rng(args.seed)
    picks = [cands[i] for i in rng.choice(len(cands), min(args.n, len(cands)), replace=False)]
    print(f"대상 {len(picks)}개 / 모집단 {len(cands):,}", flush=True)

    # ── 인코더 ──
    from models.lightning.teachers import build_teacher_encoder
    enc = build_teacher_encoder("uni3d", device="cuda", uni3d_scale="giant",
                                uni3d_checkpoint=args.ckpt)

    def encode_batches(arrs):
        out = []
        for s in range(0, len(arrs), args.batch):
            out.append(enc.encode(np.stack(arrs[s:s + args.batch]).astype(np.float32)))
        return np.concatenate(out)

    rel_pcs, our_pcs, yaw_pcs, gts = [], [], [], []
    fails = 0
    for uid, cat, rel in picks:
        try:
            d = np.load(io.BytesIO(z.read(PREFIX + rel)), allow_pickle=True).item()
            ref = np.concatenate([normalize_xyz(d["xyz"]),
                                  np.asarray(d["rgb"], np.float32)], 1)      # (10000,6)
            pc = extract_colored_pc(glb[uid], 10000).astype(np.float32)      # 정규화 포함
            yaw = pc.copy()
            yaw[:, 0] *= -1; yaw[:, 2] *= -1                                 # (x,z)→(-x,-z)
            rel_pcs.append(ref); our_pcs.append(pc); yaw_pcs.append(yaw)
            gts.append(lab2idx[cat])
        except Exception:
            fails += 1
    print(f"추출 성공 {len(gts)} / 실패 {fails}", flush=True)

    print("인코딩: released ...", flush=True)
    E_rel = encode_batches(rel_pcs)
    print("인코딩: ours ...", flush=True)
    E_our = encode_batches(our_pcs)
    print("인코딩: ours+yaw180 ...", flush=True)
    E_yaw = encode_batches(yaw_pcs)

    def l2n(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    E_rel, E_our, E_yaw = l2n(E_rel), l2n(E_our), l2n(E_yaw)
    gt = np.array(gts)

    txt = np.load(args.text_feat)
    txt = l2n(txt)

    def report(name, E):
        cos = (E * E_rel).sum(1)
        pred = (E @ txt.T).argmax(1)
        pred_rel = (E_rel @ txt.T).argmax(1)
        agree = float((pred == pred_rel).mean())
        acc = float((pred == gt).mean())
        acc_rel = float((pred_rel == gt).mean())
        print(f"[{name}] cos: mean {cos.mean():.4f} med {np.median(cos):.4f} "
              f"p10 {np.percentile(cos,10):.4f} | top1일치 {agree*100:.1f}% "
              f"| acc(우리) {acc*100:.1f}% vs acc(released) {acc_rel*100:.1f}%")
        return dict(cos_mean=float(cos.mean()), cos_med=float(np.median(cos)),
                    cos_p10=float(np.percentile(cos, 10)), top1_agree=agree,
                    acc=acc, acc_released=acc_rel)

    res = {"n": len(gt),
           "ours": report("우리 그대로", E_our),
           "ours_yaw180": report("우리+yaw180", E_yaw)}
    json.dump(res, open(args.out, "w"), indent=1)
    print("✓ 저장:", args.out)


if __name__ == "__main__":
    main()
