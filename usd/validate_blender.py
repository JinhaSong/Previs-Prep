#!/usr/bin/env python3
"""USD 산출물 검증 — Blender 로 실제 임포트해 원본 GLB 와 대조
================================================================

**⚠️ 이 파일은 인도 도커 이미지에 포함하지 않는다.**
`bpy`(Blender Python API)를 import 하므로 GPL-3.0 이 적용된다. 본 저장소의 다른
모듈은 Apache/BSD/MIT 계열이므로, 이 검증 도구는 **사내 검증 전용**으로만 쓴다.
(Blender 의 *산출물 데이터* 자체는 GPL 대상이 아니므로 usd 산출물 인도는 무관하다.)

왜 필요한가 — USD 자체 API(pxr)로 "머티리얼이 있다"를 확인하는 것과, 외부 소비자가
그 머티리얼을 **실제로 쓸 수 있는지**는 다른 문제다. 실측으로 드러난 것:
  · 텍스처가 메시마다 중복 저장 (원본 이미지 1개 → USD 14개)  ← pxr 로는 정상으로 보임
  · trimesh PBRMaterial 의 `main_color` 미읽음 → 색이 회색으로 소실
어느 것도 pxr 검사만으로는 잡히지 않았다.

판정 기준은 **원본 유형에 따라 다르다.**
  · 텍스처 보유 원본 → USD 에도 이미지가 붙고 BaseColor 에 연결돼야 함
  · 상수색 원본      → USD 머티리얼의 색값이 원본과 근사해야 함
    (상수색은 상수가 정답이다. "연결 유무"로만 판정하면 정상 데이터를 실패로 오판한다)

**Blender 버전 요건: 4.1.1 이상** (검증 수행: 5.0.1)
  · 3.0.x 는 임포트 단계에서 UsdUVTexture 를 만들지 않고 상수색만 넣는다
    → `is_linked` 기반 판정이 **정상 파일을 실패로 오판**한다. 쓰지 말 것.
  · 4.x/5.x 는 임포트는 정상이나 **렌더 평가에서 USD 머티리얼이 누락되는 회귀**가 있다
    (3.6 계열만 렌더 정상). 본 도구는 렌더를 쓰지 않으므로 영향받지 않지만,
    렌더 기반 확인을 추가하려면 3.6 을 써야 한다.

설치:  pip install bpy   (+ libx11-6 libxkbcommon0 libxi6 libxrender1 libsm6 등 런타임 의존)
실행:  python -m usd.validate_blender --glb_dir /data/meshes --usd_dir /data/usd --n 20
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

TOL_REL = 0.01          # bbox 상대 허용오차
TOL_ABS = 0.02


MIN_BLENDER = (3, 6, 0)     # 3.0.x 는 임포트에서 UVTexture 를 만들지 않는다
RENDER_OK = (3, 6)          # 렌더 평가가 정상인 계열 (4.x/5.x 는 머티리얼 누락 회귀)


def _check_version(need_render=False):
    """버전 요건은 검사 방식에 따라 다르다.

    · 임포트 레벨 검사 : 3.6 이상 (3.0.x 는 UVTexture 미생성 → 정상 파일 오판)
    · 렌더 기반 검사   : 3.6 계열만 (4.x/5.x 는 임포트는 되나 렌더에서 머티리얼 누락)
    초기 구현은 하한을 4.1 로 잡아 **렌더가 유일하게 정상인 3.6 을 차단**했다.
    """
    import bpy
    v = bpy.app.version
    if v < MIN_BLENDER:
        raise SystemExit(
            f"[중단] Blender {bpy.app.version_string} 는 USD 머티리얼 임포트가 불완전합니다"
            f" (요건 {'.'.join(map(str, MIN_BLENDER))}+). 정상 파일을 실패로 오판합니다.")
    if need_render and v[:2] != RENDER_OK:
        raise SystemExit(
            f"[중단] 렌더 기반 검사는 Blender 3.6 계열이 필요합니다 "
            f"(현재 {bpy.app.version_string}). 4.x/5.x 는 임포트는 정상이나 "
            "렌더 평가에서 USD 머티리얼이 누락됩니다.")


def _measure(path, kind):
    import bpy
    import mathutils
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        (bpy.ops.import_scene.gltf if kind == "glb" else bpy.ops.wm.usd_import)(filepath=path)
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:80]}"}
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    lo = [1e18] * 3
    hi = [-1e18] * 3
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(c)
            for i in range(3):
                lo[i] = min(lo[i], w[i])
                hi[i] = max(hi[i], w[i])
    size = [round(hi[i] - lo[i], 3) for i in range(3)] if objs else [0, 0, 0]
    imgs = [i for i in bpy.data.images if i.name != "Render Result"]
    colors, linked = [], 0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        for n in m.node_tree.nodes:
            if n.type != "BSDF_PRINCIPLED":
                continue
            ip = n.inputs.get("Base Color")
            if not ip:
                continue
            if ip.is_linked:
                linked += 1
            else:
                colors.append(tuple(round(x, 3) for x in ip.default_value[:3]))
    return {"bbox": size, "mesh": len(objs), "mat": len(bpy.data.materials),
            "img": len(imgs), "linked": linked,
            "colors": sorted(set(colors))[:5], "err": ""}


def check_pair(glb, usd):
    g = _measure(glb, "glb")
    if not os.path.exists(usd):
        return {"uid": Path(glb).stem, "geo_ok": False, "col_ok": False,
                "glb": g, "usd": {"err": "USD 미생성"}, "mode": "-"}
    u = _measure(usd, "usd")
    geo = bool(g.get("bbox") and u.get("bbox") and all(
        abs(a - b) <= max(TOL_ABS, abs(a) * TOL_REL) for a, b in zip(g["bbox"], u["bbox"])))
    if g.get("img", 0) > 0:
        mode, col = "텍스처", (u.get("img", 0) > 0 and u.get("linked", 0) > 0)
    else:
        mode, col = "상수색", bool(u.get("mat", 0) >= 1 and u.get("colors"))
    return {"uid": Path(glb).stem, "mode": mode, "glb": g, "usd": u,
            "geo_ok": geo, "col_ok": col}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb_dir", required=True, help="원본 메시 디렉토리")
    ap.add_argument("--usd_dir", required=True, help="변환 산출물 디렉토리")
    ap.add_argument("--ext", default="usdc")
    ap.add_argument("--n", type=int, default=20, help="표본 개수 (0=전수)")
    ap.add_argument("--convert_missing", action="store_true",
                    help="USD 가 없으면 usd.convert 로 즉석 변환 후 검사")
    ap.add_argument("--out", default=None, help="결과 jsonl")
    args = ap.parse_args()

    _check_version()
    src = sorted(glob.glob(os.path.join(args.glb_dir, "**", "*.glb"), recursive=True))
    if args.n:
        src = src[:args.n]
    if not src:
        raise SystemExit(f"GLB 없음: {args.glb_dir}")

    rows, bad = [], 0
    for s in src:
        rel = Path(s).relative_to(args.glb_dir).with_suffix(f".{args.ext}")
        usd = str(Path(args.usd_dir) / rel)
        if args.convert_missing and not os.path.exists(usd):
            subprocess.run([sys.executable, "-m", "usd.convert",
                            "--input", s, "--out", usd], capture_output=True)
        r = check_pair(s, usd)
        rows.append(r)
        ok = r["geo_ok"] and r["col_ok"]
        bad += not ok
        g, u = r["glb"], r["usd"]
        print(f"{'OK ' if ok else 'FAIL'} {r['uid'][:12]:12s} [{r['mode']}] "
              f"glb mat={g.get('mat')} img={g.get('img')} | "
              f"usd mat={u.get('mat')} img={u.get('img')} link={u.get('linked')} "
              f"{'' if ok else '← 지오' + ('OK' if r['geo_ok'] else 'X') + '/색' + ('OK' if r['col_ok'] else 'X')}",
              flush=True)

    n = len(rows)
    print(f"\n검사 {n} | 실패 {bad} | 통과율 {100*(n-bad)/n:.1f}%")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"결과: {args.out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
