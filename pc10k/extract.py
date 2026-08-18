#!/usr/bin/env python3
"""
pc10k — 10,000점 + RGB 포인트클라우드 추출 (OpenShape/Uni3D 규격)
================================================================

3D 메시(GLB/OBJ/PLY 등)에서 색상 포함 포인트클라우드를 추출한다.
OpenShape released 데이터와의 동등성이 검증된 파이프라인 (SPEC.md 참조):
  - 면적가중 삼각형 샘플 + barycentric UV 보간 + 텍스처 룩업 (정점색/재질 폴백)
  - Open3D 우선(read_triangle_mesh → 실패 시 read_triangle_model 서브메시 병합),
    trimesh 최후 폴백
  - unit-sphere 정규화(centroid + max-norm), 좌표축 flip 없음, rgb [0,1]
  - 출력: (N, 6) float16 npy  [xyz | rgb]

사용:
  # 단일 메시
  python -m pc10k.extract --input model.glb --out out/model.npy
  # 디렉토리 배치 (재귀 탐색, resume: 기존 npy/.att 마커 skip)
  python -m pc10k.extract --input mesh_dir/ --out out_dir/ --workers 8 --shard 0 --num_shards 1
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")
NPOINTS = 10000
MESH_EXTS = (".glb", ".gltf", ".obj", ".ply", ".off", ".stl", ".fbx")


def _img_to_np(t):
    """Open3D legacy Image → float32 [0,1] HxWx3. 비표준 포맷(16bit 등)도 안전 변환.

    ※ np.asarray(Image) 는 미지원 포맷에서 buffer protocol 내 C++ 예외로 프로세스가
    즉사(terminate)하므로 반드시 tensor API(from_legacy) 경유로 변환한다."""
    import open3d as o3d
    try:
        if t is None or t.is_empty():
            return None
        a = o3d.t.geometry.Image.from_legacy(t).as_tensor().numpy()
        if a.ndim == 2:
            a = a[..., None].repeat(3, -1)
        if a.ndim != 3 or a.shape[-1] < 3:
            return None
        a = a[..., :3].astype(np.float32)
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        mx = a.max() if a.size else 0
        if mx > 256:                                # uint16 등
            a = a / 65535.0
        elif mx > 1.5:                              # uint8
            a = a / 255.0
        return np.clip(a, 0, 1)
    except Exception:
        return None


# UV v축 뒤집기. Open3D triangle_uvs 는 이미 이미지 좌표계라 뒤집지 않는 것이 정답
# (OpenShape released 와 A/B 검증: flip 시 RGB-MAE 0.28, no-flip 0.159).
V_FLIP = os.environ.get("PC_V_FLIP", "0") == "1"


def _tex_lookup(tex_arr, uv, v_flip=None):
    """텍스처 배열에서 UV 좌표 색 조회 (nearest, wrap). uv: (N,2) [0,1] 범위 밖 허용."""
    if v_flip is None:
        v_flip = V_FLIP
    h, w = tex_arr.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    if v_flip:                                     # OpenGL 관례: v 원점 하단 → 배열 상단 원점
        v = 1.0 - v
    x = np.clip((u * (w - 1)).round().astype(int), 0, w - 1)
    y = np.clip((v * (h - 1)).round().astype(int), 0, h - 1)
    c = tex_arr[y, x, :3].astype(np.float32)
    if c.max() > 1.5:                              # uint8 → [0,1]
        c = c / 255.0
    return c


def _sample_mesh_colored(verts, tris, n, tri_uvs=None, textures=None, tri_mat_ids=None,
                         vert_colors=None, flat_color=None, v_flip=None):
    """면적가중 삼각형 샘플 + barycentric 보간으로 (xyz, rgb) 샘플.

    색 우선순위: UV+텍스처 > 정점색 보간 > flat_color > 회색 0.4.
    tri_uvs: (3*|T|, 2) — Open3D triangle_uvs 레이아웃. textures: [HxWxC array].
    tri_mat_ids: (|T|,) 삼각형별 텍스처 인덱스 (없으면 0)."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    nrm_all = cr / (np.linalg.norm(cr, axis=1, keepdims=True) + 1e-12)   # 면법선
    tot = areas.sum()
    if not np.isfinite(tot) or tot <= 0:
        probs = np.full(len(tris), 1.0 / len(tris))
    else:
        probs = areas / tot
    ti = np.random.choice(len(tris), n, p=probs)
    # barycentric (균일)
    r1 = np.sqrt(np.random.rand(n, 1)); r2 = np.random.rand(n, 1)
    b0, b1, b2 = 1 - r1, r1 * (1 - r2), r1 * r2
    xyz = (b0 * verts[tris[ti, 0]] + b1 * verts[tris[ti, 1]] + b2 * verts[tris[ti, 2]]
           ).astype(np.float32)
    nrm = nrm_all[ti].astype(np.float32)

    rgb = None
    if tri_uvs is not None and textures:
        uv = (b0 * tri_uvs[3 * ti] + b1 * tri_uvs[3 * ti + 1] + b2 * tri_uvs[3 * ti + 2])
        mids = tri_mat_ids[ti] if tri_mat_ids is not None else np.zeros(n, int)
        rgb = np.full((n, 3), 0.4, np.float32)
        done = np.zeros(n, bool)
        for mi in np.unique(mids):
            if 0 <= mi < len(textures) and textures[mi] is not None:
                sel = mids == mi
                rgb[sel] = _tex_lookup(textures[mi], uv[sel], v_flip)
                done |= sel
        if not done.any():
            rgb = None
    if rgb is None and vert_colors is not None and len(vert_colors) == len(verts):
        rgb = (b0 * vert_colors[tris[ti, 0]] + b1 * vert_colors[tris[ti, 1]]
               + b2 * vert_colors[tris[ti, 2]]).astype(np.float32)
    if rgb is None:
        rgb = np.tile(np.asarray(flat_color if flat_color is not None else [0.4] * 3,
                                 np.float32), (n, 1))
    return xyz, rgb, nrm


def _mesh_to_arrays(m):
    """Open3D legacy TriangleMesh → 샘플러 입력 배열들."""
    verts = np.asarray(m.vertices, np.float64)
    tris = np.asarray(m.triangles, np.int64)
    tri_uvs = np.asarray(m.triangle_uvs, np.float64) if m.has_triangle_uvs() else None
    textures = None
    if tri_uvs is not None and len(m.textures):
        textures = [_img_to_np(t) for t in m.textures]
    tri_mat_ids = (np.asarray(m.triangle_material_ids, np.int64)
                   if len(m.triangle_material_ids) == len(tris) else None)
    vc = np.asarray(m.vertex_colors, np.float64) if m.has_vertex_colors() else None
    return verts, tris, tri_uvs, textures, tri_mat_ids, vc


def _extract_open3d(p, num_points):
    """OpenShape 규격 재현: Open3D 로드 + UV 텍스처 직접 샘플링.

    sample_points_uniformly 는 정점색만 샘플해 텍스처 색이 소실됨 → 면적가중
    삼각형 샘플 + barycentric UV 보간 + 텍스처 룩업을 직접 수행.
    단일메시(read_triangle_mesh) 실패 시 read_triangle_model 로 서브메시별
    (albedo 텍스처 포함) 샘플 후 면적 비례로 합침."""
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    m = o3d.io.read_triangle_mesh(p, enable_post_processing=True)
    if len(m.triangles) > 0:
        verts, tris, tri_uvs, textures, tmi, vc = _mesh_to_arrays(m)
        return _sample_mesh_colored(verts, tris, num_points, tri_uvs, textures, tmi, vc)  # (xyz,rgb,nrm)

    # 다중 지오메트리 → 모델 로드, 서브메시별 샘플 (면적 비례 배분)
    model = o3d.io.read_triangle_model(p)
    subs = []
    for mi in model.meshes:
        sub = mi.mesh
        if len(sub.triangles) == 0:
            continue
        verts = np.asarray(sub.vertices, np.float64)
        tris = np.asarray(sub.triangles, np.int64)
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        area = float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())
        tex = flat = None
        try:
            mat = model.materials[mi.material_idx]
            tex = _img_to_np(getattr(mat, "albedo_img", None))
            base = np.asarray(mat.base_color, np.float64)[:3]
            if base.shape == (3,):
                flat = base
        except Exception:
            pass
        tri_uvs = (np.asarray(sub.triangle_uvs, np.float64)
                   if sub.has_triangle_uvs() else None)
        vc = np.asarray(sub.vertex_colors, np.float64) if sub.has_vertex_colors() else None
        subs.append((area, verts, tris, tri_uvs, tex, vc, flat))
    if not subs:
        raise ValueError("open3d: empty mesh")
    tot = sum(s[0] for s in subs) or 1.0
    xyzs, rgbs, nrms = [], [], []
    for k, (area, verts, tris, tri_uvs, tex, vc, flat) in enumerate(subs):
        nk = max(1, int(round(num_points * area / tot))) if k < len(subs) - 1 \
            else max(1, num_points - sum(len(x) for x in xyzs))
        x, c, nn = _sample_mesh_colored(verts, tris, nk, tri_uvs,
                                        [tex] if tex is not None else None,
                                        None, vc, flat)
        xyzs.append(x); rgbs.append(c); nrms.append(nn)
    xyz = np.concatenate(xyzs)[:num_points]
    rgb = np.concatenate(rgbs)[:num_points]
    nrm = np.concatenate(nrms)[:num_points]
    if len(xyz) < num_points:                      # 부족 시 복원 샘플
        idx = np.random.choice(len(xyz), num_points - len(xyz))
        xyz = np.concatenate([xyz, xyz[idx]]); rgb = np.concatenate([rgb, rgb[idx]])
        nrm = np.concatenate([nrm, nrm[idx]])
    return xyz.astype(np.float32), rgb.astype(np.float32), nrm.astype(np.float32)


def _extract_trimesh(p, num_points):
    """폴백: trimesh 로 Scene 의 mesh 들을 색상 baking 후 concat → 표면 샘플."""
    import trimesh
    loaded = trimesh.load(p, force="scene")
    if isinstance(loaded, trimesh.Scene):
        parts = []
        for g in loaded.geometry.values():
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0:
                try:
                    if hasattr(g.visual, "to_color"):
                        g = g.copy(); g.visual = g.visual.to_color()
                except Exception:
                    pass
                parts.append(g)
        if parts:
            m = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        else:
            pcs = [g for g in loaded.geometry.values()
                   if hasattr(g, "vertices") and len(getattr(g, "vertices", [])) > 0]
            if not pcs:
                raise ValueError("trimesh: no geometry")
            m = trimesh.PointCloud(np.concatenate([np.asarray(g.vertices) for g in pcs]))
    else:
        m = loaded
    nrm = None
    if hasattr(m, "faces") and len(getattr(m, "faces", [])) > 0:
        try:
            pts, fidx, col = trimesh.sample.sample_surface(m, num_points, sample_color=True)
            rgb = np.asarray(col)[:, :3].astype(np.float32) / 255.0
            nrm = np.asarray(m.face_normals[fidx], np.float32)
        except Exception:
            pts = m.sample(num_points)
            rgb = np.full((num_points, 3), 0.4, np.float32)
        xyz = np.asarray(pts, np.float32)
    else:
        verts = np.asarray(m.vertices, np.float32)
        if len(verts) == 0:
            raise ValueError("trimesh: no vertices")
        idx = np.random.choice(len(verts), num_points, replace=len(verts) < num_points)
        xyz = verts[idx]
        rgb = np.full((num_points, 3), 0.4, np.float32)
    if nrm is None:
        nrm = np.zeros_like(xyz)
    return xyz, rgb, nrm


def extract_colored_pc(mesh_path, num_points=NPOINTS, with_normals=False,
                       return_meta=False):
    """mesh → float16 배열. flip 없음, xyz unit-sphere, rgb [0,1].

    with_normals=False: (N, 6) [xyz | rgb]        (모델 입력·학습용, 기본)
    with_normals=True : (N, 9) [xyz | nx,ny,nz | rgb]  (납품 규격 — 면법선)
    return_meta=True  : (배열, meta dict) — 정규화 파라미터 동봉

    Open3D 우선 (OpenShape released 와 동등성 검증), 실패 시 trimesh 폴백."""
    p = str(mesh_path)
    try:
        xyz, rgb, nrm = _extract_open3d(p, num_points)
    except Exception:
        xyz, rgb, nrm = _extract_trimesh(p, num_points)
    if len(xyz) == 0:
        raise ValueError("no geometry")
    if rgb.shape != xyz.shape or len(rgb) == 0:
        rgb = np.full_like(xyz, 0.4)
    if len(xyz) != num_points:
        idx = np.random.choice(len(xyz), num_points, replace=len(xyz) < num_points)
        xyz, rgb = xyz[idx], rgb[idx]
    # 정규화 파라미터는 **샘플된 점 기준**으로 계산된다. 샘플링이 난수라
    # 같은 메시라도 실행마다 값이 달라지므로(실측 AABB 편차 최대 0.018),
    # 사후에 원본 좌표계로 되돌리려면 이 값을 그때 기록해 두는 수밖에 없다.
    # (용역 SDA-3 "카메라 파라미터와 동일 좌표계 정합", DIV-3 "3D-2D 투영 검증")
    centroid = xyz.mean(0)
    xyz = xyz - centroid
    mmax = float(np.linalg.norm(xyz, axis=1).max())
    if mmax > 1e-6:
        xyz = xyz / mmax                            # 평행이동+등방스케일 → 법선 불변
    else:
        mmax = 1.0
    if with_normals:
        pc = np.concatenate([xyz, nrm[:len(xyz)], rgb], axis=1).astype(np.float16)
    else:
        pc = np.concatenate([xyz, rgb], axis=1).astype(np.float16)
    if not return_meta:
        return pc
    meta = {
        "num_points": int(len(xyz)),
        "normalization": "unit_sphere",     # centroid 제거 후 max-norm 으로 나눔
        # 원본 좌표 복원:  p_orig = p_norm * scale + offset
        "offset": [float(v) for v in centroid],
        "scale": mmax,
        "aabb_normalized": {"min": [float(v) for v in xyz.min(0)],
                            "max": [float(v) for v in xyz.max(0)]},
        "aabb_original": {"min": [float(v) for v in (xyz.min(0) * mmax + centroid)],
                          "max": [float(v) for v in (xyz.max(0) * mmax + centroid)]},
        "axis_convention": "mesh_original",  # YZ flip 등 축 변환 없음
    }
    return pc, meta


def write_ply(path, pc9):
    """(N,9) [xyz|nrm|rgb] → 납품 규격 PLY (binary_little_endian).

    property: x,y,z float32 / nx,ny,nz float32 / red,green,blue uchar."""
    pc9 = np.asarray(pc9, np.float32)
    n = len(pc9)
    arr = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, k in enumerate(["x", "y", "z", "nx", "ny", "nz"]):
        arr[k] = pc9[:, i]
    rgb = np.clip(pc9[:, 6:9] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment pc10k (OpenShape/Uni3D-compatible, unit-sphere, no axis flip)\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def _save(out_path, pc9, fmt, meta=None):
    """fmt 에 따라 저장. ply=납품규격 / npy=(N,9) f16 / both=둘 다.

    meta 가 주어지면 같은 이름의 `.norm.json` 으로 정규화 파라미터를 남긴다."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("ply", "both"):
        write_ply(out.with_suffix(".ply"), pc9)
    if fmt in ("npy", "both"):
        np.save(out.with_suffix(".npy"), np.asarray(pc9, np.float16))
    if meta is not None:
        with open(out.with_suffix(".norm.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def _timeout(signum, frame):
    raise TimeoutError("extract timeout")


def _one(a):
    """(mesh_path, out_stem, npoints, timeout_s, fmt) 처리. .att 마커로 poison 영구 skip."""
    mesh_path, out_path, npoints, timeout_s, fmt = a
    primary = str(Path(out_path).with_suffix(".ply" if fmt in ("ply", "both") else ".npy"))
    if os.path.exists(primary):
        return "skip"
    attempt = primary + ".att"
    if os.path.exists(attempt):
        return "poison"
    import signal
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        open(attempt, "w").close()
        try:
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(timeout_s)
        except Exception:
            pass
        pc, meta = extract_colored_pc(mesh_path, npoints, with_normals=True,
                                      return_meta=True)
        meta["source_mesh"] = os.path.basename(str(mesh_path))
        _save(out_path, pc, fmt, meta)
        try:
            os.remove(attempt)
        except Exception:
            pass
        return "ok"
    except Exception as e:
        try:
            os.remove(attempt)
        except Exception:
            pass
        return f"fail:{e.__class__.__name__}"
    finally:
        try:
            signal.alarm(0)
        except Exception:
            pass


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FTimeout
    ap = argparse.ArgumentParser(description="10,000점+RGB 포인트클라우드 추출")
    ap.add_argument("--input", required=True, help="메시 파일 또는 디렉토리")
    ap.add_argument("--out", required=True, help="출력 npy 파일(단일) 또는 디렉토리(배치)")
    ap.add_argument("--npoints", type=int, default=NPOINTS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=90, help="객체별 추출 제한(초)")
    ap.add_argument("--format", default="ply", choices=["ply", "npy", "both"],
                    help="ply=납품 규격(법선 포함) / npy=(N,9) float16 / both")
    args = ap.parse_args()

    inp = Path(args.input)
    if inp.is_file():                                   # 단일 파일
        pc, meta = extract_colored_pc(str(inp), args.npoints, with_normals=True,
                                      return_meta=True)
        meta["source_mesh"] = inp.name
        _save(args.out, pc, args.format, meta)
        print(f"OK {Path(args.out).with_suffix('')}.[{args.format}]  {pc.shape}")
        return

    # 디렉토리 배치: 재귀 탐색, 상대경로 유지, resume/poison skip
    meshes = sorted(p for p in inp.rglob("*") if p.suffix.lower() in MESH_EXTS)
    meshes = meshes[args.shard::args.num_shards]
    out_dir = Path(args.out)
    pri = ".ply" if args.format in ("ply", "both") else ".npy"
    existing = set()
    if out_dir.exists():
        existing = {str(p.relative_to(out_dir)) for p in out_dir.rglob("*")
                    if p.suffix in (".ply", ".npy", ".att")}
    tasks = []
    for m in meshes:
        rel = m.relative_to(inp).with_suffix(pri)
        if str(rel) in existing or str(rel) + ".att" in existing:
            continue
        tasks.append((str(m), str(out_dir / rel), args.npoints, args.timeout, args.format))
    print(f"[batch] 발견 {len(meshes):,} → 처리대상 {len(tasks):,} "
          f"(shard {args.shard}/{args.num_shards})", flush=True)

    stat = {"ok": 0, "skip": 0, "fail": 0, "poison": 0}
    done = 0
    CHUNK = 300
    for ci in range(0, len(tasks), CHUNK):              # 청크별 fresh pool (hang 방탄)
        chunk = tasks[ci:ci + CHUNK]
        try:
            pool = ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=50)
        except TypeError:
            pool = ProcessPoolExecutor(max_workers=args.workers)
        futs = [pool.submit(_one, t) for t in chunk]
        try:
            for fut in as_completed(futs, timeout=max(1200, args.timeout * 4)):
                try:
                    r = fut.result(timeout=1)
                except Exception:
                    r = "fail:worker"
                k = "fail" if r.startswith("fail") else r
                stat[k] = stat.get(k, 0) + 1
                done += 1
                if done % 500 == 0:
                    print(f"  [{done:,}/{len(tasks):,}] {stat}", flush=True)
        except FTimeout:
            print(f"  [{done:,}] 청크 타임아웃 → 워커 정리 후 다음", flush=True)
        for pr in list(getattr(pool, "_processes", {}).values()):
            try:
                pr.kill()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"DONE {stat}", flush=True)


if __name__ == "__main__":
    main()
