#!/usr/bin/env python3
"""메시 → USD 변환 코어 (OpenUSD / usd-core, Apache-2.0)

기존 사내 변환기(`/home/etri/omniverse-converter`)는 정점·면·법선만 기록해
**UV 와 머티리얼이 빠졌다**. 검색 결과를 뷰어에 띄울 때 텍스처가 사라지므로,
본 구현은 UsdPreviewSurface 로 재질을 함께 기록한다.

기록 항목
  - 지오메트리: points / faceVertexIndices / faceVertexCounts / normals
  - UV        : primvars:st (TexCoord2fArray)
  - 머티리얼  : UsdPreviewSurface (+ UsdUVTexture ← 추출한 baseColor 텍스처)
  - 폴백      : 텍스처 없으면 baseColorFactor, 그것도 없으면 정점색 displayColor
"""

import hashlib
from pathlib import Path

import numpy as np

TEX_DIRNAME = "textures"          # {출력}.usdc 옆에 textures/ 로 저장


# glTF COLOR_0 직접 파싱 — trimesh 는 머티리얼이 PBRMaterial(TextureVisuals) 이면
# COLOR_0 을 버린다(vertex_attributes 가 비고 visual.vertex_colors 도 없음).
# 실측: 표본 120개 중 **5.0% 가 텍스처 없이 COLOR_0 만으로 색을 표현**하며,
# 이를 놓치면 해당 객체의 색이 통째로 사라진다.
_COMP = {5121: (np.uint8, 255.0), 5123: (np.uint16, 65535.0), 5126: (np.float32, 1.0)}
_NCOMP = {"VEC3": 3, "VEC4": 4}


def _gltf_color0(path):
    """glTF/GLB 의 mesh 별 COLOR_0 을 읽어 {(정점수, 위치키): (N,3) float 0~1} 로 반환."""
    if Path(path).suffix.lower() not in (".glb", ".gltf"):
        return {}
    try:
        from pygltflib import GLTF2
        g = GLTF2().load(str(path))
        blob = g.binary_blob()
        if blob is None:
            return {}

        def read(acc_idx):
            acc = g.accessors[acc_idx]
            bv = g.bufferViews[acc.bufferView]
            dt, scale = _COMP.get(acc.componentType, (None, None))
            n = _NCOMP.get(acc.type)
            if dt is None or n is None:
                return None
            off = (bv.byteOffset or 0) + (acc.byteOffset or 0)
            a = np.frombuffer(blob, dtype=dt, count=acc.count * n, offset=off)
            return a.reshape(acc.count, n).astype(np.float32) / scale

        out = {}
        for m in g.meshes:
            for pr in m.primitives:
                ci = getattr(pr.attributes, "COLOR_0", None)
                pi = getattr(pr.attributes, "POSITION", None)
                if ci is None or pi is None:
                    continue
                col = read(ci)
                pos = read(pi)
                if col is None or pos is None:
                    continue
                out[(len(pos), _pos_key(pos))] = col[:, :3]
        return out
    except Exception:
        return {}                       # 파싱 실패 시 조용히 기존 경로로 진행


def _pos_key(v):
    """정점 배열의 지문 — trimesh 지오메트리와 glTF primitive 를 이어붙이는 데 쓴다.
    (이름·순서는 로더에 따라 달라질 수 있어 좌표 자체로 대조한다)"""
    v = np.asarray(v, np.float64)
    return tuple(np.round(v[0], 4).tolist()) + tuple(np.round(v.mean(0), 3).tolist())


def _mesh_list(obj):
    """trimesh Scene/Trimesh → [(name, geom)] (변환행렬 적용).

    Objaverse 에는 면이 없는 **포인트클라우드 전용 GLB** 가 섞여 있다.
    이를 실패로 버리면 변환 대상이 통째로 빠지므로, UsdGeomPoints 로 기록한다."""
    import trimesh
    if isinstance(obj, (trimesh.Trimesh, trimesh.PointCloud)):
        return [("mesh_0", obj, (len(obj.vertices), _pos_key(obj.vertices)))]
    out = []
    for i, (name, geom) in enumerate(obj.geometry.items()):
        is_mesh = isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        is_pts = isinstance(geom, trimesh.PointCloud) and len(geom.vertices) > 0
        if not (is_mesh or is_pts):
            continue
        key = (len(geom.vertices), _pos_key(geom.vertices))   # 변환 적용 전 기준
        g = geom.copy()
        # Scene graph 의 변환을 정점에 굽는다 (USD 에 계층을 만들지 않고 평탄화)
        for node in obj.graph.nodes_geometry:
            tf, gname = obj.graph[node]
            if gname == name:
                g.apply_transform(tf)
                break
        out.append((f"mesh_{i}", g, key))
    return out


def _save_texture(mesh, out_usd, idx, cache=None):
    """baseColor 텍스처를 PNG 로 저장하고 상대경로 반환. 없으면 None.

    같은 이미지를 여러 메시가 공유하는 경우가 많아, 내용 해시로 **중복 저장을 막는다**
    (미적용 시 실측: glb 텍스처 10개 → usd 36개로 팽창)."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        return None
    try:
        rgba = img.convert("RGBA")
        key = hashlib.md5(rgba.tobytes()).hexdigest()[:12]
        if cache is not None and key in cache:
            return cache[key]
        tex_dir = Path(out_usd).parent / TEX_DIRNAME
        tex_dir.mkdir(parents=True, exist_ok=True)
        name = f"{Path(out_usd).stem}_{key}.png"
        rgba.save(tex_dir / name)
        rel = f"./{TEX_DIRNAME}/{name}"
        if cache is not None:
            cache[key] = rel
        return rel
    except Exception:
        return None


def _base_color(mesh):
    """텍스처가 없을 때 쓸 단색. (r,g,b) 0~1 또는 None.

    trimesh PBRMaterial 은 glTF 원본에 따라 색을 서로 다른 속성에 담는다.
    `main_color` 만 가진 경우가 흔한데(실측), 이를 빠뜨리면 색이 회색으로
    떨어져 원본과 달라진다."""
    mat = getattr(mesh.visual, "material", None)
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        v = getattr(mat, attr, None) if mat is not None else None
        if v is not None and len(v) >= 3:
            v = np.asarray(v, dtype=float)
            if v.max() > 1.0:                      # 0~255 로 들어오는 경우
                v = v / 255.0
            return tuple(float(x) for x in v[:3])
    return None


def _uv(mesh):
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        return None
    return np.asarray(uv, dtype=np.float32)


def _vertex_colors(mesh):
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None or len(vc) != len(mesh.vertices):
        return None
    return np.asarray(vc, dtype=np.float32)[:, :3] / 255.0


def write_usd(src, dst, up_axis="Y", meters_per_unit=1.0):
    """src(glb/obj/ply/…) → dst(.usd/.usda/.usdc). 성공 시 True."""
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    src, dst = Path(src), Path(dst)
    loaded = trimesh.load(str(src), force="scene", process=False)
    meshes = _mesh_list(loaded)
    color0 = _gltf_color0(src)          # trimesh 가 버리는 COLOR_0 을 직접 확보
    if not meshes:
        raise ValueError("메시 없음")

    dst.parent.mkdir(parents=True, exist_ok=True)
    # 중단 시 반쪽 산출물이 남지 않도록 임시 파일에 쓴 뒤 교체한다.
    # USD 는 **확장자로 포맷을 판별**하므로 임시 파일도 .usdc 등을 유지해야 한다
    # (`.part` 를 붙이면 SdfLayer::_CreateNew 가 fileFormat 검증에서 실패).
    tmp = dst.with_name(f"{dst.stem}.__tmp__{dst.suffix}")
    tmp.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(tmp))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up_axis.upper() == "Y"
                           else UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)

    tex_cache = {}                       # 이미지 해시 → 상대경로 (중복 저장 방지)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    for i, (name, m, key) in enumerate(meshes):
        prim_path = f"/Root/{name}"

        if not hasattr(m, "faces") or len(getattr(m, "faces", [])) == 0:
            # 포인트클라우드 → UsdGeomPoints (정점색이 있으면 displayColor 로 보존)
            pts = UsdGeom.Points.Define(stage, prim_path)
            v = np.asarray(m.vertices, np.float32)
            pts.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v.tolist()]))
            scale = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
            pts.GetWidthsAttr().Set(Vt.FloatArray([scale * 0.002] * len(v)))
            pts.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            vcol = color0.get(key)
            if vcol is None:
                vcol = _vertex_colors(m)
            if vcol is not None:
                pts.GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*c) for c in vcol.tolist()]))
            pts.GetExtentAttr().Set(Vt.Vec3fArray(
                [Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())]))
            continue

        usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
        faces = np.asarray(m.faces)
        usd_mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*p) for p in np.asarray(m.vertices, np.float32).tolist()]))
        usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
        usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
        try:
            n = np.asarray(m.vertex_normals, np.float32)
            if len(n) == len(m.vertices):
                usd_mesh.GetNormalsAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*v) for v in n.tolist()]))
                usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        except Exception:
            pass
        # 뷰어가 바운딩을 잡을 수 있게 extent 기록
        bb = np.asarray(m.bounds, np.float32)
        usd_mesh.GetExtentAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(*bb[0].tolist()), Gf.Vec3f(*bb[1].tolist())]))

        uv = _uv(m)
        if uv is not None:
            st = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in uv.tolist()]))

        tex_rel = _save_texture(m, dst, i, tex_cache) if uv is not None else None
        color = _base_color(m)
        # COLOR_0 우선 — trimesh 의 main_color 는 요약값이라, 정점색으로 표현된
        # 객체를 단색으로 뭉갠다(실측: 크림/연노랑 객체가 균일 회색이 됐다).
        vcol = color0.get(key)
        if vcol is None:
            vcol = _vertex_colors(m)
        if vcol is not None and len(vcol) != len(m.vertices):
            vcol = None                 # 정점수 불일치 시 사용하지 않는다

        # 정점색이 있으면 항상 displayColor 로 남긴다(머티리얼 유무와 무관).
        if vcol is not None:
            dc = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in vcol.tolist()]))
        if tex_rel is None and color is None and vcol is None:
            continue

        mat_path = f"{prim_path}/Material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        if tex_rel:
            reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/diffuseTexture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                reader.ConnectableAPI(), "result")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb")
        elif vcol is not None:
            # 정점색 구동 머티리얼 — 상수색으로 두면 Blender 등에서 연결이 끊긴다.
            # USD 표준대로 displayColor primvar 를 읽어 diffuseColor 에 잇는다.
            rd = UsdShade.Shader.Define(stage, f"{mat_path}/displayColorReader")
            rd.CreateIdAttr("UsdPrimvarReader_float3")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                rd.ConnectableAPI(), "result")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*(color or (0.5, 0.5, 0.5))))

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(usd_mesh).Apply(usd_mesh.GetPrim())
        UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)

    stage.GetRootLayer().Save()
    del stage                      # 레이어 핸들을 닫고 나서 교체
    tmp.replace(dst)
    return True
