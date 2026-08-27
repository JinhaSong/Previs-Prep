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

from pathlib import Path

import numpy as np

TEX_DIRNAME = "textures"          # {출력}.usdc 옆에 textures/ 로 저장


def _mesh_list(obj):
    """trimesh Scene/Trimesh → [(name, geom)] (변환행렬 적용).

    Objaverse 에는 면이 없는 **포인트클라우드 전용 GLB** 가 섞여 있다.
    이를 실패로 버리면 변환 대상이 통째로 빠지므로, UsdGeomPoints 로 기록한다."""
    import trimesh
    if isinstance(obj, (trimesh.Trimesh, trimesh.PointCloud)):
        return [("mesh_0", obj)]
    out = []
    for i, (name, geom) in enumerate(obj.geometry.items()):
        is_mesh = isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        is_pts = isinstance(geom, trimesh.PointCloud) and len(geom.vertices) > 0
        if not (is_mesh or is_pts):
            continue
        g = geom.copy()
        # Scene graph 의 변환을 정점에 굽는다 (USD 에 계층을 만들지 않고 평탄화)
        for node in obj.graph.nodes_geometry:
            tf, gname = obj.graph[node]
            if gname == name:
                g.apply_transform(tf)
                break
        out.append((f"mesh_{i}", g))
    return out


def _save_texture(mesh, out_usd, idx):
    """baseColor 텍스처를 PNG 로 저장하고 상대경로 반환. 없으면 None."""
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None:
        return None
    try:
        tex_dir = Path(out_usd).parent / TEX_DIRNAME
        tex_dir.mkdir(parents=True, exist_ok=True)
        name = f"{Path(out_usd).stem}_{idx}.png"
        img.convert("RGBA").save(tex_dir / name)
        return f"./{TEX_DIRNAME}/{name}"
    except Exception:
        return None


def _base_color(mesh):
    """텍스처가 없을 때 쓸 단색. (r,g,b) 0~1 또는 None."""
    mat = getattr(mesh.visual, "material", None)
    for attr in ("baseColorFactor", "diffuse"):
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

    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())

    for i, (name, m) in enumerate(meshes):
        prim_path = f"/Root/{name}"

        if not hasattr(m, "faces") or len(getattr(m, "faces", [])) == 0:
            # 포인트클라우드 → UsdGeomPoints (정점색이 있으면 displayColor 로 보존)
            pts = UsdGeom.Points.Define(stage, prim_path)
            v = np.asarray(m.vertices, np.float32)
            pts.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v.tolist()]))
            scale = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
            pts.GetWidthsAttr().Set(Vt.FloatArray([scale * 0.002] * len(v)))
            pts.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
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

        tex_rel = _save_texture(m, dst, i) if uv is not None else None  # 최종명 기준
        color = _base_color(m)
        vcol = _vertex_colors(m)

        if tex_rel is None and color is None and vcol is not None:
            # 머티리얼이 전혀 없으면 정점색만이라도 살린다
            dc = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in vcol.tolist()]))
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
