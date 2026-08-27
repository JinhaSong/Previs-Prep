# USD 변환 규격서 (OpenUSD)

> 3D 메시(glb/gltf/obj/ply/stl/dae)를 **USD** 로 변환한다.
> 엔진은 **usd-core(Pixar OpenUSD)** + trimesh 이며, 메인 이미지에 포함되어
> 별도 컨테이너·계정 없이 바로 실행된다.

## 1. 엔진 선정 — Omniverse Kit 을 쓰지 않은 이유

| | usd-core (**채택**) | Omniverse Kit |
|---|---|---|
| 라이선스 | **Pixar OpenUSD (Apache-2.0)** ✅ | NVIDIA Omniverse License ⚠️ |
| 취득 | pip | **NGC 계정 + EULA 동의** 필요 |
| 배치 | 메인 이미지 내장 | 별도 컨테이너(~10GB) |
| 재질·텍스처 | 본 구현에서 보강 (§3) | 기본 지원 |

계약이 Apache/BSD/MIT 계열을 요구하므로 Omniverse Kit 은 배제했다.
Kit 의 우위였던 **UV·머티리얼 처리는 본 구현에서 직접 기록**해 격차를 없앴다.

> 사내에 있던 기존 변환기(`/home/etri/omniverse-converter`)도 usd-core 기반이지만
> **정점·면·법선만 기록**해 UV·머티리얼이 빠져 있었다. 뷰어에서 텍스처가 사라지므로
> 본 규격에서 이를 보강했다.

## 2. 출력 규격

```
{out}/{상대경로}.usdc          # 기본. 바이너리 USD (--ext usda/usd 로 변경 가능)
{out}/textures/{stem}_{i}.png  # 추출된 baseColor 텍스처
```

| 항목 | 값 |
|------|-----|
| upAxis | **Y** (glTF 계열 기준, `--up_axis Z` 로 변경) |
| metersPerUnit | 1.0 |
| defaultPrim | `/Root` |
| 입력 확장자 | `glb, gltf, obj, ply, stl, off, dae` |

디렉토리 배치 시 **입력 기준 상대경로 구조를 그대로 유지**한다
(`meshes/a/b.glb` → `usd/a/b.usdc`).

## 3. 기록 내용

| 대상 | USD 표현 |
|------|----------|
| 지오메트리 | `UsdGeomMesh` — points / faceVertexIndices / faceVertexCounts / extent |
| 법선 | `normals` (vertex interpolation) |
| **UV** | `primvars:st` (TexCoord2fArray, vertex) |
| **머티리얼** | `UsdPreviewSurface` + `UsdUVTexture` ← 추출 PNG |
| 폴백 1 | 텍스처 없음 → `diffuseColor` 단색(baseColorFactor) |
| 폴백 2 | 머티리얼 없음 → `displayColor` (정점색) |
| **포인트클라우드** | `UsdGeomPoints` — points / widths / displayColor |

**Scene graph 는 평탄화한다.** trimesh Scene 의 노드 변환행렬을 정점에 굽고
`/Root/mesh_{i}` 로 나열한다. 검색·뷰어 용도에서는 계층 보존보다 단순함이 유리하다.

> **포인트클라우드 지원 이유**: Objaverse 에는 면이 없는 포인트클라우드 전용 GLB 가
> 섞여 있다(실측 10건 중 1건). 이를 실패로 버리면 변환 대상이 통째로 빠지므로
> `UsdGeomPoints` 로 기록한다.

## 4. 실행

```bash
# 단일 파일
python -m usd.convert --input model.glb --out model.usdc

# 디렉토리 배치 (재귀, resume, 병렬)
python -m usd.convert --input /data/meshes/ --out /data/usd/ --workers 8

# 여러 서버 분산 + 실패 목록
python -m usd.convert --input /data/meshes/ --out /data/usd/ \
    --shard 0 --num_shards 4 --fail_log /data/usd_fail.jsonl
```

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--ext` | `usdc` | `usdc`(바이너리) / `usda`(텍스트) / `usd` |
| `--workers` | CPU 수 | 프로세스 병렬 (GPU 불필요) |
| `--timeout` | `120` | 파일당 제한(초). 손상 메시 대비 |
| `--overwrite` | off | 기본은 resume — 기존 산출물 건너뜀 |
| `--up_axis` | `Y` | `Y`/`Z` |
| `--shard`/`--num_shards` | `0`/`1` | 서버·노드 분산 |
| `--fail_log` | — | 실패 목록 jsonl (입력 경로 + 사유) |

## 5. 견고성

- **원자적 쓰기**: 임시 파일에 쓴 뒤 `replace`. 중단돼도 반쪽 USD 가 남지 않는다.
  USD 는 **확장자로 포맷을 판별**하므로 임시 파일도 유효한 USD 확장자를 유지한다
  (`.part` 를 붙이면 `SdfLayer::_CreateNew` 가 fileFormat 검증에서 실패 — 실측).
- **프로세스 격리 + 타임아웃**: 손상 메시에서 크래시·무한 루프가 나도 해당 건만 실패한다.
- **resume**: 산출물이 있고 크기 > 0 이면 건너뜀.
- **실패 격리**: `--fail_log` 로 입력 경로와 사유를 남긴다. 전체 실패율을 확인하고 진행할 것.

## 6. 검증 실측

동일 GLB 20건(Objaverse) 변환, 컨테이너 `jinhasong/previs-prep:2.0` 기준:

| 항목 | 결과 |
|------|------|
| 변환 성공 | **20 / 20 (실패 0%)** |
| 소요 | 14.4초 (8 workers) |
| Mesh | 1,127개 — 그중 **UV 보유 987** |
| Material | **1,127** (전 메시 바인딩) |
| UVTexture | 112 (텍스처 보유 메시) |
| Points | 1 (포인트클라우드 GLB) |
| 텍스처 PNG | 112개 추출 |
| resume 재실행 | 20건 전부 건너뜀 ✅ |

기존 사내 변환기와 동일 파일 비교: **UV 0 → 2, Material 0 → 2** (해당 파일 기준).
