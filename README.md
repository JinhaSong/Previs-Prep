# Previs-Prep

3D 객체 검색 AI 학습 데이터 구축을 위한 전처리 툴킷.
(생성형 AI 기반 사전시각화 과제 — 3D 객체 데이터셋 구축·검수용)

## 구성

| 모듈 | 내용 | 상태 |
|------|------|------|
| [`pc10k/`](pc10k/) | **10,000점+RGB 포인트클라우드 추출** (OpenShape/Uni3D 규격) + 검수 도구 3종 | ✅ 검증 완료 ([SPEC.md](pc10k/SPEC.md)) |
| [`render/hunyuan3d/`](render/hunyuan3d/) | Blender 멀티뷰 렌더링 (커스텀 채널·조명조건별) — Hunyuan3D-2.1 도구 수정판 | 사용 중 |
| [`caption/`](caption/) | **영어 캡션 생성 + 한국어 번역 + 한/영 정합성 검수** (Qwen 계열, 전 구간 Apache-2.0) | ✅ 검증 완료 ([SPEC.md](caption/SPEC.md)) |
| [`usd/`](usd/) | **USD 변환** (OpenUSD) — glb/obj/ply/stl → usdc, UV·머티리얼 포함 | ✅ 검증 완료 ([SPEC.md](usd/SPEC.md)) |
| `docker/` | 컨테이너 환경 2종 (개발용 / 배포용 슬림), `/bin/bash` 진입 | ✅ 빌드·스모크 검증 |

## 컨테이너 이미지 3종

| 이미지 | dockerfile | 크기 | 용도 | 배포 |
|--------|-----------|------|------|------|
| **`jinhasong/previs-prep:2.0`** | `docker/dockerfile.slim` | **13.1GB** | **배포·인도용** (nvidia/cuda:12.8.1-runtime 기반) | Docker Hub |
| `previs-prep-validate:2.0` | `docker/dockerfile.validate` | +1.6GB | USD 산출물 **검증 전용** (bpy 추가) | **미배포** |
| `previs-prep:dev` | `docker/dockerfile` | 59.1GB | 사내 개발용 (jinhasong/previs:1.0 상속) | 미배포 |

> 검증 이미지를 분리한 이유는 용량이 아니라 **라이선스**다. `bpy` 는 GPL-3.0 이라
> 인도 이미지에 넣으면 GPL 바이너리 재배포가 된다(Omniverse Kit 을 뺀 이유와 동일).
> Blender 의 *산출물 데이터* 는 GPL 대상이 아니므로 USD 산출물 인도에는 영향이 없다.

> **2.0** = 포인트클라우드(pc10k) + 문장 생성·검수(caption) 모듈을 모두 포함한 버전.

두 이미지는 동일 입력에 **동일 결과**를 낸다 (검증: 아래 스모크 테스트 수치 일치).
슬림 쪽은 모듈이 `/opt/previs-prep` 에 설치되어 레포 마운트 없이 바로 실행된다.

```bash
# 배포용 슬림 빌드·실행
# Docker Hub 에서 바로 사용
docker pull jinhasong/previs-prep:2.0
docker run --rm -it --gpus all -v /data:/data jinhasong/previs-prep:2.0

# 또는 직접 빌드
docker build -f docker/dockerfile.slim -t previs-prep:2.0 .
```

> 빌드 시 `docker/verify_image.py` 가 자동 실행되어 의존성·python 정식릴리스 여부·
> CUDA 빌드 버전(sm_120 지원)을 확인하고, 하나라도 실패하면 **이미지를 만들지 않는다.**
> 이 검사로 실제로 5건의 결함을 빌드 단계에서 잡았다(pip 부재, cu128 미적용,
> libgomp 누락, transformers 메이저 이탈, python RC 버전).

## 사용 방법

### 0. 실행 환경

```bash
# 모델 가중치 캐시를 호스트에 두어 재다운로드를 막는다 (Qwen2.5-VL 7B + Qwen3 8B + LaBSE ≈ 35GB)
mkdir -p ~/hf-cache

docker run --rm -it --gpus all \
  -v /data:/data \
  -v ~/hf-cache:/root/.cache/huggingface \
  jinhasong/previs-prep:2.0
```

모듈은 이미지 `/opt/previs-prep` 에 설치되어 있어 **레포 마운트 없이** `python -m <모듈>` 로 바로 실행된다.
가중치는 이미지에 굽지 않고 최초 실행 시 자동 내려받는다(위 캐시 볼륨 권장).

| 단계 | GPU | 비고 |
|------|-----|------|
| pc10k 추출 | 불필요 (CPU 멀티프로세스) | `--workers` 로 코어 수만큼 |
| 영어 캡션 | **필요** (VRAM ≥ 20GB) | Qwen2.5-VL-7B |
| 한국어 번역 | **필요** (VRAM ≥ 20GB) | Qwen3-8B + LaBSE 동시 적재 |
| 검수·병합 | 불필요 (`--no_semantic` 시) | LaBSE 사용 시 GPU 권장 |

> **RTX 50xx(sm_120) 사용 시 주의**: 반드시 본 이미지(cu128)를 쓸 것. cu124 빌드는
> sm_120 커널이 없어 **오류 없이 잘못된 결과를 생성**한다(실측: 10,215건 전량 무효).

### 1. 전체 파이프라인

```
3D 메시 ──┬─> [pc10k] 포인트클라우드 (.ply/.npy/.norm.json)
          │
          └─> 멀티뷰 렌더(8뷰+) ──> [caption] 영어 문장 ──> 한국어 번역
                                                              │
                                            [qc_bilingual] 검수 ──> [merge] 인도 데이터셋
```

### 2. 포인트클라우드 추출 (SDA-3)

```bash
# 단일 메시
python -m pc10k.extract --input model.glb --out out/model --format both

# 배치 — 디렉토리 재귀 탐색, resume 지원(기존 산출물 skip)
python -m pc10k.extract --input /data/meshes/ --out /data/pc10k/ \
    --format both --workers 16 --timeout 90

# 여러 서버 분산 (서버마다 --shard 를 0,1,2… 로)
python -m pc10k.extract --input /data/meshes/ --out /data/pc10k/ \
    --shard 0 --num_shards 4 --workers 16
```

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--format` | `ply` | `ply` / `npy` / `both` |
| `--npoints` | `10000` | 점 개수 |
| `--workers` | CPU 수 | 프로세스 병렬 |
| `--timeout` | `90` | 메시당 제한(초). 초과 시 `.att` 마커로 영구 skip |
| `--shard` / `--num_shards` | `0` / `1` | 서버·노드 분산 |

**산출물** (`--format both` 기준)

```
model.ply         납품 규격 (binary LE, xyz + 면법선 + rgb)
model.npy         (10000, 9) float16
model.norm.json   정규화 파라미터 — 원본 좌표 복원용
```

> `.norm.json` 은 **매 추출마다 반드시 함께 보관**할 것. centroid·scale 이
> 샘플된 점 기준으로 계산되고 샘플링이 난수라, 나중에 원본 메시로 역산할 수 없다.
> 복원식: `p_orig = p_norm * scale + offset`

### 3. 추출물 검수

```bash
python -m pc10k.validate.compare_openshape --n 200 \
    --zip <test_datasets.zip> --glb_index <glb_index.json>   # 기하·색 정량 비교
python -m pc10k.validate.functional_equiv --n 200            # Uni3D 모델 동등성 (GPU)
python -m pc10k.validate.render_sidebyside --n 4             # 정성 비교 PNG
```

### 4. 영어 문장 생성 (SDA-4)

```bash
# uid 목록으로 생성 — 객체당 3문장 (SDA-4 "복수 문장" 요구)
python -m caption.qwen_caption \
    --uid_list uids.txt --render_root /data/renders \
    --num_captions 3 --out caps_s0.jsonl

# GPU 여러 장에 분산 (장비마다 --shard 를 바꿔 동시 실행)
python -m caption.qwen_caption \
    --uid_list uids.txt --render_root /data/renders --num_captions 3 \
    --shard 0 --num_shards 4 --out caps_s0.jsonl \
    --done_from "caps_*.jsonl"
```

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--num_captions` | `1` | **SDA-4 대응 시 3 이상 지정** |
| `--num_views` | `8` | 입력 뷰 수 |
| `--done_from` | — | 다른 산출물까지 합쳐 resume (**중복 생성 방지**) |
| `--sample_per_class` | — | uid 목록 대신 클래스별 N개 샘플 |

> `--done_from` 을 빼면 프로세스마다 자기 출력 파일만 보고 재개하므로
> 같은 객체를 중복 생성한다(실측 2,180건). 여러 프로세스로 돌릴 때는 항상 지정할 것.

### 5. 한국어 번역

```bash
python -m caption.qwen_translate_ko \
    --in "caps_s*.jsonl" --out caps_ko.jsonl --batch 16

# 검수 기준을 올린 뒤, 결함 건만 다시 번역해 병합 (전량 재생성 불필요)
python -m caption.qwen_translate_ko \
    --repair "caps_ko*.jsonl" --in x --out caps_ko_fixed.jsonl
```

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--batch` | `32` | 배치 크기 (VRAM 20GB 기준 16 권장) |
| `--repair` | — | 결함 건만 재번역·병합. `.partial` 로 중단 후 재개 |
| `--no_semantic_gate` | off | LaBSE 의미 게이트 비활성 (VRAM 부족 시) |

### 6. 한/영 정합성 검수 (SDA-4 검수 체계)

```bash
python -m caption.qc_bilingual --in caps_ko_fixed.jsonl --out defects.jsonl
```

형식·내용·의미 3층 검사 후 **결함 건만** `defects.jsonl` 로 추출 → 사람 검수 대상.
`--no_semantic` 으로 LaBSE 없이(CPU만) 형식·내용 검사만 돌릴 수 있다.

### 7. 병합 → 인도 데이터셋

```bash
python -m caption.merge_captions \
    --in "caps_ko_fixed*.jsonl" --out dataset.jsonl \
    --require_ko --target 20430 --min_per_class 20
```

uid 중복 제거 · 실패 레코드 제거 · 커버리지/클래스 분포 보고.

> **병합은 검수가 아니다.** 병합 과정에서 중복 해소·필드 정리가 일어나므로,
> 인도 판정은 항상 **병합된 최종 파일**에 대해 6단계를 다시 돌려 내린다.

**최종 스키마**

```jsonl
{"uid": "...", "category": "...", "n_views": 8,
 "caption": "A worn hardcover book with a brown cover and a red bookmark ribbon.",
 "captions": ["...", "...", "..."],
 "caption_ko": "갈색 표지와 빨간 책갈피 끈이 달린 낡은 양장본 책"}
```

`caption` 은 항상 대표문 1개이므로, 복수 문장을 쓰지 않는 소비자도 그대로 읽을 수 있다.

### 8. USD 변환

```bash
# 디렉토리 배치 (재귀, resume, 병렬 — GPU 불필요)
python -m usd.convert --input /data/meshes/ --out /data/usd/ --workers 8 \
    --fail_log /data/usd_fail.jsonl
```

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--ext` | `usdc` | `usdc`(바이너리) / `usda`(텍스트) |
| `--workers` | CPU 수 | 프로세스 병렬 |
| `--timeout` | `120` | 파일당 제한(초) |
| `--overwrite` | off | 기본은 resume |
| `--shard`/`--num_shards` | `0`/`1` | 서버 분산 |

지오메트리·법선에 더해 **UV(`primvars:st`)와 UsdPreviewSurface 머티리얼**(텍스처 PNG 추출 포함)을
기록하고, 면이 없는 포인트클라우드 GLB 는 `UsdGeomPoints` 로 변환한다.
엔진은 usd-core(Pixar OpenUSD, Apache-2.0)라 **별도 컨테이너·NGC 계정이 필요 없다.**

#### 산출물 검증 (별도 이미지)

USD 자체 API 로는 정상으로 보이는 결함(텍스처 중복, 색 소실)이 **Blender 실임포트**에서 잡힌다.
실제로 이 검증으로 결함 4건을 발견했다 — 상세·수치는 [usd/SPEC.md](usd/SPEC.md) §5.1.

```bash
# 1) 검증 이미지 빌드 — 인도 이미지 위에 bpy 만 얹는다 (레지스트리에 올리지 않음)
docker build -f docker/dockerfile.validate -t previs-prep-validate:2.0 .

#    CUDA 이미지에서 처음부터 쌓으려면
docker build -f docker/dockerfile.validate \
    --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04 \
    -t previs-prep-validate:2.0 .

# 2) 검증 실행 — 원본 GLB 와 USD 산출물을 대조
docker run --rm -v /data:/data previs-prep-validate:2.0 \
    python -m usd.validate_blender --glb_dir /data/meshes --usd_dir /data/usd \
        --n 20 --out /data/usd_validate.jsonl
```

판정 기준은 **원본 유형에 따라 다르다** — 텍스처 원본은 이미지 연결을, 상수색 원본은
색값 근사를 본다. "연결 유무"로만 보면 상수색이 정답인 객체를 실패로 오판한다.

| 옵션 | 설명 |
|------|------|
| `--n` | 표본 개수 (`0` = 전수) |
| `--convert_missing` | USD 가 없으면 즉석 변환 후 검사 |
| `--out` | 결과 jsonl |

> **Blender 4.1.1 이상 필요** — 3.0.x 는 임포트 단계에서 UsdUVTexture 를 만들지 않아
> 정상 파일을 실패로 오판한다(도구가 실행을 차단한다). 4.x/5.x 는 임포트는 정상이나
> 렌더 평가에서 USD 머티리얼이 누락되는 회귀가 있어, **렌더 기반 대조가 필요하면
> Blender 3.6 계열(Python 3.10)로 별도 구성**해야 한다. 본 도구는 렌더를 쓰지 않는다.
>
> ⚠️ 이 이미지는 **인도물이 아니다.** `bpy`(GPL-3.0) 때문에 레지스트리에 올리지 않는다.

### 9. 개발용 컨테이너 (사내)

```bash
docker compose up -d --build
docker exec -it previs /bin/bash
```

## 출력 규격 (요약 — 상세는 [pc10k/SPEC.md](pc10k/SPEC.md))

- **정식: `.ply`** (binary LE) — x,y,z(float) + **nx,ny,nz(면법선)** + rgb(uchar) / 병행: `.npy` `(10000,9) float16`
- **`.norm.json`** — 정규화 파라미터(`offset`, `scale`, AABB). 복원식 `p_orig = p_norm * scale + offset`
  (용역 SDA-3 좌표계 정합 · DIV-3 3D-2D 투영 검증용. 검증: 복원 오차 1.3e-03)
- `--format ply|npy|both` (기본 ply). 모델 입력엔 xyz+rgb만 사용 (Uni3D/OpenShape 호환)
- xyz: unit-sphere 정규화(centroid+max-norm), **축 변환(YZ flip 등) 금지**
- rgb: UV 텍스처 직접 샘플링(v-flip 없음) → 정점색 → base color → 회색 0.4
- 검증 실측 (**LVIS 테스트셋 전수 46,205개**): Uni3D-g zero-shot **acc 46.17% vs
  OpenShape released 47.17%** (−1.0%p, 97.9% 유지 — 합격 기준 ±2%p 통과),
  임베딩 cos med 0.970, Chamfer med 0.023. released 측정치가 Uni3D 논문(47.2%)을
  재현하므로 평가 프로토콜 자체도 검증됨

## 캡션 규격 (요약 — 상세는 [caption/SPEC.md](caption/SPEC.md))

- 영어 캡션: `Qwen2.5-VL-7B-Instruct` 에 **8뷰 동시 입력** → 통합 캡션 1문장 직접 생성
  (Cap3D 의 [BLIP-2 → CLIP 선택 → GPT-4 통합] 3단계를 오픈 모델 1단계로 대체)
- 한국어: `Qwen3-8B` + 도메인 few-shot → 명사구 종결, 브랜드·표면 문구 원문 유지
- 검수: **형식(규칙) · 내용(숫자·인용문구·색상 사전) · 의미(LaBSE cos)** 3층 자동 검수
- 검증 실측 (**20,430쌍 전수**, 1,156 LVIS 클래스 전체):
  - text→3D 검색 **R@1 24.14% / R@5 50.15% / R@10 61.96% / MedR 5**
    → Cap3D(23.64 / 48.29 / 60.35 / 6) 대비 **전 지표 상회** (갤러리 20,245 동일 조건)
  - 한/영 정합성 **정상률 96.82%**, LaBSE cos 중앙 **0.820**, 의미 오류 **0.01%**(3건)
  - 잔여 결함 649건(3.18%)은 결함 목록으로 추출 → 사람 검수

## 렌더링 도구 주의사항

`render/hunyuan3d/`는 **Tencent Hunyuan NON-COMMERCIAL 라이선스** 헤더를 가진
Hunyuan3D-2.1 파생물입니다. 상업/납품 산출물 포함 여부는 발주기관과 사전 협의 필요.
(원형은 Objaverse-XL Blender 렌더 스크립트 계열)

## 스모크 테스트 (두 이미지 동일 결과)

| 항목 | previs-prep:dev (59.1GB) | **previs-prep:2.0** (13.1GB) |
|------|-----------------|------------------|
| python | 3.11.9 | 3.11.15 |
| torch / CUDA | 2.11.0+cu128 / 12.8 | 2.11.0+cu128 / 12.8 |
| GPU arch | sm_75~**sm_120** | sm_75~**sm_120** |
| CLI 5개 모듈 | ok | ok |
| pc10k 추출 (동일 GLB) | 10,000점 / max-norm 1.0002 / rgb 0.547 | 10,000점 / max-norm 1.0001 / rgb 0.548 |
| 한/영 검수 (동일 200건) | 정상률 98.00% | 정상률 98.00% |

## 라이선스

| 모듈 | 라이선스 | 납품 가능 |
|------|----------|-----------|
| **pc10k** | open3d(MIT) · trimesh(MIT) · scipy·numpy(BSD) | ✅ |
| **caption** | Qwen2.5-VL / Qwen3 / LaBSE — 전부 **Apache-2.0** | ✅ |
| **usd** | usd-core(Pixar OpenUSD, Apache-2.0) · trimesh·pygltflib(MIT) | ✅ |
| **render/hunyuan3d** | **Tencent Hunyuan NON-COMMERCIAL** | ⚠️ 협의 필요 |

> NLLB-200(CC-BY-NC)은 기술이전 부적합으로 배제했다.
> USD 변환은 초기에 Omniverse Kit(NVIDIA 라이선스, NGC 계정 필요)을 검토했으나
> 계약 조건과 맞지 않아 **usd-core 로 대체**했다. 협의가 필요한 모듈은
> `render/hunyuan3d/` 뿐이며, 디렉토리 단위로 분리되어 있어 제외해도 나머지는 그대로 동작한다.
