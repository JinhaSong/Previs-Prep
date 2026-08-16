# Previs-Prep

3D 객체 검색 AI 학습 데이터 구축을 위한 전처리 툴킷.
(생성형 AI 기반 사전시각화 과제 — 3D 객체 데이터셋 구축·검수용)

## 구성

| 모듈 | 내용 | 상태 |
|------|------|------|
| [`pc10k/`](pc10k/) | **10,000점+RGB 포인트클라우드 추출** (OpenShape/Uni3D 규격) + 검수 도구 3종 | ✅ 검증 완료 ([SPEC.md](pc10k/SPEC.md)) |
| [`render/hunyuan3d/`](render/hunyuan3d/) | Blender 멀티뷰 렌더링 (커스텀 채널·조명조건별) — Hunyuan3D-2.1 도구 수정판 | 사용 중 |
| [`caption/`](caption/) | **영어 캡션 생성 + 한국어 번역 + 한/영 정합성 검수** (Qwen 계열, 전 구간 Apache-2.0) | ✅ 검증 완료 ([SPEC.md](caption/SPEC.md)) |
| `docker/` | 컨테이너 환경 2종 (개발용 / 배포용 슬림), `/bin/bash` 진입 | ✅ 빌드·스모크 검증 |

## 컨테이너 이미지 2종

| 이미지 | dockerfile | 크기 | 용도 |
|--------|-----------|------|------|
| **`jinhasong/previs-prep:2.0`** | `docker/dockerfile.slim` | **13.1GB** | **배포·인도용** (Docker Hub 공개, nvidia/cuda:12.8.1-runtime 기반) |
| `previs-prep:dev` | `docker/dockerfile` | 59.1GB | 사내 개발용 (jinhasong/previs:1.0 상속, 미배포) |

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

## 빠른 시작

```bash
# 개발용 컨테이너 빌드·진입
docker compose up -d --build
docker exec -it previs /bin/bash

# 포인트클라우드 추출 — 단일 메시
python -m pc10k.extract --input model.glb --out out/model.npy

# 배치 (디렉토리 재귀, resume 지원, 워커/샤딩)
python -m pc10k.extract --input /data/meshes/ --out /data/pc10k/ --workers 8

# 검수 (OpenShape released 대비 — 기준·수치는 pc10k/SPEC.md)
python -m pc10k.validate.compare_openshape --n 200 --zip <test_datasets.zip> --glb_index <glb_index.json>
python -m pc10k.validate.functional_equiv --n 200        # Uni3D 모델 동등성 (GPU)
python -m pc10k.validate.render_sidebyside --n 4         # 정성 비교 PNG

# 캡션 생성 (멀티뷰 렌더 → 한/영 문장 쌍)
python caption/qwen_caption.py --uid_list uids.txt --render_root /data/renders --out caps.jsonl
python caption/qwen_translate_ko.py --in caps.jsonl --out caps_ko.jsonl
python caption/qc_bilingual.py --in caps_ko.jsonl --out defects.jsonl
```

## 출력 규격 (요약 — 상세는 [pc10k/SPEC.md](pc10k/SPEC.md))

- **정식: `.ply`** (binary LE) — x,y,z(float) + **nx,ny,nz(면법선)** + rgb(uchar) / 병행: `.npy` `(10000,9) float16`
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

**pc10k**: open3d(MIT) · trimesh(MIT) · scipy(BSD) · numpy(BSD)
**caption**: Qwen2.5-VL / Qwen3 / LaBSE — 전부 **Apache-2.0** (생성 결과물 활용·상용 배포 제약 없음)

> NLLB-200(CC-BY-NC)은 기술이전 부적합으로 배제했다.
> 예외는 위 `render/hunyuan3d/` 뿐이며, 별도 협의 대상이다.
