# Previs-Prep

3D 객체 검색 AI 학습 데이터 구축을 위한 전처리 툴킷.
(생성형 AI 기반 사전시각화 과제 — 3D 객체 데이터셋 구축·검수용)

## 구성

| 모듈 | 내용 | 상태 |
|------|------|------|
| [`pc10k/`](pc10k/) | **10,000점+RGB 포인트클라우드 추출** (OpenShape/Uni3D 규격) + 검수 도구 3종 | ✅ 검증 완료 ([SPEC.md](pc10k/SPEC.md)) |
| [`render/hunyuan3d/`](render/hunyuan3d/) | Blender 멀티뷰 렌더링 (커스텀 채널·조명조건별) — Hunyuan3D-2.1 도구 수정판 | 사용 중 |
| `docker/` | 컨테이너 환경 (compose, `/bin/bash` 진입) | |

## 빠른 시작

```bash
# 컨테이너 빌드·진입
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

## 렌더링 도구 주의사항

`render/hunyuan3d/`는 **Tencent Hunyuan NON-COMMERCIAL 라이선스** 헤더를 가진
Hunyuan3D-2.1 파생물입니다. 상업/납품 산출물 포함 여부는 발주기관과 사전 협의 필요.
(원형은 Objaverse-XL Blender 렌더 스크립트 계열)

## 라이선스 (pc10k 모듈 의존성)

open3d(MIT) · trimesh(MIT) · scipy(BSD) · numpy(BSD) — 허용적 라이선스만 사용.
