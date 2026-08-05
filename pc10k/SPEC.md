# pc10k 포인트클라우드 규격서 (OpenShape/Uni3D 호환)

> 3D 객체 검색 AI 학습 데이터 구축용 포인트클라우드 추출 규격.
> 본 규격은 OpenShape released 데이터(Uni3D 학습·평가 데이터) 및 Uni3D-g 모델과의
> **정량·정성·기능적 동등성 검증**을 통과한 파이프라인 기준이다. (검증: 2026-08, ETRI)
> 최종 검증: **Objaverse-LVIS 테스트셋 전수(46,205개)** — released 인코딩이 Uni3D 논문
> 수치(47.2%)를 재현(47.17%)하는 프로토콜에서, 본 규격 추출 데이터는 46.17%(−1.0%p).

## 1. 출력 규격

**정식(납품) 포맷: `.ply`** — binary_little_endian, 아래 property 순서:

```
element vertex 10000
property float x, y, z          # unit-sphere 정규화 좌표
property float nx, ny, nz       # 법선 (샘플점 소속 삼각형의 면법선, 단위벡터)
property uchar red, green, blue # 색 (0~255)
```

**학습용 병행 포맷: `.npy`** — `(10000, 9) float16` `[xyz | nx,ny,nz | rgb]`
(모델 입력 시 xyz+rgb 6채널만 사용 — Uni3D/OpenShape 호환)

| 항목 | 값 |
|------|-----|
| 점 개수 | **10,000** (다중 해상도 필요 시 본 규격에서 다운샘플) |
| xyz 정규화 | **unit-sphere**: centroid 제거 후 max-norm 나눔 (`‖p‖max = 1`) |
| 법선 | 면법선(face normal), 정규화·등방 스케일에 불변 |
| 좌표축 | 메시 원본 축 유지 — **YZ flip 등 축 변환 금지** |
| rgb 스케일 | PLY: uchar 0~255 / npy: float [0,1] |

※ 참고 — 기존 공개 데이터 포맷과의 관계: Cap3D `.ply`(ASCII, xyz+rgb, 법선 없음),
OpenShape `.npy`(dict xyz+rgb, 법선 없음). 본 규격은 **법선을 추가한 상위호환**이며
용역 요구(XYZ+Normal+RGB, SDA-3)를 충족한다.

## 2. 색 샘플링 규칙 (우선순위)

1. **UV 텍스처**: 면적가중 삼각형 샘플 + barycentric UV 보간 → 텍스처 룩업
   - UV v축 **뒤집지 않음** (Open3D triangle_uvs 는 이미 이미지 좌표계 — A/B 검증됨)
2. 정점색(vertex color) barycentric 보간
3. 재질 base color (다중 지오메트리는 서브메시별 albedo 텍스처/base color)
4. 폴백: 회색 `0.4` (OpenShape/Uni3D 관례)

로더: Open3D `read_triangle_mesh(enable_post_processing=True)` → 빈 메시 시
`read_triangle_model` 서브메시 병합(면적 비례 배분) → trimesh 최후 폴백.
비표준 텍스처(16bit 등)는 tensor API 경유 변환(직접 `np.asarray` 금지 — 프로세스 크래시).

## 3. 합격 기준 (검수) — OpenShape released 대비

| 검증 | 도구 | 기준 | 근거(실측) |
|------|------|------|-----------|
| 기하 | `validate/compare_openshape.py` | Chamfer(축정렬 후) **median < 0.03**, p90 < 0.05 | 실측 med 0.023, p90 0.033 (n=848+200) |
| 좌표축 | 〃 (48개 축변환 탐색) | 최적변환의 축순열이 identity (yaw 회전만 허용) | 실측 92% (0,1,2) 순열 |
| 색 | 〃 | 유색쌍 RGB-MAE **median < 0.2** | 실측 med 0.128~0.159 (공식 데이터셋 간 편차 0.152와 동급) |
| **기능(최종)** | `validate/functional_equiv.py` | Uni3D-g **zero-shot acc 차이 ±2%p 이내**, 임베딩 cos median ≥ 0.95 | **LVIS 테스트셋 전수(n=46,205)**: acc 46.17% vs released 47.17% (−1.00%p, 97.9% 유지), cos med 0.970, top-1 일치 83.6% |
| 정성 | `validate/render_sidebyside.py` | released 와 나란히 렌더 육안 검수 | 색·형태 재현 확인 |

※ 참고: 공개 데이터셋 간에도 편차가 존재한다 — Cap3D↔OpenShape 동일 객체 비교:
YZ 축 스왑, RGB-MAE med 0.152 (n=30). 따라서 합격 기준은 절대 일치가 아닌 임계값 방식.

## 4. 알려진 한계

- OpenShape released 색은 렌더(음영 포함) 기반, 본 규격은 순수 albedo → 체계적 색 차이 일부 존재 (기능 동등성엔 무영향 검증됨)
- released 와 일관된 yaw 180° 오프셋 존재 (Blender forward 규약 기인, 모델 성능 무영향 — 필요 시 `(x,y,z)→(−x,y,−z)` 보정)
- 무텍스처 단색 객체는 색 분산 기준 지표에서 '무색'으로 집계될 수 있음 (측정 한계)
