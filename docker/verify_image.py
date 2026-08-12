#!/usr/bin/env python3
"""이미지 빌드 자체 검증 — 실패 시 빌드를 중단시킨다.

의존성이 깨진 이미지가 인도되면 사용 기관에서 원인 파악에 시간을 쓰게 된다.
빌드 단계에서 import·버전·GPU 아키텍처 지원 범위를 확인해 조기에 실패시킨다.

GPU 는 빌드 시점에 없을 수 있으므로 **런타임 가용성이 아니라 빌드에 포함된
아키텍처 목록**을 확인한다(sm_120 누락이 RTX 50xx 무효 생성의 원인이었다).
"""
import sys

FAIL = []


def check(name, fn):
    try:
        print(f"  {name:22s} {fn()}")
    except Exception as e:
        FAIL.append(f"{name}: {e.__class__.__name__}: {e}")
        print(f"  {name:22s} ✗ {e.__class__.__name__}: {e}")


print(f"[python] {sys.version.split()[0]}  ({sys.executable})")
if any(x in sys.version.lower() for x in ("rc", "alpha", "beta")):
    FAIL.append(f"python {sys.version.split()[0]} 는 정식 릴리스가 아님 "
                "(Ubuntu 22.04 기본 python3.11 은 3.11.0rc1 — deadsnakes PPA 사용)")
    print(f"  ✗ 정식 릴리스 아님")

print("\n[pc10k 의존성]")
check("numpy", lambda: __import__("numpy").__version__)
check("open3d", lambda: __import__("open3d").__version__)
check("trimesh", lambda: __import__("trimesh").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("h5py", lambda: __import__("h5py").__version__)
check("PIL", lambda: __import__("PIL").__version__)

print("\n[caption 의존성]")
check("torch", lambda: __import__("torch").__version__)
check("torchvision", lambda: __import__("torchvision").__version__)
check("transformers", lambda: __import__("transformers").__version__)
check("accelerate", lambda: __import__("accelerate").__version__)


def cuda_build():
    """빌드 시점에는 GPU 가 없으므로 CUDA **빌드 버전**으로 판정한다.

    torch.cuda.get_arch_list() 는 내부적으로 is_available() 을 먼저 보고
    GPU 가 없으면 무조건 [] 를 반환하므로 빌드 단계 검사에 쓸 수 없다.
    cu128(CUDA 12.8) 이상이면 sm_120 커널이 포함된다."""
    import torch
    v = torch.version.cuda
    if v is None:
        raise RuntimeError("CPU 전용 torch — CUDA 빌드를 설치할 것")
    if tuple(int(x) for x in v.split(".")[:2]) < (12, 8):
        raise RuntimeError(
            f"torch CUDA {v} — RTX 50xx(sm_120) 커널 없음. "
            "cu128 이상으로 빌드할 것 (--build-arg TORCH_CUDA=cu128). "
            "베이스 이미지에 torch 가 있으면 --force-reinstall 없이는 무시된다")
    if torch.cuda.is_available():           # 런타임 실행 시에는 실제 목록까지 확인
        archs = torch.cuda.get_arch_list()
        if not any("120" in a for a in archs):
            raise RuntimeError(f"sm_120 커널 없음 — {archs}")
        return f"{torch.__version__} / CUDA {v} / arch {archs}"
    return f"{torch.__version__} / CUDA {v}  (빌드 시 GPU 없음 → arch 목록은 런타임 검증)"


print("\n[GPU 아키텍처 지원]")
check("torch CUDA build", cuda_build)


def qwen_classes():
    import transformers as t
    for c in ("Qwen2_5_VLForConditionalGeneration", "AutoModelForCausalLM"):
        if not hasattr(t, c):
            raise RuntimeError(f"{c} 없음 — transformers>=4.51 필요")
    return "Qwen2.5-VL / CausalLM 클래스 확인"


print("\n[모델 클래스]")
check("transformers Qwen", qwen_classes)

if FAIL:
    print("\n✗ 검증 실패:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("\n✓ 이미지 검증 통과")
