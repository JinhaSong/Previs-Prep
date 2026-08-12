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


def arch_list():
    import torch
    archs = torch.cuda.get_arch_list()
    if not any(a in archs for a in ("sm_120", "compute_120")):
        raise RuntimeError(
            f"sm_120(RTX 50xx) 커널 없음 — {archs}. "
            "cu128 이상으로 빌드할 것 (--build-arg TORCH_CUDA=cu128)")
    return archs


print("\n[GPU 아키텍처 지원]")
check("torch arch_list", arch_list)


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
