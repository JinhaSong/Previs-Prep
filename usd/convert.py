#!/usr/bin/env python3
"""USD 변환 (OpenUSD / usd-core) — 메인 컨테이너에서 바로 실행
================================================================

메시(glb/gltf/obj/ply/stl/…) → USD. 엔진은 **usd-core(Pixar OpenUSD, Apache-2.0)**
+ trimesh 로, 메인 이미지에 그대로 들어간다(별도 런타임·계정 불필요).

> 초기에는 Omniverse Kit(별도 컨테이너) 을 검토했으나, NVIDIA Omniverse License 라
> 계약의 허용 라이선스 조건과 맞지 않고 NGC 계정·EULA 동의가 필요해 배제했다.
> 대신 기존 사내 변환기가 빠뜨렸던 **UV·머티리얼을 보강**해 품질 격차를 줄였다.

지오메트리 + UV + UsdPreviewSurface 머티리얼(텍스처 포함)을 기록한다.
상세 규격: usd/SPEC.md

실행:
  # 단일 파일
  python -m usd.convert --input model.glb --out model.usdc
  # 디렉토리 배치 (재귀, resume, 병렬)
  python -m usd.convert --input /data/meshes/ --out /data/usd/ --workers 8
  # 여러 서버 분산
  python -m usd.convert --input /data/meshes/ --out /data/usd/ --shard 0 --num_shards 4
"""

import argparse
import json
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SRC_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off", ".dae"}
OUT_EXT = {".usd", ".usda", ".usdc", ".usdz"}


def _one(args):
    """워커 프로세스 1건 처리. (입력, 상태, 메시지) 반환.

    변환은 외부 메시를 다루므로 손상 파일에서 무한 루프·크래시가 날 수 있다.
    프로세스 분리 + SIGALRM 으로 개별 건을 제한한다."""
    src, dst, timeout, overwrite, up_axis = args
    src, dst = Path(src), Path(dst)
    try:
        if dst.exists() and dst.stat().st_size > 0 and not overwrite:
            return str(src), "skip", ""
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError(f"{timeout}s 초과")))
        signal.alarm(timeout)
        try:
            from usd.usd_writer import write_usd
            write_usd(src, dst, up_axis=up_axis)   # 원자적 쓰기는 writer 내부 처리
        finally:
            signal.alarm(0)
        return str(src), "ok", f"{dst.stat().st_size:,}B"
    except Exception as e:
        dst.with_name(f"{dst.stem}.__tmp__{dst.suffix}").unlink(missing_ok=True)
        return str(src), "fail", f"{e.__class__.__name__}: {str(e)[:160]}"


def collect(inp, out, ext):
    p = Path(inp)
    if p.is_file():
        return [(p, Path(out))]
    files = [q for q in sorted(p.rglob("*"))
             if q.is_file() and q.suffix.lower() in SRC_EXT]
    return [(q, Path(out) / q.relative_to(p).with_suffix(f".{ext}")) for q in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="메시 파일 또는 디렉토리")
    ap.add_argument("--out", required=True, help="출력 파일 또는 디렉토리")
    ap.add_argument("--ext", default="usdc", help="usdc|usda|usd (기본 usdc, 바이너리)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=int, default=120, help="파일당 제한(초)")
    ap.add_argument("--overwrite", action="store_true", help="기존 산출물 재변환")
    ap.add_argument("--up_axis", default="Y", choices=["Y", "Z"],
                    help="USD upAxis (glTF 계열은 Y)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--fail_log", default=None, help="실패 목록 jsonl")
    args = ap.parse_args()

    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        raise SystemExit("[중단] usd-core 미설치 — pip install usd-core trimesh pygltflib")

    pairs = collect(args.input, args.out, args.ext)
    if args.num_shards > 1:
        pairs = pairs[args.shard::args.num_shards]
    if not pairs:
        raise SystemExit(f"변환 대상 없음: {args.input}")
    print(f"대상 {len(pairs):,}개 → .{args.ext} "
          f"(shard {args.shard}/{args.num_shards}, workers {args.workers})", flush=True)

    jobs = [(str(s), str(d), args.timeout, args.overwrite, args.up_axis) for s, d in pairs]
    ok = skip = fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            src, st, msg = fu.result()
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
                fails.append({"input": src, "error": msg})
                print(f"  ✗ {Path(src).name}: {msg}", flush=True)
            if i % 200 == 0 or i == len(jobs):
                print(f"  [{i:,}/{len(jobs):,}] 변환 {ok:,} 건너뜀 {skip:,} 실패 {fail:,}",
                      flush=True)

    print(f"\n✓ 완료 — 변환 {ok:,} / 건너뜀 {skip:,} / 실패 {fail:,} "
          f"(실패율 {100*fail/max(len(jobs),1):.2f}%)", flush=True)
    if fails and args.fail_log:
        with open(args.fail_log, "w", encoding="utf-8") as f:
            for r in fails:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  실패 목록: {args.fail_log}", flush=True)
    sys.exit(1 if fail and not ok else 0)


if __name__ == "__main__":
    main()
