#!/usr/bin/env python3
"""
Qwen2.5-VL 멀티뷰 3D 객체 캡션 생성 (Cap3D 대체 파이프라인)
=========================================================

Cap3D 는 [뷰별 BLIP-2 생성 → CLIP 선택 → GPT-4 통합] 3단계지만, Qwen2.5-VL 은
멀티이미지 입력을 네이티브 지원하므로 **여러 뷰를 한 번에 넣어 통합 캡션을 직접 생성**한다.
(= Cap3D 의 GPT-4 통합 단계를 오픈 모델로 대체)

프롬프트는 Cap3D 캡션 스타일(속성 기반 1문장: 종류·형태·색상·재질)에 맞춤.

입력: {render_root}/{uid}/*.png  (알파 → 흰 배경 합성)
출력: jsonl {uid, category, caption, n_views}

실행(previs env, GPU):
  python caption/qwen_caption.py --uid_list uids.txt --out caps.jsonl
  python caption/qwen_caption.py --sample_per_class 20 --out caps.jsonl   # LVIS 클래스별 N개
"""

import argparse
import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PREFIX = "uni3d_data/objaverse_lvis/"

# Cap3D 스타일 few-shot: 실제 Cap3D 캡션(핵심 명사구 중심, 객체 자체 속성만).
# 길이는 이미 Cap3D 와 동일(평균 15단어)했으나 어휘를 배경·부속물·과잉 디테일에 쓰는
# 경향이 있어, 예시로 "무엇을 쓰고 무엇을 버릴지"를 고정한다.
CAP3D_EXAMPLES = [
    "A worn hardcover book with a brown cover and a red bookmark ribbon.",
    "A vintage reel-to-reel tape recorder with two large reels and multiple control knobs.",
    "A stylized green apple with a stem and a single leaf.",
    "Red guitar amplifier with a black front grille, control knobs on the top panel.",
    "A Ferris wheel.",
]

PROMPT = (
    "These images show the SAME 3D object from different viewpoints. "
    "Write ONE short English caption of the object, in the exact style of these examples:\n"
    + "\n".join(f"- {e}" for e in CAP3D_EXAMPLES) +
    "\n\nRules:\n"
    "1. Always NAME what the object is (e.g. 'amplifier', 'air conditioning unit'), "
    "never a vague shape like 'a rectangular object' or 'a red box'.\n"
    "2. MOST IMPORTANT — make the caption specific enough to tell THIS object apart from "
    "other objects of the same kind. Include its distinguishing details: exact color "
    "combination, pattern or texture, material, attached or protruding parts, and any "
    "visible brand or label text.\n"
    "3. Describe ONLY the object — no background, ground, surroundings, lighting, or display setup.\n"
    "4. Never write 'is shown', 'in various angles', 'The object is', '3D model of'.\n"
    "5. One sentence, 15-20 words, starting directly with the object (e.g. 'A red...').\n"
    "Output only the caption sentence."
)


def load_views(obj_dir, num_views=8, size=448):
    """렌더 디렉토리에서 균등 간격 뷰 선택, RGBA → 흰배경 RGB, 리사이즈."""
    pngs = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
    if not pngs:
        raise FileNotFoundError(f"no renders: {obj_dir}")
    idx = np.linspace(0, len(pngs) - 1, min(num_views, len(pngs))).astype(int)
    out = []
    for i in idx:
        im = Image.open(pngs[i], encoding="utf-8").convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        out.append(Image.alpha_composite(bg, im).convert("RGB").resize((size, size)))
    return out


def sample_by_class(zip_path, render_root, per_class, seed=0):
    """LVIS 테스트셋에서 클래스별 per_class 개 (렌더 보유 객체만) 샘플."""
    z = zipfile.ZipFile(zip_path)
    lines = z.read(PREFIX + "lvis_testset.txt").decode().splitlines()
    by_cat = {}
    for ln in lines:
        p = ln.split(",")
        by_cat.setdefault(p[1], []).append(p[2])
    rng = np.random.default_rng(seed)
    picks = []
    for cat, uids in sorted(by_cat.items()):
        ok = [u for u in uids if os.path.isdir(os.path.join(render_root, u))]
        if not ok:
            continue
        sel = rng.permutation(len(ok))[:per_class]
        picks += [(ok[i], cat) for i in sel]
    return picks


def _require_cuda():
    """GPU 가 안 잡히면 즉시 중단한다.

    device_map="auto" 는 CUDA 초기화 실패 시 조용히 CPU 로 떨어진다. 8B 모델이
    CPU 에서 돌면 사실상 정지 상태인데 로그는 정상으로 보인다(실측: 컨테이너
    device cgroup 이 초기화돼 4분간 0건 진행). 실패를 눈에 보이게 만든다."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "[중단] CUDA 를 사용할 수 없습니다. CPU 폴백은 실질적으로 진행되지 않습니다.\n"
            "  컨테이너에서 nvidia-smi 확인 → 실패 시 `docker restart <container>`")


class QwenCaptioner:
    def __init__(self, model=MODEL, device_map="auto"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        _require_cuda()
        print(f"[load] {model} ...", flush=True)
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map=device_map).eval()

    @torch.no_grad()
    def caption(self, images, max_new_tokens=80, n=1, prompt=PROMPT):
        """n=1: greedy 1문장 / n>1: nucleus sampling 으로 서로 다른 표현 n개.

        용역 SDA-4 는 '객체당 복수 문장(다양한 표현·길이)' 을 요구하므로
        n>1 이면 첫 문장은 greedy(대표문), 나머지는 sampling(변형문)으로 만든다."""
        msgs = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in images] +
                 [{"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images, return_tensors="pt").to(self.model.device)
        plen = inputs["input_ids"].shape[1]

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        caps = [self.proc.decode(out[0][plen:], skip_special_tokens=True).strip()]
        if n > 1:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=True, top_p=0.9, temperature=0.9,
                                      num_return_sequences=n * 2 - 1)
            for o in out:
                c = self.proc.decode(o[plen:], skip_special_tokens=True).strip()
                if c and c not in caps:
                    caps.append(c)
                if len(caps) >= n:
                    break
        return caps if n > 1 else caps[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid_list", default=None, help="uid 목록 파일 (uid[,category] per line)")
    ap.add_argument("--sample_per_class", type=int, default=0, help="LVIS 클래스별 N개 샘플")
    ap.add_argument("--render_root", default="/nas/objaverse/multiview_images")
    ap.add_argument("--zip", default="/nas/integrated/trainvaltest/test_datasets.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--num_captions", type=int, default=1,
                    help="객체당 문장 수 (>1 이면 sampling 변형문 추가, 용역 SDA-4 대응)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1, help="여러 서버 분산 (picks[shard::N])")
    ap.add_argument("--done_from", default=None,
                    help="완료분으로 간주할 기존 jsonl glob (여러 파일·서버 산출물 통합 resume)")
    args = ap.parse_args()

    if args.sample_per_class:
        picks = sample_by_class(args.zip, args.render_root, args.sample_per_class)
    elif args.uid_list:
        picks = []
        for ln in open(args.uid_list, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                p = ln.split(",")
                picks.append((p[0], p[1] if len(p) > 1 else ""))
    else:
        raise SystemExit("--uid_list 또는 --sample_per_class 필요")
    if args.num_shards > 1:
        picks = picks[args.shard::args.num_shards]
    if args.limit:
        picks = picks[:args.limit]
    print(f"대상 {len(picks):,}개 (shard {args.shard}/{args.num_shards})", flush=True)

    # resume — 자기 출력 + (선택) 다른 산출물까지 합쳐 중복 생성을 막는다
    done = set()
    for f in (glob.glob(args.done_from) if args.done_from else []):
        for l in open(f, encoding="utf-8"):
            try:
                d = json.loads(l)
                if d.get("caption"):
                    done.add(d["uid"])
            except Exception:
                pass
    if done:
        print(f"[resume] 외부 산출물 {len(done):,}개 skip", flush=True)
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["uid"])
            except Exception:
                pass
        print(f"[resume] 기존 {len(done):,}개 skip", flush=True)

    cap = QwenCaptioner(args.model)
    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for uid, cat in picks:
            if uid in done:
                continue
            try:
                imgs = load_views(os.path.join(args.render_root, uid), args.num_views)
                c = cap.caption(imgs, n=args.num_captions)
                rec = {"uid": uid, "category": cat, "n_views": len(imgs)}
                if args.num_captions > 1:
                    rec["caption"] = c[0]          # 대표문 (하위호환)
                    rec["captions"] = c            # 복수문 전체
                else:
                    rec["caption"] = c
            except Exception as e:
                rec = {"uid": uid, "category": cat, "fail": f"{e.__class__.__name__}: {str(e)[:60]}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(picks)}]", flush=True)
    print(f"✓ 저장: {args.out}", flush=True)


if __name__ == "__main__":
    main()
