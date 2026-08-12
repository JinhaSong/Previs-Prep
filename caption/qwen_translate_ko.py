#!/usr/bin/env python3
"""
Qwen2.5 한국어 번역 (3D 객체 캡션 전용)
=====================================

영어 캡션 → 자연스러운 한국어. 전 파이프라인을 Apache 2.0 으로 통일하고
(기존 NLLB-200 은 CC-BY-NC → 기술이전 부적합), 서술형 문장 품질을 높인다.

입력: jsonl ({uid, caption, ...}) 또는 {uid, en} 형식 모두 지원
출력: jsonl ({uid, caption, caption_ko, ...}) — 원본 필드 보존 + caption_ko 추가

배치 생성 + resume 지원.

실행:
  python caption/qwen_translate_ko.py --in caps.jsonl --out caps_ko.jsonl
  python caption/qwen_translate_ko.py --in "qwen_v4_*.jsonl" --out v4_ko.jsonl --batch 32
"""

import argparse
import glob
import json
import os

import torch

MODEL = "Qwen/Qwen3-8B"   # A/B 결과: 정상률 92.5% vs Qwen2.5-7B 85.0%, 한자혼입 0

SYS = "당신은 3D 에셋 검색용 설명문을 영어에서 한국어로 옮기는 전문 번역가입니다."

# few-shot: v1(규칙만)은 한자 혼입·토큰깨짐(Brown→ROWN)·오역(spine→척수)이 다발했다.
# 도메인 어휘(책등/양장본/노브)와 명사구 종결을 예시로 고정한다.
FEWSHOT = [
    ("A worn hardcover book with a brown cover and a red bookmark ribbon.",
     "갈색 표지와 빨간 책갈피 끈이 달린 낡은 양장본 책"),
    ("A Holy Bible with a black leather cover, gold trim, and \"HOLY BIBLE\" embossed on the spine.",
     "검은 가죽 표지에 금색 테두리가 있고 책등에 \"HOLY BIBLE\"이 양각된 성경"),
    ("A brown, curved object resembling a worm with a textured surface and small red dots.",
     "표면에 질감이 있고 작은 빨간 점이 박힌, 지렁이를 닮은 갈색 곡선형 물체"),
    ("Red guitar amplifier with a black front grille, control knobs on the top panel.",
     "검은색 전면 그릴과 상단 패널의 조절 노브가 있는 빨간색 기타 앰프"),
    ("A blue and white boat with red accents, including a red cabin and red masts.",
     "빨간 선실과 빨간 돛대 등 빨간색 포인트가 들어간 파란색과 흰색 배"),
]

RULES = (
    "다음 3D 객체 설명문을 한국어로 번역하세요.\n"
    "규칙:\n"
    "1. 반드시 한국어로만 출력. 한자·중국어·일본어 절대 금지.\n"
    "2. 브랜드명과 객체에 실제로 적힌 문구는 영문 원문 그대로 유지 (예: \"HOLY BIBLE\", Band-Aid).\n"
    "3. 명사구로 끝낼 것 (예: \"~이 있는 갈색 의자\"). '~입니다', '~있다' 같은 서술 종결 금지.\n"
    "4. 색상·재질·형태·부착물을 빠뜨리지 말 것.\n"
    "5. 직역투('~되어진', '~와 함께') 금지.\n"
    "6. 번역문 한 줄만 출력 (설명·따옴표·머리말 금지).\n"
)
# few-shot 없이 쓸 때를 위해 도메인 용어를 규칙으로 직접 준다
# (예시문이 없으면 '책등/양장본' 같은 어휘가 흔들리기 때문).
GLOSSARY = ("7. 도메인 용어: spine=책등, hardcover=양장본, knob=노브, trim=테두리, "
            "embossed=양각, ribbon=끈, foliage=잎, trunk=줄기, grip=손잡이.\n")


# few-shot 은 **멀티턴 대화**로 넣는다. 한 user 턴에 예시를 텍스트로 나열하면
# 모델이 그 나열을 '이어쓰기'로 인식해 예시 답변을 그대로 뱉는다
# (실측: 1,344건 중 3.6% → 멀티턴 전환 후 1.7%).
#
# 다만 멀티턴으로도 **결정론적 복사**가 남는다(batch=1 에서도 동일 출력 재현).
# 반대로 few-shot 을 빼면 복사는 사라지지만 종결어미 위반이 0 → 27.5% 로 폭증한다.
# 두 실패 모드가 배타적이므로, 기본은 few-shot 을 쓰고 **복사가 감지된 건만**
# few-shot 없이 다시 번역한다(translate 참조).
def build_messages(en, fewshot=True):
    if not fewshot:
        return [{"role": "system", "content": SYS + "\n\n" + RULES + GLOSSARY},
                {"role": "user", "content": f"영어: {en}"}]
    msgs = [{"role": "system", "content": SYS + "\n\n" + RULES}]
    for e, k in FEWSHOT:
        msgs.append({"role": "user", "content": f"영어: {e}"})
        msgs.append({"role": "assistant", "content": k})
    msgs.append({"role": "user", "content": f"영어: {en}"})
    return msgs

# 품질 게이트: CJK 한자 / 한글 부재 / 과도한 라틴문자 비율
import re as _re
_HANJA = _re.compile(r"[一-鿿぀-ヿ]")
_HANGUL = _re.compile(r"[가-힣]")
_FOREIGN = _re.compile(r"[\u0400-\u04FF\u0370-\u03FF\u0E00-\u0E7F]")  # 키릴·그리스·타이
_LATIN = _re.compile(r"[A-Za-z]{3,}")
_QUOTED_EN = _re.compile(r'"([^"]{2,40})"')


def _latin_leftover(ko, en):
    """번역되지 않고 남은 영어 단어. 단, 규칙상 유지해야 하는 것은 제외한다:
    원문에서 따옴표로 인용된 문구(객체 표면 문구)와 대문자로 시작하는 고유명사."""
    keep = set()
    for q in _QUOTED_EN.findall(en or ""):
        keep.update(_LATIN.findall(q))
    return [w for w in _LATIN.findall(ko) if w not in keep and not w[0].isupper()]


def _ngrams(s, n=3):
    s = "".join(s.split())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}


_FEWSHOT_NG = [(_ngrams(k), k) for _, k in FEWSHOT]
_FEWSHOT_EN = {e for e, _ in FEWSHOT}


def _copies_fewshot(ko, en, thr=0.30):
    """출력이 few-shot 답변과 과하게 겹치는지 (원문이 그 예시가 아닌 경우)."""
    if en in _FEWSHOT_EN:
        return False
    g = _ngrams(ko)
    for fg, _ in _FEWSHOT_NG:
        if len(g & fg) / max(len(g | fg), 1) >= thr:
            return True
    return False


# 색상은 3D 에셋 검색의 1차 속성이라 누락·오역을 생성 시점에 잡아야 한다.
# (실측: 사후 검수에만 두었더니 866건/5.4% 가 그대로 남았다 — pink→노란색 등)
COLOR_KO = {
    "red": ["빨간", "빨강", "붉은", "적색", "레드"],
    "blue": ["파란", "파랑", "푸른", "청색", "블루", "남색"],
    "green": ["초록", "녹색", "연두", "그린"],
    "yellow": ["노란", "노랑", "황색", "옐로"],
    "black": ["검은", "검정", "까만", "블랙"],
    "white": ["하얀", "흰", "백색", "화이트"],
    "brown": ["갈색", "밤색", "브라운"],
    "gray": ["회색", "그레이"], "grey": ["회색", "그레이"],
    "orange": ["주황", "오렌지"], "purple": ["보라", "퍼플"],
    "pink": ["분홍", "핑크"], "gold": ["금색", "금빛", "골드"],
    "silver": ["은색", "은빛", "실버"],
}


def _color_lost(ko, en):
    lo, lk = en.lower(), ko.lower()
    return [c for c, ks in COLOR_KO.items()
            if _re.search(rf"\b{c}\b", lo) and not any(k in lk for k in ks)]


def _quote_lost(ko, en):
    """객체 표면에 적힌 문구는 원문 유지가 규칙(2번)이다."""
    return [q for q in (x.strip(" .,!?;:") for x in _QUOTED_EN.findall(en))
            if len(q) >= 2 and q not in ko]


def ko_defects(s, en=None):
    """번역문 결함 목록 (빈 리스트면 정상). 재시도·QC 양쪽에서 사용.

    en 을 주면 few-shot 예시 복사(입력과 무관한 예시문 반복)까지 잡는다."""
    d = []
    if en is not None and _copies_fewshot(s, en):
        d.append("fewshot_copy")
    if "번역" in s or "：" in s:
        d.append("meta_leak")     # "~의 한국어 번역은:" 같은 메타 발화 누출
    if _HANJA.search(s):
        d.append("cjk")           # 한자/가나 혼입
    if _FOREIGN.search(s):
        d.append("foreign")       # 키릴 등 제3언어 혼입 (실측: "свет browm색")
    if en is not None and _latin_leftover(s, en):
        d.append("latin_left")    # 미번역 영어 잔존 (실측: "stylized한")
    if en is not None and _color_lost(s, en):
        d.append("color_lost")    # 색상 누락·오역 (검색 1차 속성)
    if en is not None and _quote_lost(s, en):
        d.append("quote_lost")    # 표면 문구 소실 (규칙 2)
    if not _HANGUL.search(s):
        d.append("no_hangul")     # 번역 실패 (원문 그대로 등)
    if len(s) < 4 or len(s) > 200:
        d.append("length")
    if s.rstrip(" .").endswith(("입니다", "있습니다", "있다", "이다", "합니다", "됩니다")):
        d.append("verb_ending")   # 종결어미 (명사구 규칙 위반)
    return d


# 결함 심각도. '다른 물건을 서술'(복사·의미)은 데이터셋을 오염시키지만,
# 종결어미 위반은 문체 문제일 뿐이다. 동등 취급하면 '완벽한 재시도만 수용'하게 되어
# 오히려 오염된 원본이 살아남는다(실측: 최악 6건 중 4건이 그렇게 유지됐다).
_SEVERITY = {"fewshot_copy": 100, "no_hangul": 100, "meta_leak": 50,
             "cjk": 20, "foreign": 20, "length": 20,
             "latin_left": 15, "color_lost": 15, "quote_lost": 10,
             "verb_ending": 5}


def defect_score(ko, en, sim=1.0, sem_thr=0.60):
    """작을수록 좋음. 재시도 수용 여부를 '더 나은가'로 판정하기 위한 점수."""
    s = sum(_SEVERITY.get(d, 10) for d in ko_defects(ko, en))
    if sim < sem_thr:
        s += 100
    return s


def load_records(pattern):
    recs = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            en = d.get("caption") or d.get("en")
            if en:
                d["_en"] = en
                recs.append(d)
    seen, uniq = set(), []          # 샤드 파일이 겹칠 수 있으므로 uid 중복 제거
    for r in recs:
        if r.get("uid") in seen:
            continue
        seen.add(r.get("uid"))
        uniq.append(r)
    if len(uniq) != len(recs):
        print(f"[dedup] {len(recs):,} → {len(uniq):,}", flush=True)
    return uniq


LABSE = "sentence-transformers/LaBSE"


def _require_cuda():
    """GPU 가 안 잡히면 즉시 중단한다.

    device_map="auto" 는 CUDA 초기화 실패 시 조용히 CPU 로 떨어진다. 8B 모델이
    CPU 에서 돌면 사실상 정지 상태인데 로그는 정상으로 보인다(실측: 컨테이너
    device cgroup 이 초기화돼 4분간 0건 진행). 실패를 눈에 보이게 만든다."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "[중단] CUDA 를 사용할 수 없습니다. CPU 폴백은 실질적으로 진행되지 않습니다.\n"
            "  컨테이너에서 nvidia-smi 확인 → 실패 시 `docker restart <container>`")


class Translator:
    def __init__(self, model=MODEL, semantic_gate=True, sem_thr=0.60):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _require_cuda()
        print(f"[load] {model} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="auto").eval()
        # 규칙 검사만으로는 '문법·문체는 멀쩡한데 다른 물건을 서술한' 실패를 못 잡는다.
        # LaBSE(번역쌍 판별 전용, 470M)를 생성 시점 게이트로 함께 올린다.
        self.sem_thr, self.lab_tok, self.lab = sem_thr, None, None
        if semantic_gate:
            from transformers import AutoModel
            print(f"[load] {LABSE} (의미 게이트) ...", flush=True)
            self.lab_tok = AutoTokenizer.from_pretrained(LABSE)
            self.lab = AutoModel.from_pretrained(LABSE).eval().to(self.model.device)

    @torch.no_grad()
    def _sem(self, ens, kos):
        """EN↔KO 의미 유사도. 게이트 미사용 시 전부 1.0 반환."""
        if self.lab is None:
            return [1.0] * len(ens)
        def emb(t):
            e = self.lab_tok(t, padding=True, truncation=True, max_length=128,
                             return_tensors="pt").to(self.lab.device)
            return torch.nn.functional.normalize(self.lab(**e).pooler_output, dim=-1)
        return (emb(ens) * emb(kos)).sum(1).tolist()

    @torch.no_grad()
    def _gen(self, texts, max_new_tokens, sample=False, fewshot=True):
        prompts = []
        for t in texts:
            msgs = build_messages(t, fewshot)
            try:      # Qwen3 계열: thinking 모드 끄기 (번역엔 불필요)
                pr = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                pr = self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(pr)
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, **kw,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        res = []
        for i in range(len(texts)):
            gen = out[i][enc["input_ids"].shape[1]:]
            s = self.tok.decode(gen, skip_special_tokens=True).strip()
            s = s.split("\n")[0].strip().strip('"').strip("'")
            res.append(s)
        return res

    def _scores(self, texts, res):
        sims = self._sem(texts, res)
        return [defect_score(r, t, v, self.sem_thr)
                for t, r, v in zip(texts, res, sims)]

    def translate(self, texts, max_new_tokens=96):
        """실패 유형별로 다른 재시도를 걸고, **점수가 개선될 때만** 교체한다.

        few-shot 유지/제거는 실패 모드가 배타적이라(복사 vs 문체) 둘 다 시도한다.
          - few-shot 제거      → 예시 복사·의미 오류 해소
          - few-shot + sampling → 한자·종결어미 해소
        """
        res = self._gen(texts, max_new_tokens)
        score = self._scores(texts, res)

        for kw in (dict(fewshot=False),                 # 복사 해소용
                   dict(fewshot=False, sample=True),    # 위가 실패하면 다양성 부여
                   dict(sample=True)):                  # 문체 해소용
            idxs = [i for i, s in enumerate(score) if s > 0]
            if not idxs:
                break
            sub = [texts[i] for i in idxs]
            cand = self._gen(sub, max_new_tokens, **kw)
            cs = self._scores(sub, cand)
            for i, r, c in zip(idxs, cand, cs):
                if c < score[i]:        # 더 나을 때만 교체 (완벽하지 않아도 수용)
                    res[i], score[i] = r, c
        return res


def repair(args):
    """결함 건만 다시 번역해 병합한다. 정상 건은 원본 그대로 보존."""
    recs = []
    for f in sorted(glob.glob(args.repair)):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    print(f"기존 {len(recs):,}건 로드", flush=True)

    tr = Translator(args.model, semantic_gate=not args.no_semantic_gate)
    # 1차 선별: 규칙 결함 → 그 다음 의미까지 확인 (LaBSE 는 결함 후보에만 돌려 비용 절감)
    cand = [r for r in recs if ko_defects(r.get("caption_ko", ""), r["caption"])]
    print(f"규칙 결함 {len(cand):,}건 재번역 대상", flush=True)

    ck = args.out + ".partial"          # 중단 대비: 배치마다 교정분을 남긴다
    prev = {}
    if os.path.exists(ck):
        for l in open(ck, encoding="utf-8"):
            try:
                d = json.loads(l)
                prev[d["uid"]] = d["caption_ko"]
            except Exception:
                pass
        print(f"[resume] 이전 보정 {len(prev):,}건 반영", flush=True)
        for r in recs:
            if r["uid"] in prev:
                r["caption_ko"], r["_repaired"] = prev[r["uid"]], True
        cand = [c for c in cand if c["uid"] not in prev]

    fixed = kept = 0
    cf = open(ck, "a", encoding="utf-8")
    for i in range(0, len(cand), args.batch):
        chunk = cand[i:i + args.batch]
        ens = [c["caption"] for c in chunk]
        olds = [c.get("caption_ko", "") for c in chunk]
        try:
            news = tr.translate(ens)
        except Exception as e:
            print(f"  [배치 실패 {e.__class__.__name__}] 원본 유지", flush=True)
            continue
        so, sn = tr._scores(ens, olds), tr._scores(ens, news)
        for c, n, a, b in zip(chunk, news, so, sn):
            if b < a:                       # 개선될 때만 반영
                c["caption_ko"], c["_repaired"] = n, True
                fixed += 1
                cf.write(json.dumps({"uid": c["uid"], "caption_ko": n},
                                    ensure_ascii=False) + "\n")
            else:
                kept += 1
        cf.flush()
        if (i // args.batch) % 10 == 0:
            print(f"  [{i + len(chunk):,}/{len(cand):,}] 교정 {fixed:,} 유지 {kept:,}", flush=True)

    cf.close()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ 저장: {args.out}  (교정 {fixed:,} / 대상 {len(cand):,}, 미개선 {kept:,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="입력 jsonl (glob 가능)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--done_from", default=None,
                    help="완료분으로 간주할 기존 jsonl glob (다른 서버 산출물 통합 resume)")
    ap.add_argument("--repair", default=None,
                    help="기존 출력 jsonl(glob). 결함 건만 재번역해 --out 으로 병합")
    ap.add_argument("--no_semantic_gate", action="store_true",
                    help="LaBSE 의미 게이트 비활성 (GPU 메모리 부족 시)")
    args = ap.parse_args()

    if args.repair:
        return repair(args)

    recs = load_records(args.inp)
    if args.num_shards > 1:
        recs = recs[args.shard::args.num_shards]
    if args.limit:
        recs = recs[:args.limit]
    done = set()
    for f in (glob.glob(args.done_from) if args.done_from else []):
        for l in open(f, encoding="utf-8"):
            try:
                d = json.loads(l)
                if d.get("caption_ko"):
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
    todo = [r for r in recs if r["uid"] not in done]
    print(f"번역 대상 {len(todo):,} / 전체 {len(recs):,} "
          f"(shard {args.shard}/{args.num_shards})", flush=True)
    if not todo:
        return

    tr = Translator(args.model, semantic_gate=not args.no_semantic_gate)
    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.batch):
            chunk = todo[s:s + args.batch]
            try:
                kos = tr.translate([r["_en"] for r in chunk])
            except Exception as e:
                kos = [f"FAIL:{e.__class__.__name__}"] * len(chunk)
            for r, ko in zip(chunk, kos):
                r.pop("_en", None)
                r["caption_ko"] = ko
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += len(chunk)
            if (s // args.batch) % 10 == 0:
                print(f"  [{n:,}/{len(todo):,}] 예: {kos[0][:60]}", flush=True)
    # 결함 집계 (한/영 정합성 검수 1차 지표)
    from collections import Counter
    cnt, tot = Counter(), 0
    for l in open(args.out, encoding="utf-8"):
        try:
            ko = json.loads(l).get("caption_ko", "")
        except Exception:
            continue
        tot += 1
        for d in ko_defects(ko):
            cnt[d] += 1
    print(f"✓ 저장: {args.out}  (총 {tot:,})", flush=True)
    print(f"  결함: {dict(cnt) or '없음'}  "
          f"정상률 {100*(tot-sum(cnt.values()))/max(tot,1):.1f}%", flush=True)


if __name__ == "__main__":
    main()
