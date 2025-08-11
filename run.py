import os
import re
import io
import json
import time
import base64
import hashlib
import pathlib
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image

# LiteLLM
try:
    from litellm import batch_completion
except Exception:
    batch_completion = None

# =====================================
# App / Paths
# =====================================
st.set_page_config(page_title="다문제 수집 · Judge · OCR (LiteLLM)", page_icon="🧠", layout="wide")

APP_DATA_DIR = pathlib.Path("data")
RECORDS_DIR = APP_DATA_DIR / "records"
Q_CORRECT_DIR = APP_DATA_DIR / "correct_images"
Q_INCORRECT_DIR = APP_DATA_DIR / "incorrect_images"
ANS_IMG_DIR = APP_DATA_DIR / "answer_images"
CTX_IMG_DIR = APP_DATA_DIR / "context_images"
for d in [RECORDS_DIR, Q_CORRECT_DIR, Q_INCORRECT_DIR, ANS_IMG_DIR, CTX_IMG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default models
MODEL_SOLVE = "gemini/gemini-2.5-pro"
MODEL_JUDGE = "gemini/gemini-2.5-flash"
MODEL_OCR   = "gemini/gemini-2.5-pro"

SUPPORTED_IMG_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

# =====================================
# Utils
# =====================================
def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)

def infer_mime(name: str) -> str:
    n = name.lower()
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".webp"): return "image/webp"
    return "image/jpeg"

def to_data_url(b: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(b).decode('utf-8')}"

def load_files(files) -> List[Dict]:
    out=[]
    for f in files or []:
        b = f.read(); f.seek(0)
        name = sanitize(getattr(f,"name", f"img_{sha16(b)}"))
        mime = getattr(f,"type", None) or infer_mime(name)
        out.append({"name": name, "mime": mime, "bytes": b})
    return out

def messages_for_images(prompt_text: str, images: List[Tuple[bytes, str]]) -> List[Dict]:
    content = [{"type":"text","text":prompt_text}]
    for b, m in images:
        content.append({"type":"image_url","image_url":{"url": to_data_url(b,m)}})
    return [{"role":"user","content": content}]

def extract_text_from_resp(resp) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        try:
            ch0 = resp.choices[0]
            msg = getattr(ch0,"message",{})
            if isinstance(msg, dict): return msg.get("content","") or ""
            return getattr(msg,"content","") or ""
        except Exception:
            return ""

def build_solve_prompt() -> str:
    return (
        "역할: 다양한 학과목 문제 풀이 보조\n"
        "지시:\n"
        "1) 문제의 조건, 변수, 단위를 정리하고 각 질문을 독립적으로 풀이하세요.\n"
        "2) 계산과 근거와 공식을 단계별로 명확히 제시하세요. 객관식이면 소거법 후 남은 보기 검증.\n"
        "3) 마지막에 한 줄로 '최종 정답:'을 표기하세요.\n"
        "추가 가정은 최소화하고 보이는 정보만 사용하세요."
    )

def build_ocr_prompt() -> str:
    return (
        "역할: OCR 엔진\n"
        "지시사항: 업로드된 이미지에서 보이는 텍스트를 있는 그대로 추출하세요.\n"
        "- 줄바꿈과 번호와 구분을 가능한 보존합니다.\n"
        "- 수식과 특수기호는 유사한 텍스트로 표기합니다.\n"
        "- 중요 사항: 추출된 텍스트만 출력하고, 어떠한 해설이나 설명도 덧붙이지 마세요."
    )

def build_judge_prompt(model_solution: str, golden_answer_text: str, has_images: bool) -> str:
    """
    Golden answer 기반으로 모델 정답 여부를 엄밀 판정
    JSON만 출력
    """
    img_note = (
        "Golden Answer에는 텍스트 외에도 정답을 설명하거나 증명하는 이미지가 포함될 수 있습니다. "
        "이미지에 적힌 수치, 단위, 표기, 선택지 라벨을 필요시 읽어 반영하세요.\n"
        if has_images else
        ""
    )
    return (
        "You are an impartial grading judge. Your task is to determine whether the model's final answer is correct "
        "based strictly on the provided Golden Answer.\n\n"
        + img_note +
        "Evaluation Procedures:\n"
        "1) Extract the model's FINAL answer from the solution transcript. The final line often contains a cue like "
        "'최종 정답:'. If missing, infer the last decisive numeric or algebraic result.\n"
        "2) Normalize both the model's final answer and the Golden Answer:\n"
        "   - Ignore non semantic punctuation and whitespace.\n"
        "   - Unify case and common synonyms such as true와 참, false와 거짓.\n"
        "   - For numbers, ignore formatting differences like commas and leading zeros.\n"
        "   - Units must be semantically equivalent. If Golden Answer includes a unit, the model must match or be explicitly equivalent.\n"
        "3) Numeric tolerance:\n"
        "   - Default absolute tolerance is 1e-6, or relative tolerance 1 percent for non integer values, whichever is larger.\n"
        "4) Algebraic or format equivalence:\n"
        "   - Equivalent algebraic expressions are acceptable if identically valued.\n"
        "   - For multiple choice, letter or number labels must match the Golden Answer unless a clear mapping is proven.\n"
        "5) If the model's answer cannot be unambiguously mapped to the Golden Answer, mark as incorrect.\n\n"
        "Output strictly the following JSON object and nothing else:\n"
        '{"verdict":"correct|incorrect","model_final_answer":"...","explanation":"<=200 chars justification"}\n\n'
        "[MODEL_SOLUTION]\n" + model_solution + "\n\n"
        "[GOLDEN_ANSWER_TEXT]\n" + golden_answer_text + "\n"
        "If images are included below, consider their content as part of the Golden Answer."
    )

def parse_judge_json(text: str) -> Dict:
    try:
        data = json.loads(text)
        v = str(data.get("verdict","")).lower()
        if v not in ("correct","incorrect"):
            v = "correct" if "correct" in v else "incorrect"
        return {
            "verdict": v,
            "model_final_answer": data.get("model_final_answer",""),
            "explanation": data.get("explanation","")
        }
    except Exception:
        v = "correct" if "correct" in text.lower() and "incorrect" not in text.lower() else "incorrect"
        return {"verdict": v, "model_final_answer":"", "explanation": text[:200]}

# 해시 기반 그룹 ID 생성. 동일 입력이면 동일 group_id로 고정
def compute_group_id(q_items: List[Dict], ctx_items: List[Dict]) -> str:
    h = hashlib.sha256()
    for it in q_items or []:
        h.update(it["bytes"])
        h.update(it["name"].encode("utf-8"))
    for it in ctx_items or []:
        h.update(it["bytes"])
        h.update(it["name"].encode("utf-8"))
    return h.hexdigest()[:16]

def save_group_problem_images(q_items: List[Dict], indices: List[int], folder: pathlib.Path, group_id: str, p: int) -> List[str]:
    paths=[]
    for i, idx in enumerate(indices, start=1):
        it = q_items[idx]
        ext = ".png" if it["mime"]=="image/png" else ".jpg"
        path = folder / f"g{group_id}_p{p:02d}_{i:02d}{ext}"
        if not path.exists():
            with open(path,"wb") as w:
                w.write(it["bytes"])
        paths.append(str(path))
    return paths

def save_group_context_images(ctx_items: List[Dict], group_id: str) -> List[str]:
    paths=[]
    for i, it in enumerate(ctx_items, start=1):
        ext = ".png" if it["mime"]=="image/png" else ".jpg"
        path = CTX_IMG_DIR / f"g{group_id}_ctx_{i:02d}{ext}"
        if not path.exists():
            with open(path,"wb") as w:
                w.write(it["bytes"])
        paths.append(str(path))
    return paths

def save_group_answer_images(files, group_id: str, p: int) -> List[str]:
    saved=[]
    ts = int(time.time())
    for j, f in enumerate(files or [], start=1):
        b = f.read(); f.seek(0)
        name = sanitize(getattr(f,"name", f"a_{sha16(b)}"))
        mime = getattr(f,"type", None) or infer_mime(name)
        ext = ".png" if mime=="image/png" else ".jpg"
        out = ANS_IMG_DIR / f"g{group_id}_p{p:02d}_ans_{ts}_{j:02d}{ext}"
        if not out.exists():
            with open(out,"wb") as w:
                w.write(b)
        saved.append(str(out))
    return saved

def save_answer_items_from_bytes(items: List[Dict], group_id: str, p: int, tag: str="ansj") -> List[str]:
    """
    judge 단계에 업로드된 정답 이미지처럼 이미 바이트로 보유한 항목을 저장
    content hash 기반 파일명으로 중복 방지
    """
    saved=[]
    for it in items or []:
        ext = ".png" if it["mime"]=="image/png" else ".jpg"
        fn = f"g{group_id}_p{p:02d}_{tag}_{sha16(it['bytes'])}{ext}"
        out = ANS_IMG_DIR / fn
        if not out.exists():
            with open(out,"wb") as w:
                w.write(it["bytes"])
        saved.append(str(out))
    return saved

def reset_to_initial(keep_settings=True):
    """
    experimental_rerun 없이 초기화. 설정만 유지하고 재실행 유도
    """
    keep_keys = {}
    if keep_settings:
        for k in ["api_env_name","api_key_value","model_solve","model_judge","model_ocr",
                  "n_problems","context_on","year_text"]:
            keep_keys[k] = st.session_state.get(k)
    st.session_state.clear()
    for k,v in keep_keys.items():
        st.session_state[k] = v

    try:
        st.rerun()
        return
    except Exception:
        pass

    try:
        ts = str(int(time.time()))
        try:
            st.query_params["_"] = ts
        except Exception:
            st.experimental_set_query_params(**{"_": ts})
    except Exception:
        st.warning("초기화는 완료되었지만 자동 새로고침에 실패했습니다. 새로고침 해주세요.")

# =====================================
# State
# =====================================
def init_state():
    st.session_state.setdefault("api_env_name", "GEMINI_API_KEY")
    st.session_state.setdefault("api_key_value", os.getenv("GEMINI_API_KEY",""))
    st.session_state.setdefault("model_solve", MODEL_SOLVE)
    st.session_state.setdefault("model_judge", MODEL_JUDGE)
    st.session_state.setdefault("model_ocr", MODEL_OCR)

    st.session_state.setdefault("n_problems", 1)
    st.session_state.setdefault("context_on", False)
    st.session_state.setdefault("year_text", "")

    st.session_state.setdefault("q_items", [])      
    st.session_state.setdefault("ctx_items", [])    
    st.session_state.setdefault("assignments", {})  

    st.session_state.setdefault("gt_answers", {})       
    st.session_state.setdefault("solve_outputs", {})    
    st.session_state.setdefault("judge_raw", {})        
    st.session_state.setdefault("judge_json", {})       

    st.session_state.setdefault("group_id", "")
    st.session_state.setdefault("saved_paths", {})      
    st.session_state.setdefault("question_texts", {})   
    st.session_state.setdefault("ctx_saved_paths", [])
    st.session_state.setdefault("ctx_ocr_text", "")
    st.session_state.setdefault("save_dir_for_current", "correct")

    # Judge용 정답 이미지 바이트 저장소 {p: [ {name, mime, bytes}, ... ]}
    st.session_state.setdefault("judge_ans_items", {})

init_state()

# =====================================
# Sidebar
# =====================================
st.sidebar.title("⚙️ 설정")
st.sidebar.text_input("API Key Env Var Name", value=st.session_state.api_env_name, key="api_env_name")
st.sidebar.text_input("API Key Value", value=st.session_state.api_key_value, type="password", key="api_key_value")
st.sidebar.selectbox("풀이 모델", [st.session_state.model_solve, "openai/gpt-4o-mini"], index=0, key="model_solve")
st.sidebar.selectbox("Judge 모델", [st.session_state.model_judge, "openai/gpt-4o-mini"], index=0, key="model_judge")
st.sidebar.selectbox("OCR 모델", [st.session_state.model_ocr, "openai/gpt-4o-mini"], index=0, key="model_ocr")
st.sidebar.number_input("문제 개수", min_value=1, max_value=50, value=st.session_state.n_problems, step=1, key="n_problems")
st.sidebar.checkbox("컨텍스트(문단) 존재", value=st.session_state.context_on, key="context_on")
st.sidebar.text_input("연도(예: 2023)", value=st.session_state.year_text, key="year_text")

# API Key 주입
if st.session_state.api_env_name and st.session_state.api_key_value:
    os.environ[st.session_state.api_env_name] = st.session_state.api_key_value

# =====================================
# 0) 업로드
# =====================================
st.title("🧠 다문제 수집 · Judge · OCR (LiteLLM)")

st.markdown("### 0) 이미지 업로드")
if st.session_state.context_on:
    c1, c2 = st.columns(2)
    with c1:
        q_files = st.file_uploader("문제 스크린샷 업로드 (미리보기 없음, 다중)", type=["png","jpg","jpeg","webp"],
                                   accept_multiple_files=True, key="q_uploads")
        if q_files:
            st.session_state.q_items = load_files(q_files)
            st.caption("문제 파일: " + (", ".join([x["name"] for x in st.session_state.q_items]) or "없음"))
    with c2:
        ctx_files = st.file_uploader("문단(컨텍스트) 스크린샷 업로드 (미리보기 없음, 다중)", type=["png","jpg","jpeg","webp"],
                                     accept_multiple_files=True, key="ctx_uploads")
        if ctx_files:
            st.session_state.ctx_items = load_files(ctx_files)
            st.caption("컨텍스트 파일: " + (", ".join([x["name"] for x in st.session_state.ctx_items]) or "없음"))
else:
    q_files = st.file_uploader("문제 스크린샷 업로드 (미리보기 없음, 다중)", type=["png","jpg","jpeg","webp"],
                               accept_multiple_files=True, key="q_uploads_alone")
    if q_files:
        st.session_state.q_items = load_files(q_files)
        st.caption("문제 파일: " + (", ".join([x["name"] for x in st.session_state.q_items]) or "없음"))

if not st.session_state.q_items:
    st.info("문제 이미지를 먼저 업로드하세요.")
    st.stop()

# =====================================
# 1) 문제 구성
# =====================================
st.markdown("### 1) 문제 구성 (문제별 이미지 선택)")
names = [it["name"] for it in st.session_state.q_items]
for p in range(1, st.session_state.n_problems+1):
    default_sel = st.session_state.assignments.get(p, [])
    st.session_state.assignments[p] = st.multiselect(
        f"문제 {p}에 포함될 이미지",
        options=list(range(len(names))),
        default=default_sel,
        format_func=lambda idx: names[idx],
        key=f"assign_{p}"
    )

# =====================================
# 2) 풀이와 Judge
# =====================================
st.markdown("***")
st.markdown("### 2) 모든 문제 풀이 시작 (배치) + LLM as a Judge")

# Judge용 정답 입력 UI: 좌측 텍스트, 우측 이미지
for p in range(1, st.session_state.n_problems+1):
    if not st.session_state.assignments.get(p):
        continue
    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        st.session_state.gt_answers[p] = st.text_area(
            f"[문제 {p}] 정답 텍스트 입력 LLM as a Judge용 필수",
            value=st.session_state.gt_answers.get(p, ""),
            height=140,
            key=f"gt_{p}"
        )
    with c2:
        judge_files = st.file_uploader(
            f"[문제 {p}] 정답 이미지 업로드 LLM as a Judge용 선택",
            type=["png","jpg","jpeg","webp"],
            accept_multiple_files=True,
            key=f"gt_imgs_{p}"
        )
        if judge_files:
            st.session_state.judge_ans_items[p] = load_files(judge_files)
        names_j = [it["name"] for it in st.session_state.judge_ans_items.get(p, [])]
        st.caption("판정용 정답 이미지: " + (", ".join(names_j) if names_j else "없음"))

run_btn = st.button("🧠 모든 문제 풀이와 채점 실행", use_container_width=True, type="primary")
if run_btn:
    if batch_completion is None:
        st.error("litellm 미설치: pip install litellm")
        st.stop()

    missing = [p for p in range(1, st.session_state.n_problems+1)
               if st.session_state.assignments.get(p) and not (st.session_state.gt_answers.get(p) or "").strip()]
    if missing:
        st.error(f"모든 정답 텍스트 박스에 값을 입력하세요. 누락: {missing}")
        st.stop()

    # 2-1) 풀이 배치
    solve_tasks, solve_pids = [], []
    for p in range(1, st.session_state.n_problems+1):
        idxs = st.session_state.assignments.get(p, [])
        if not idxs: continue
        images = [(st.session_state.q_items[i]["bytes"], st.session_state.q_items[i]["mime"]) for i in idxs]
        solve_tasks.append(messages_for_images(build_solve_prompt(), images))
        solve_pids.append(p)

    if solve_tasks:
        with st.spinner("문제 풀이 중..."):
            try:
                solve_resps = batch_completion(
                    model=st.session_state.model_solve,
                    messages=solve_tasks,
                    temperature=0.6, top_p=0.95, max_tokens=16384
                )
                for p, resp in zip(solve_pids, solve_resps):
                    st.session_state.solve_outputs[p] = extract_text_from_resp(resp)
            except Exception as e:
                st.error(f"풀이 배치 오류: {e}")

    # 2-2) Judge 배치 텍스트와 이미지 동시 제공
    judge_tasks, judge_pids = [], []
    for p in range(1, st.session_state.n_problems+1):
        if not st.session_state.assignments.get(p): 
            continue
        model_out = st.session_state.solve_outputs.get(p, "")
        golden_text = st.session_state.gt_answers.get(p, "")
        j_items = st.session_state.judge_ans_items.get(p, [])
        j_imgs = [(it["bytes"], it["mime"]) for it in j_items]
        judge_prompt = build_judge_prompt(model_out, golden_text, has_images=bool(j_imgs))
        judge_tasks.append(messages_for_images(judge_prompt, j_imgs))
        judge_pids.append(p)

    if judge_tasks:
        with st.spinner("채점 중..."):
            try:
                judge_resps = batch_completion(
                    model=st.session_state.model_judge,
                    messages=judge_tasks,
                    temperature=0.0, max_tokens=4096
                )
                for p, resp in zip(judge_pids, judge_resps):
                    raw = extract_text_from_resp(resp)
                    st.session_state.judge_raw[p] = raw
                    st.session_state.judge_json[p] = parse_judge_json(raw)
            except Exception as e:
                st.error(f"Judge 배치 오류: {e}")

# 결과 표시
if st.session_state.solve_outputs or st.session_state.judge_json:
    st.markdown("#### 결과 요약")
for p in range(1, st.session_state.n_problems+1):
    if not st.session_state.assignments.get(p): continue
    s_out = st.session_state.solve_outputs.get(p, "")
    j_out = st.session_state.judge_raw.get(p, "")
    if s_out:
        st.text_area(f"[문제 {p}] 모델 풀이 결과", value=s_out, height=260, key=f"solve_out_{p}")
    if j_out:
        st.text_area(f"[문제 {p}] Judge 출력 JSON", value=j_out, height=200, key=f"judge_out_{p}")

# =====================================
# 3) 판별 버튼
# =====================================
if st.session_state.judge_json:
    col_ok, col_no = st.columns(2)
    with col_ok:
        correct_btn = st.button("✅ 정답 이미지 저장 및 초기화", use_container_width=True, type="secondary")
    with col_no:
        wrong_btn = st.button("❌ 오답 이미지 저장 및 OCR 진행", use_container_width=True)

    # 해시 기반 group_id 부여. 동일 입력이면 동일 값
    if (correct_btn or wrong_btn) and not st.session_state.group_id:
        st.session_state.group_id = compute_group_id(st.session_state.q_items, st.session_state.ctx_items)

    if correct_btn:
        for p in range(1, st.session_state.n_problems+1):
            idxs = st.session_state.assignments.get(p, [])
            if not idxs: continue
            save_group_problem_images(st.session_state.q_items, idxs, Q_CORRECT_DIR, st.session_state.group_id, p)
        if st.session_state.context_on and st.session_state.ctx_items:
            save_group_context_images(st.session_state.ctx_items, st.session_state.group_id)
        st.session_state.save_dir_for_current = "correct"
        st.success("정답 처리 완료. 초기화합니다.")
        reset_to_initial(keep_settings=True)

    if wrong_btn:
        # 문제 이미지를 오답 폴더에 저장
        for p in range(1, st.session_state.n_problems+1):
            idxs = st.session_state.assignments.get(p, [])
            if not idxs: continue
            saved = save_group_problem_images(st.session_state.q_items, idxs, Q_INCORRECT_DIR, st.session_state.group_id, p)
            st.session_state.saved_paths[p] = saved

        if st.session_state.context_on and st.session_state.ctx_items:
            st.session_state.ctx_saved_paths = save_group_context_images(st.session_state.ctx_items, st.session_state.group_id)

        # OCR 수행
        ocr_tasks, ocr_pids = [], []
        ocr_prompt = build_ocr_prompt()
        for p in range(1, st.session_state.n_problems+1):
            idxs = st.session_state.assignments.get(p, [])
            if not idxs: continue
            images = [(st.session_state.q_items[i]["bytes"], st.session_state.q_items[i]["mime"]) for i in idxs]
            ocr_tasks.append(messages_for_images(ocr_prompt, images))
            ocr_pids.append(p)

        if ocr_tasks:
            if batch_completion is None:
                st.error("litellm 미설치: pip install litellm")
                st.stop()
            with st.spinner("오답으로 분류됨. OCR 실행 중..."):
                try:
                    ocr_resps = batch_completion(
                        model=st.session_state.model_ocr,
                        messages=ocr_tasks,
                        temperature=0.0, max_tokens=16384
                    )
                    for p, resp in zip(ocr_pids, ocr_resps):
                        st.session_state.question_texts[p] = extract_text_from_resp(resp)
                except Exception as e:
                    st.error(f"OCR 오류: {e}")

        # 컨텍스트 OCR
        if st.session_state.context_on and st.session_state.ctx_items:
            if not st.session_state.ctx_ocr_text:
                ctx_msgs = messages_for_images(ocr_prompt, [(it["bytes"], it["mime"]) for it in st.session_state.ctx_items])
                try:
                    ctx_resp = batch_completion(
                        model=st.session_state.model_ocr,
                        messages=[ctx_msgs],
                        temperature=0.0, max_tokens=16384
                    )[0]
                    st.session_state.ctx_ocr_text = extract_text_from_resp(ctx_resp)
                except Exception as e:
                    st.error(f"컨텍스트 OCR 오류: {e}")

        st.session_state.save_dir_for_current = "incorrect"
        st.success("오답 처리 완료. 아래 4단계에서 검수 후 저장하세요.")

# =====================================
# 4) 문항 텍스트 검수 및 정답 입력
# =====================================
st.markdown("***")
st.markdown("### 4) 문항 텍스트 검수 및 정답 입력")

for p in range(1, st.session_state.n_problems+1):
    if not st.session_state.assignments.get(p):
        continue
    st.markdown(f"#### 문제 {p}")

    # 문항 텍스트
    default_qtext = st.session_state.question_texts.get(p, "")
    st.text_area(f"[문제 {p}] 문항 텍스트 검수", value=default_qtext, height=220, key=f"qtext_{p}")

    # 문항 이미지 매핑
    saved = st.session_state.saved_paths.get(p, [])
    if saved:
        options = saved
        default = saved
        fmt = lambda path: pathlib.Path(path).name
    else:
        idxs = st.session_state.assignments.get(p, [])
        options = idxs
        default = idxs
        fmt = lambda idx: names[idx]
    st.multiselect(
        f"[문제 {p}] 텍스트와 연결할 문제 이미지",
        options=options,
        default=default,
        format_func=fmt,
        key=f"qimgs_{p}"
    )

    # 정답 입력 UI 좌우 배치
    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        prefill_ans = st.session_state.gt_answers.get(p, "")
        st.text_area(f"[문제 {p}] 정답 텍스트", value=prefill_ans, height=160, key=f"ans_text_{p}")
    with c2:
        # Judge 단계에서 올린 이미지 포함 여부
        has_judge_imgs = bool(st.session_state.judge_ans_items.get(p))
        include_default = True
        st.checkbox(
            f"[문제 {p}] Judge 단계의 정답 이미지 포함",
            value=include_default if has_judge_imgs else False,
            key=f"include_jimg_{p}",
            disabled=not has_judge_imgs
        )
        names_j = [it["name"] for it in st.session_state.judge_ans_items.get(p, [])]
        if names_j:
            st.caption("포함 예정: " + ", ".join(names_j))
        # 추가 업로드
        st.file_uploader(
            f"[문제 {p}] 추가 정답 이미지 업로드",
            type=["png","jpg","jpeg","webp"],
            accept_multiple_files=True,
            key=f"ans_imgs_{p}"
        )

# 컨텍스트 텍스트
if st.session_state.context_on:
    st.markdown("#### 컨텍스트 문단")
    st.text_area("컨텍스트 텍스트 검수", value=st.session_state.ctx_ocr_text, height=220, key="ctx_text_review")

# =====================================
# 5) 최종 저장
# =====================================
st.markdown("***")
st.markdown("### 5) 최종 저장")

save_btn = st.button("💾 JSON 저장", type="primary", use_container_width=True)
if save_btn:
    if not st.session_state.group_id:
        st.session_state.group_id = compute_group_id(st.session_state.q_items, st.session_state.ctx_items)

    # 한 문단 기준으로 하나의 JSON 파일
    if st.session_state.context_on:
        if not st.session_state.ctx_saved_paths and st.session_state.ctx_items:
            st.session_state.ctx_saved_paths = save_group_context_images(st.session_state.ctx_items, st.session_state.group_id)
        context_block = {
            "text": st.session_state.get("ctx_text_review", st.session_state.ctx_ocr_text),
            "image": st.session_state.ctx_saved_paths or []
        }
    else:
        context_block = ""

    year_val = st.session_state.year_text.strip()

    questions_payload = []
    for p in range(1, st.session_state.n_problems+1):
        if not st.session_state.assignments.get(p):
            continue

        # 문항 텍스트
        q_text = st.session_state.get(f"qtext_{p}", "").strip()

        # 문제 이미지 경로
        selected = st.session_state.get(f"qimgs_{p}", []) or []
        if selected and isinstance(selected[0], str):
            q_imgs_paths = selected
        elif selected:
            target_dir = Q_CORRECT_DIR if st.session_state.get("save_dir_for_current") != "incorrect" else Q_INCORRECT_DIR
            idxs = [int(x) for x in selected]
            q_imgs_paths = save_group_problem_images(st.session_state.q_items, idxs, target_dir, st.session_state.group_id, p)
        else:
            q_imgs_paths = []

        # 정답 텍스트
        a_text = st.session_state.get(f"ans_text_{p}", "").strip()

        # 정답 이미지 경로
        a_imgs_paths = []

        # 1) Judge 단계에서 올린 이미지 포함
        if st.session_state.get(f"include_jimg_{p}", False):
            a_imgs_paths += save_answer_items_from_bytes(
                st.session_state.judge_ans_items.get(p, []),
                st.session_state.group_id,
                p,
                tag="ansj"
            )

        # 2) 4단계에서 추가 업로드한 이미지 저장
        ans_files = st.session_state.get(f"ans_imgs_{p}", [])
        a_imgs_paths += save_group_answer_images(ans_files, st.session_state.group_id, p)

        questions_payload.append({
            "text": q_text,
            "text_image": q_imgs_paths,
            "answer": a_text,
            "answer_image": a_imgs_paths
        })

    group_obj = [{
        "context": context_block,
        "question": questions_payload,
        "year": year_val
    }]

    out_path = RECORDS_DIR / f"bundle_{st.session_state.group_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(group_obj, f, ensure_ascii=False, indent=2)

    st.success(f"저장 완료: {out_path}")
    reset_to_initial(keep_settings=True)
