import json
import math
import re
import subprocess
import sys
import textwrap
from collections import Counter
from typing import Optional

_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_LAST_NUM_RE = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)")

# Qwen3 and similar models emit chain-of-thought inside <think>...</think> tags.
# These tokens are reasoning artifacts and must be stripped before any
# text-based evaluation of model outputs.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# ---- AG_NEWS ----------------------------------------------------------------
# Canonical lowercase names (match HuggingFace ClassLabel order 0-3).
_AG_NEWS_LABELS_ORDERED = ["world", "sports", "business", "sci/tech"]
# Map all reasonable synonyms / substrings → canonical form.
# Sorted longest-first so "sci/tech" is tried before "sci" or "tech".
_AG_NEWS_SYNONYMS = [
    ("sci/tech",     "sci/tech"),
    ("science/tech", "sci/tech"),
    ("sci / tech",   "sci/tech"),
    ("science and technology", "sci/tech"),
    ("science & technology",   "sci/tech"),
    ("technology",   "sci/tech"),
    ("science",      "sci/tech"),
    ("sci",          "sci/tech"),
    ("tech",         "sci/tech"),
    ("business",     "business"),
    ("sports",       "sports"),
    ("sport",        "sports"),
    ("world",        "world"),
    ("politics",     "world"),    # sometimes models say "Politics" meaning World news
    ("international","world"),
]

# ---- CommonsenseQA / ARC-Challenge / MedMCQA / SciQ -----------------------
# Match a stand-alone A-D or A-E letter (not adjacent to another letter).
_MCQA_LETTER_RE = re.compile(r'(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])')
_MCQA4_LETTER_RE = re.compile(r'(?<![A-Za-z])([A-Da-d])(?![A-Za-z])')
# BBH MCQ: targets like (A), (B), (C), (D), (E) — letter wrapped in parens.
_BBH_PAREN_LETTER_RE = re.compile(r'\(([A-Ea-e])\)')

# ---- MATH (EleutherAI/hendrycks_math) ---------------------------------------
# Extract the final \boxed{...} from model output or solution text.
_BOXED_RE = re.compile(r'\\boxed\{([^{}]*)\}')
# Nested boxed: also try \boxed{\frac{...}{...}} etc.
_BOXED_NESTED_RE = re.compile(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}')

# ---- TriviaQA ---------------------------------------------------------------
# Acceptable answers are stored joined by this separator.
_TRIVIA_ALIAS_SEP = "|||"
# Standard TriviaQA/SQuAD normalization regexes.
_TRIVIA_ARTICLES_RE = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_TRIVIA_PUNCT_RE    = re.compile(r'[^\w\s]')   # remove non-word, non-space chars
_TRIVIA_PREFIX_RE   = re.compile(r'^#+\s*')    # strip leading #### wrapper

# ---- Code datasets (MBPP, HumanEval) ----------------------------------------
# Regex to extract fenced code blocks (```python ... ``` or plain ``` ... ```).
_CODE_FENCE_RE  = re.compile(
    r'```(?:python)?\s*\n(.*?)```', re.DOTALL | re.IGNORECASE
)
# Execution timeout in seconds for code verification subprocess.
_CODE_EXEC_TIMEOUT = 10


def _extract_code_block(text: str) -> str:
    """Extract Python code from model output.

    Tries (in order):
      1. Content of the first ```python``` / ``` fenced block.
      2. Any sequence of lines that look like Python (start with 'def ', 'class '
         or are indented).
      3. The raw text as fallback.
    Returns the extracted code string (may be empty).
    """
    # Remove <think> blocks but do NOT call _strip_thinking() — that calls
    # .strip() which would destroy leading indentation on function bodies.
    if text is None:
        return ""
    text = _THINK_RE.sub("", text)
    # 1. Fenced block
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # 2. Heuristic: find first 'def ' or 'class ' line and take from there.
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r'\s*(def |class |import |from )', line):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip()
    # 3. Fallback: return the text with only trailing whitespace stripped.
    # Do NOT call strip() — that would destroy the shared leading indent that
    # _normalize_code() uses via textwrap.dedent().
    return text.rstrip()


def _normalize_code(code: str) -> str:
    """Normalize Python code for comparison / conformal fact clustering.

    - Strip comments and docstrings
    - Dedent
    - Collapse runs of blank lines to a single blank line
    - Lower-case keywords are NOT changed (Python is case-sensitive)
    Returns a canonical string up to 400 chars.
    """
    # Remove single-line comments
    code = re.sub(r'#[^\n]*', '', code)
    # Remove docstrings (triple-quoted)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    try:
        code = textwrap.dedent(code)
    except Exception:
        pass
    # Collapse whitespace lines
    lines = [ln.rstrip() for ln in code.splitlines() if ln.strip()]
    normalized = "\n".join(lines)
    return normalized[:400]


def _run_code_with_tests(
    code: str,
    tests: list,
    setup: str = "",
    timeout: int = _CODE_EXEC_TIMEOUT,
) -> bool:
    """Execute code + optional setup + assertion tests in a subprocess.

    Returns True if all assertions pass within the timeout, False otherwise.
    Executes in an isolated subprocess so that crashes / infinite loops do not
    affect the training process.
    """
    # Build the script to run
    test_block = "\n".join(str(t) for t in tests)
    script = f"{setup}\n{code}\n{test_block}\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _run_humaneval_code(
    prompt: str,
    completion: str,
    test_fn: str,
    entry_point: str,
    timeout: int = _CODE_EXEC_TIMEOUT,
) -> bool:
    """Execute a HumanEval completion against its check() function.

    The script is structured as:
        <prompt>           # includes the function signature
        <completion>       # the function body generated by the model
        <test_fn>          # defines check(candidate)
        check(<entry_point>)
    """
    script = f"{prompt}{completion}\n{test_fn}\ncheck({entry_point})\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _normalize_trivia(text: str) -> str:
    """Normalize a TriviaQA answer for comparison.

    Follows the standard TriviaQA/SQuAD evaluation pipeline:
      strip <think> blocks → strip #### prefix → lowercase
      → remove punctuation → remove articles → collapse whitespace.
    """
    text = _strip_thinking(text)
    text = _TRIVIA_PREFIX_RE.sub("", text.strip())
    text = text.lower()
    text = _TRIVIA_PUNCT_RE.sub(" ", text)
    text = _TRIVIA_ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Qwen3-series models (base and instruct) may spontaneously emit chain-of-
    thought reasoning wrapped in <think>...</think> tags.  Including these
    tokens in precision/F1 calculations severely dilutes scores because the
    reasoning text is NOT the target Wikipedia passage.  For example, a
    256-token completion that is 220 think-tokens + 36 answer-tokens will have
    token-precision ≈ 0.05 even when the 36-token answer is perfectly on-topic.
    """
    if text is None:
        return ""
    return _THINK_RE.sub("", text).strip()


def _canonicalize(s: str) -> str:
    """
    Normalize a numeric string to canonical form so that
    '42', '42.0', '42.00', '42,000' all produce consistent strings.
    Integers are returned without decimal point ('42' not '42.0').
    """
    s = s.replace(",", "").strip()
    if s == "":
        return ""
    try:
        val = float(s)
        if val == int(val) and not ("e" in s.lower() or "E" in s):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


def extract_final_answer(text: str) -> str:
    """
    Robust extraction:
      1) Prefer '#### <number>'
      2) Otherwise use the last number appearing in the text
    Returns CANONICALIZED numeric string so that '42.0' and '42'
    both become '42'. This is critical for conformal prediction
    where answers are compared by string equality.
    """
    if text is None or text == "":
        return ""
    
    # First try to find #### pattern
    m = _FINAL_RE.search(text)
    if m:
        return _canonicalize(m.group(1))

    # fallback: last number anywhere in the text
    nums = _LAST_NUM_RE.findall(text)
    if nums:
        return _canonicalize(nums[-1])
    return ""

def _normalize_num(s: str) -> Optional[float]:
    try:
        # allow commas like "1,234"
        s = s.replace(",", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

def verify_gsm8k(pred_text: str, gold_text: str) -> int:
    """
    Returns 1 if numeric answer matches, else 0.
    Works even if pred doesn't contain ####.
    """
    pred = extract_final_answer(pred_text)
    gold = extract_final_answer(gold_text)

    p = _normalize_num(pred)
    g = _normalize_num(gold)
    if p is None or g is None:
        return 0

    # GSM8K answers are typically integers; but be tolerant:
    return int(abs(p - g) < 1e-6)


def _extract_boxed(text: str) -> str:
    """Extract \\boxed{} content from model output (MATH dataset).

    Tries nested boxed first, then simple, then falls back to the last
    line of the text.
    """
    text = _strip_thinking(text)
    # Try nested/multi-char boxed first
    matches = list(_BOXED_NESTED_RE.finditer(text))
    if not matches:
        matches = list(_BOXED_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()  # last boxed in text
    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""


def _normalize_math(text: str) -> str:
    """Light normalization for MATH answers: strip spaces, remove trailing zeros.

    Handles both numeric (e.g. '7', '3.14', '\\frac{1}{2}') and
    symbolic (e.g. '2\\sqrt{3}', '\\frac{a+b}{c}') answers.
    We do NOT attempt to evaluate/simplify — just normalize whitespace and
    canonical form for exact-match comparison.
    """
    s = text.strip()
    # Remove surrounding $...$ or \(...\) math delimiters
    s = re.sub(r'^\$(.*)\$$', r'\1', s)
    s = re.sub(r'^\\\((.*)\\\)$', r'\1', s)
    # Collapse whitespace inside
    s = ' '.join(s.split())
    # Numeric: try to canonicalize if it's a pure number
    try:
        val = float(s.replace(',', ''))
        if math.isfinite(val) and val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError, OverflowError):
        pass
    return s.lower()


def canonicalize_answer(text: str, dataset_name: str) -> str:
    """Return a normalized answer string for use in frequency counts.

    - Math datasets (gsm8k, svamp, …): numeric extraction via extract_final_answer.
    - AQuaMUSE: strip <think> blocks, keep first 12 words (clusters similar answers).
    - AG_NEWS: map model output to one of {world, sports, business, sci/tech}.
    - CommonsenseQA: extract the first standalone A-E letter.
    - Others: strip + lowercase.
    """
    if text is None:
        return ""
    # MATH-500: \boxed{} style math, must be checked before generic "math" key
    if "math-500" in dataset_name.lower() or "math500" in dataset_name.lower():
        # extract_example returns the already-extracted answer (no \boxed{} wrapper),
        # so _extract_boxed would return "" on gold strings and make cal_truth all-empty.
        # Fall back to normalising the text directly when no \boxed{} is present.
        boxed = _extract_boxed(text)
        return _normalize_math(boxed if boxed else text)
    math_keys = ["gsm8k", "svamp", "math", "aqua-rat"]
    if any(k in dataset_name.lower() for k in math_keys):
        return extract_final_answer(text)
    if "aquamuse" in dataset_name.lower():
        # Strip think-blocks, then use first 12 words so that completions
        # that start the same way cluster together in the conformal set.
        clean = _strip_thinking(text).strip().lower()
        words = clean.split()
        return " ".join(words[:12])
    if "ag_news" in dataset_name.lower():
        low = _strip_thinking(text).strip().lower()
        for synonym, canon in _AG_NEWS_SYNONYMS:
            if synonym in low:
                return canon
        # Fallback: return first 20 chars (won't match any canonical → treated as wrong)
        return low[:20]
    if "commonsense_qa" in dataset_name.lower():
        clean = _strip_thinking(text).strip()
        m = _MCQA_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "ai2_arc" in dataset_name.lower():
        # Same extraction logic as commonsense_qa: first standalone A-E letter.
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        m = _MCQA_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "medmcqa" in dataset_name.lower():
        # A-D multiple choice
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        m = _MCQA4_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "sciq" in dataset_name.lower():
        # A-D multiple choice (correct slot determined by MD5 hash in extract_example)
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        m = _MCQA4_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "aqua_rat" in dataset_name.lower():
        # A-E multiple choice (algebraic MCQ)
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        m = _MCQA_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "gpqa" in dataset_name.lower():
        # A-D multiple choice (graduate-level science MCQ).
        # Model is instructed to output \boxed{X} or just a letter.
        # Try \boxed{X} first, then fall back to any standalone A-D letter.
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        boxed = re.search(r'\\boxed\{([A-Da-d])\}', clean)
        if boxed:
            return boxed.group(1).upper()
        m = _MCQA4_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if any(k in dataset_name.lower() for k in ["race", "reclor"]):
        # A-D multiple choice (reading comprehension / logical reasoning)
        clean = re.sub(r'^#+\s*', '', _strip_thinking(text).strip())
        m = _MCQA4_LETTER_RE.search(clean)
        return m.group(1).upper() if m else clean[:1].upper()
    if "lukaemon/bbh" in dataset_name.lower() or \
            ("bbh" in dataset_name.lower() and "lukaemon" in dataset_name.lower()):
        # BBH gold is either '(X)', 'True'/'False', 'Yes'/'No', 'valid'/'invalid',
        # or an integer.  Normalize: strip parens from MCQ answers, lowercase rest.
        clean = _strip_thinking(text).strip()
        # 1. MCQ: '(A)' style — extract just the letter
        m_paren = _BBH_PAREN_LETTER_RE.search(clean)
        if m_paren:
            return m_paren.group(1).upper()
        # 2. Known short binary/boolean answers — check first word match
        _low = clean.lower()
        _first_word = _low.split()[0] if _low.split() else _low
        for _kw in ("invalid", "valid", "true", "false", "yes", "no"):
            if _low == _kw or _first_word == _kw:
                return _kw
        # 3. Numeric answer — extract last integer in the text
        _nums = re.findall(r'-?\d+', clean)
        if _nums:
            return _nums[-1]
        # 4. Fallback: lowercase last line (model may put the answer at the end)
        last_line = clean.strip().split("\n")[-1].strip().lower()
        return last_line if last_line else _low[:30]
    if "hendrycks_math" in dataset_name.lower():
        # Model outputs have \boxed{answer}; gold (from extract_example) is the
        # already-extracted content (no wrapper).  Fall back to direct norm when
        # no \boxed{} is found so cal_truth entries are never the empty string.
        boxed = _extract_boxed(text)
        return _normalize_math(boxed if boxed else text)
    if "trivia_qa" in dataset_name.lower():
        # Gold stored as "alias1|||alias2|||"; return first alias as canonical.
        # Prediction: just normalize the text.
        if _TRIVIA_ALIAS_SEP in text:
            first = text.split(_TRIVIA_ALIAS_SEP)[0]
            return _normalize_trivia(first)
        return _normalize_trivia(text)
    # ---- MBPP ---------------------------------------------------------------
    # Gold is a JSON blob {ref_code, tests, setup}.  Model output is raw code.
    # This function is called on BOTH the gold (during calibration to produce
    # cal_truth) and on model completions (during rollout).
    # • Gold JSON  → extract the reference code and normalize it.
    # • Model text → extract code block and normalize it.
    # Two completions that produce the same normalized code land in the same
    # conformal set; the reward signal comes from execution in verify_answer.
    if "mbpp" in dataset_name.lower():
        # Use lstrip() only for the JSON probe; pass raw `text` to
        # _extract_code_block so leading indentation is preserved for dedent.
        if text.lstrip().startswith("{"):
            try:
                d = json.loads(text.strip())
                return _normalize_code(d.get("ref_code", ""))
            except (json.JSONDecodeError, KeyError):
                pass
        return _normalize_code(_extract_code_block(text))
    # ---- HumanEval ----------------------------------------------------------
    # Gold is a JSON blob {prompt, ref_solution, test_fn, entry_point}.
    # Model completion is the function body that continues the prompt.
    if "humaneval" in dataset_name.lower():
        if text.lstrip().startswith("{"):
            try:
                d = json.loads(text.strip())
                return _normalize_code(d.get("ref_solution", ""))
            except (json.JSONDecodeError, KeyError):
                pass
        return _normalize_code(_extract_code_block(text))
    return text.strip().lower()


def _token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 between two strings."""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _token_precision(pred: str, gold: str) -> float:
    """Precision: fraction of prediction tokens that appear in gold.

    Preferred over F1 when comparing a short factual answer against a long
    gold passage (e.g. AQuaMUSE targets are 100-300 word Wikipedia excerpts).
    In that setting recall = overlap/|gold| is structurally ~0.05 even for a
    perfect concise answer, collapsing F1 to ~0.1 regardless of quality.
    Precision = overlap/|pred| stays high (~0.6-0.8) for correct answers.

    <think>...</think> blocks are stripped before evaluation because Qwen3-series
    models emit reasoning tokens that are not part of the answer and which would
    otherwise dilute precision to near-zero.
    """
    pred = _strip_thinking(pred)
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    return sum(common.values()) / len(pred_tokens)


def verify_answer(pred_text: str, gold_text: str, dataset_name: str) -> int:
    """General verification for different dataset types.

    - Math datasets (gsm8k, svamp, math, aqua-rat): numeric extraction / comparison.
    - AQuaMUSE: token precision ≥ 0.20 (after stripping <think> blocks).
    - AG_NEWS: canonical label match (both sides normalised through canonicalize_answer).
    - CommonsenseQA: answer letter (A-E) must match the gold answerKey.
    - Others: exact string match after stripping + lowercasing.
    """
    # MATH-500: extract \\boxed{} from model output, compare to pre-extracted gold.
    # Must be checked BEFORE the generic "math" key.
    if "math-500" in dataset_name.lower() or "math500" in dataset_name.lower():
        pred_norm = _normalize_math(_extract_boxed(pred_text))
        gold_norm = _normalize_math(gold_text)   # gold is already extracted (no boxed)
        return int(pred_norm == gold_norm)

    math_keys = ["gsm8k", "svamp", "math", "aqua-rat"]
    if any(k in dataset_name.lower() for k in math_keys):
        return verify_gsm8k(pred_text, gold_text)

    if "aquamuse" in dataset_name.lower():
        # Strip think-blocks FIRST: Qwen3-32B base emits <think>...</think>
        # reasoning chains that can be 200+ tokens long; including them
        # dilutes precision from ~0.5 (good answer) to ~0.05 (spurious).
        clean = _strip_thinking(pred_text)
        if not clean.strip():
            clean = pred_text  # fallback: model output was all inside <think>
        pred_tokens = clean.strip().lower().split()
        if len(pred_tokens) < 3:
            return 0  # too short / likely degenerate
        # Threshold 0.20: distinguishes on-topic text (prec ≈ 0.25-0.80) from
        # off-topic garbage (prec ≈ 0.05-0.12).
        return int(_token_precision(clean, gold_text) >= 0.20)

    if "ag_news" in dataset_name.lower():
        # Gold is already a canonical name ("world", "sports", "business", "sci/tech").
        # Canonicalise prediction and compare.
        pred_canon = canonicalize_answer(pred_text, dataset_name)
        gold_canon = canonicalize_answer(gold_text, dataset_name)
        return int(pred_canon == gold_canon)

    if "commonsense_qa" in dataset_name.lower():
        # Gold is a single uppercase letter A-E (answerKey).
        # Use canonicalize_answer on both sides so that gold wrapped with
        # "#### " (as done by eval_pareto.py) is correctly extracted.
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "ai2_arc" in dataset_name.lower():
        # Same letter-matching logic as commonsense_qa.
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "medmcqa" in dataset_name.lower():
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "sciq" in dataset_name.lower():
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "aqua_rat" in dataset_name.lower():
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "gpqa" in dataset_name.lower():
        # Gold is a single letter A-D; pred may contain \boxed{X} or plain letter.
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if any(k in dataset_name.lower() for k in ["race", "reclor"]):
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    if "hendrycks_math" in dataset_name.lower():
        # Compare normalized \boxed{} extractions
        pred_norm = _normalize_math(_extract_boxed(pred_text))
        gold_norm = _normalize_math(_extract_boxed(gold_text) if _BOXED_RE.search(gold_text)
                                    else gold_text)
        return int(pred_norm == gold_norm) if pred_norm else 0

    # ---- BBH (lukaemon/bbh) -------------------------------------------------
    # Answers come in three flavours:
    # 1. MCQ letters: gold is '(A)', '(B)', ... '(E)'  — extract the letter
    # 2. Binary text: gold is 'True'/'False', 'Yes'/'No', 'valid'/'invalid'
    # 3. Numeric:     gold is an integer string like '24'
    # Unify by lowercasing+stripping both sides; for MCQ strip the parens.
    if "lukaemon/bbh" in dataset_name.lower() or \
            ("bbh" in dataset_name.lower() and "lukaemon" in dataset_name.lower()):
        gold_letter = canonicalize_answer(gold_text, dataset_name)
        pred_letter = canonicalize_answer(pred_text, dataset_name)
        return int(pred_letter == gold_letter)

    # ---- MBPP ---------------------------------------------------------------
    # gold_text is a JSON string: {"ref_code": ..., "tests": [...], "setup": ...}
    # pred_text is raw model output containing Python code.
    if "mbpp" in dataset_name.lower():
        try:
            d = json.loads(gold_text)
        except (json.JSONDecodeError, TypeError):
            return 0
        tests = d.get("tests", [])
        setup = d.get("setup", "")
        if not tests:
            return 0
        code = _extract_code_block(_strip_thinking(pred_text))
        if not code.strip():
            return 0
        return int(_run_code_with_tests(code, tests, setup))

    # ---- HumanEval ----------------------------------------------------------
    # gold_text is a JSON string: {"prompt": ..., "ref_solution": ...,
    #                               "test_fn": ..., "entry_point": ...}
    # pred_text is the model's function body completion.
    if "humaneval" in dataset_name.lower():
        try:
            d = json.loads(gold_text)
        except (json.JSONDecodeError, TypeError):
            return 0
        prompt      = d.get("prompt", "")
        test_fn     = d.get("test_fn", "")
        entry_point = d.get("entry_point", "")
        if not test_fn or not entry_point:
            return 0
        # Remove <think> blocks but preserve leading indentation (rstrip only).
        # _strip_thinking() calls .strip() which would destroy function-body indent.
        completion = _THINK_RE.sub("", pred_text).rstrip()
        # Only unwrap fenced blocks; preserve raw indentation otherwise.
        m = _CODE_FENCE_RE.search(completion)
        code_block = m.group(1) if m else completion
        return int(_run_humaneval_code(prompt, code_block, test_fn, entry_point))

    # Default: exact match after stripping + lowercasing
    p = pred_text.strip().lower()
    g = gold_text.strip().lower()
    return int(p == g)
