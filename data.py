import json
import os
import re
import numpy as np
from datasets import load_dataset

# AG_NEWS label integers → canonical lowercase names
# (matches ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech']))
_AG_NEWS_INT_TO_NAME = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}

# Datasets whose labelled split is "validation" rather than "test"
# (commonsense_qa test has empty answerKey; trivia_qa test answers are <unk>)
_VALIDATION_ONLY_DATASETS = ["commonsense_qa", "trivia_qa"]

# Datasets that only ship a "test" split (no "train"); we synthesise a train
# split by sampling from the test split and holding out the rest for eval.
_TEST_ONLY_DATASETS = ["openai_humaneval", "gpqa", "lukaemon/bbh"]

# BBH: 6-task mix used for C-GRPO experiments (all have clear verifiable answers)
_BBH_MIX6_TASKS = [
    "logical_deduction_five_objects",  # multi-step logic, (A)-(E)
    "date_understanding",              # temporal reasoning, (A)-(E)
    "boolean_expressions",             # symbolic eval, True/False
    "causal_judgement",                # causal inference, Yes/No
    "multistep_arithmetic_two",        # arithmetic, integer
    "tracking_shuffled_objects_three_objects",  # object tracking, (A)-(C)
]

# MedMCQA: cop field is 1-indexed integer → letter mapping
_MEDMCQA_INT_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}
# MATH (hendrycks): we filter to hard levels 3-5 only
_MATH_HARD_LEVELS = {"Level 3", "Level 4", "Level 5"}

# ARC-Challenge sometimes uses numeric choice labels 1-4 instead of A-D.
# Map them to letters so the model always sees "(A) text ...".
_ARC_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def format_prompt(question: str, dataset_name: str = "gsm8k") -> str:
    """Return a prompt string suitable for the given dataset.

    For SVAMP/GSM8K `question` is already the combined story+question.
    For ag_news `question` is the raw news text.
    For commonsense_qa `question` is pre-formatted with choices embedded
    (see extract_example).
    """
    # MATH-500 (HuggingFaceH4/MATH-500): uses \boxed{} format like hendrycks_math
    # Must be checked BEFORE the generic "math" key check below.
    if "math-500" in dataset_name.lower() or "math500" in dataset_name.lower():
        return (
            "Solve the following math problem.\n"
            "Show your reasoning step by step. At the end, write the final answer inside \\\\boxed{}.\n\n"
            f"Problem: {question}\n"
            "Solution:"
        )

    # Math: GSM8K, SVAMP, AQuA-RAT, MATH
    if any(k in dataset_name.lower() for k in ["gsm8k", "svamp", "math", "aqua-rat"]):
        return (
            "Solve the following math word problem.\n"
            "Show your reasoning step by step, then write the final numeric answer after '####'.\n\n"
            f"Problem: {question}\n"
            "Answer:\n"
        )
    # Abstractive QA (AQuaMUSE)
    if "aquamuse" in dataset_name.lower():
        return (
            "Answer the following question concisely and accurately.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    # AG_NEWS: 4-class news topic classification
    if "ag_news" in dataset_name.lower():
        return (
            "Classify the following news text into exactly one of these four categories:\n"
            "World, Sports, Business, Sci/Tech\n\n"
            "Respond with only the category name and nothing else.\n\n"
            f"Text: {question}\n"
            "Category:"
        )
    # ARC-Challenge: 4-choice science MCQ (choices already embedded by extract_example)
    if "ai2_arc" in dataset_name.lower():
        return (
            "Answer the following multiple-choice science question.\n"
            "Respond with only the letter of the correct answer (A, B, C, D, or E).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # MedMCQA: 4-choice medical MCQ (choices already embedded by extract_example)
    if "medmcqa" in dataset_name.lower():
        return (
            "Answer the following multiple-choice medical question.\n"
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # SciQ: 4-choice science MCQ (choices already embedded by extract_example)
    if "sciq" in dataset_name.lower():
        return (
            "Answer the following multiple-choice science question.\n"
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # MATH (hendrycks_math): free-form math, answer inside \\boxed{} in solution
    if "hendrycks_math" in dataset_name.lower():
        return (
            "Solve the following math problem.\n"
            "Show your reasoning step by step. At the end, write the final answer inside \\\\boxed{}.\n\n"
            f"Problem: {question}\n"
            "Solution:"
        )
    # CommonsenseQA: multiple-choice A-E (choices already embedded by extract_example)
    if "commonsense_qa" in dataset_name.lower():
        return (
            "Answer the following multiple-choice question.\n"
            "Respond with only the letter of the correct answer (A, B, C, D, or E).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # AQuA-RAT: 5-choice algebraic MCQ (choices embedded by extract_example)
    if "aqua_rat" in dataset_name.lower():
        return (
            "Solve the following problem. Choose the correct answer.\n"
            "Respond with only the letter of the correct answer (A, B, C, D, or E).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # RACE: 4-choice reading comprehension (passage + question + choices embedded by extract_example)
    if "race" in dataset_name.lower():
        return (
            "Read the passage and answer the question.\n"
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # GPQA (Graduate-Level Google-Proof Q&A): 4-choice science MCQ
    # The question text already contains (A)-(D) choices (embedded by extract_example).
    if "gpqa" in dataset_name.lower():
        return (
            "Answer the following graduate-level science question.\n"
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # ReClor: 4-choice logical reasoning MCQ (context + question + choices embedded by extract_example)
    if "reclor" in dataset_name.lower():
        return (
            "Read the passage and answer the logical reasoning question.\n"
            "Respond with only the letter of the correct answer (A, B, C, or D).\n\n"
            f"{question}\n"
            "Answer:"
        )
    # TriviaQA: open-ended factual QA
    if "trivia_qa" in dataset_name.lower():
        return (
            "Answer the following question with a short factual answer.\n"
            "Give only the answer itself — no explanation, no sentence.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    # MBPP: complete a Python function given a natural-language description.
    # The question text already contains the required signature and example tests
    # (embedded by extract_example) so the model knows the exact function name.
    if "mbpp" in dataset_name.lower():
        return (
            "Write a Python function that solves the following problem.\n"
            "Return ONLY the complete Python function definition, no explanation,\n"
            "no markdown, no extra text.  Use EXACTLY the function name shown below.\n\n"
            f"{question}\n"
            "\nSolution:\n"
        )
    # HumanEval: the prompt already contains the function signature + docstring;
    # ask the model to complete it.
    if "humaneval" in dataset_name.lower():
        return (
            "Complete the following Python function. "
            "Return ONLY the function body (the indented code that goes after the "
            "existing definition line), no explanation, no markdown.\n\n"
            f"{question}"
        )
    # BBH (lukaemon/bbh): the input field already contains the full question
    # (with embedded choices for MCQ tasks).  We add a chain-of-thought
    # instruction and ask for the final answer on the last line.
    if "lukaemon/bbh" in dataset_name.lower() or ("bbh" in dataset_name.lower() and "lukaemon" in dataset_name.lower()):
        return (
            "Think step by step, then give the final answer on the last line.\n"
            "For multiple-choice questions respond with only the letter (e.g. (A)).\n"
            "For True/False or Yes/No questions respond with only that word.\n"
            "For numeric answers respond with only the number.\n\n"
            f"{question}\n"
            "Answer:"
        )
    # Other classification (SST2, PIQA, etc.)
    if any(k in dataset_name.lower() for k in ["sst2", "piqa"]):
        return f"Text: {question}\nLabel:"
    # Generic fallback
    return f"Question: {question}\nAnswer:"


def extract_example(example: dict, dataset_name: str):
    """Return (question_text, gold_answer_str) for a single dataset example.

    Handles the specific column layouts of every supported dataset:

    - GSM8K:        question / answer
    - SVAMP:        question_concat / Answer  (str, already a number)
    - AQuaMUSE:     query / target
    - AG_NEWS:      text / label (int → label name)
    - CommonsenseQA:question + choices dict / answerKey  (letter A-E)
    """
    # ---- AQuaMUSE ----
    if "aquamuse" in dataset_name.lower():
        return example.get("query", ""), example.get("target", "")

    # ---- SVAMP ----
    # Columns: ID, Body, Question, Equation, Answer (str), Type, question_concat
    # question_concat is the pre-joined "Body + Question" string.
    if "svamp" in dataset_name.lower():
        text = example.get("question_concat", "")
        if not text:
            # Fallback: join manually
            text = (example.get("Body", "") + " " + example.get("Question", "")).strip()
        answer_raw = example.get("Answer", "0")
        # Answer is already a string like '145' or '2.5'; normalise to int-string when possible
        try:
            val = float(str(answer_raw))
            answer_str = str(int(val)) if val == int(val) else str(val)
        except (ValueError, TypeError):
            answer_str = str(answer_raw)
        return text, answer_str

    # ---- AG_NEWS ----
    # Columns: text (str), label (ClassLabel int: 0=World, 1=Sports, 2=Business, 3=Sci/Tech)
    if "ag_news" in dataset_name.lower():
        text = example.get("text", "")
        label_int = int(example.get("label", 0))
        label_name = _AG_NEWS_INT_TO_NAME.get(label_int, str(label_int))
        return text, label_name   # gold is e.g. "world", "sports", "business", "sci/tech"

    # ---- ARC-Challenge ----
    # Columns: id, question (str), choices {'text':[...], 'label':[...]}, answerKey (str A-E or 1-4)
    # Labels can be '1','2','3','4' in some items — normalize to letters.
    if "ai2_arc" in dataset_name.lower():
        question_text = example.get("question", "")
        choices = example.get("choices", {})
        labels = choices.get("label", [])
        texts  = choices.get("text",  [])
        labels_norm = [_ARC_NUM_TO_LETTER.get(str(l), str(l).upper()) for l in labels]
        choices_str = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels_norm, texts))
        full_q = f"Question: {question_text}\n{choices_str}"
        answer_key = str(example.get("answerKey", "")).strip()
        answer_key = _ARC_NUM_TO_LETTER.get(answer_key, answer_key.upper())
        return full_q, answer_key   # gold is a single letter A-E

    # ---- TriviaQA ----
    # Columns: question (str), answer {'normalized_aliases': [...], 'normalized_value': str, ...}
    # The test split has empty/unknown answers; use validation instead (see _VALIDATION_ONLY_DATASETS).
    if "trivia_qa" in dataset_name.lower():
        question_text = example.get("question", "")
        answer = example.get("answer", {})
        normalized_aliases = answer.get("normalized_aliases", [])
        if not normalized_aliases:
            normalized_aliases = [answer.get("normalized_value", answer.get("value", ""))]
        # Join all acceptable answer forms with a separator so verify_answer can check all of them.
        gold_str = "|||".join(a for a in normalized_aliases if a and a != "<unk>")
        return question_text, gold_str

    # ---- MedMCQA ----
    # Columns: id, question, opa, opb, opc, opd, cop (1-indexed int), choice_type, exp, subject_name
    if "medmcqa" in dataset_name.lower():
        question_text = example.get("question", "")
        opa = example.get("opa", "")
        opb = example.get("opb", "")
        opc = example.get("opc", "")
        opd = example.get("opd", "")
        choices_str = f"(A) {opa}\n(B) {opb}\n(C) {opc}\n(D) {opd}"
        full_q = f"Question: {question_text}\n{choices_str}"
        cop_int = int(example.get("cop", 1))
        answer_letter = _MEDMCQA_INT_TO_LETTER.get(cop_int, "A")
        return full_q, answer_letter

    # ---- SciQ ----
    # Columns: question, correct_answer, distractor1, distractor2, distractor3, support
    # We construct A-D choices where the correct answer is always placed at a fixed position
    # determined by a hash of the question (so it's reproducible but not always A).
    if "sciq" in dataset_name.lower():
        import hashlib
        question_text = example.get("question", "")
        correct = example.get("correct_answer", "")
        d1 = example.get("distractor1", "")
        d2 = example.get("distractor2", "")
        d3 = example.get("distractor3", "")
        # Shuffle choices using a hash for reproducibility (prevents always-A bias).
        h = int(hashlib.md5(question_text.encode()).hexdigest(), 16)
        distractors = [d1, d2, d3]
        correct_slot = h % 4   # 0-3
        choices = []
        d_idx = 0
        for i in range(4):
            if i == correct_slot:
                choices.append(correct)
            else:
                choices.append(distractors[d_idx])
                d_idx += 1
        letters = ["A", "B", "C", "D"]
        choices_str = "\n".join(f"({letters[i]}) {choices[i]}" for i in range(4))
        full_q = f"Question: {question_text}\n{choices_str}"
        answer_letter = letters[correct_slot]
        return full_q, answer_letter

    # ---- AQuA-RAT ----
    # Columns: question (str), options (list like ['A)text', 'B)text', ...]), rationale, correct (letter)
    if "aqua_rat" in dataset_name.lower():
        question_text = example.get("question", "")
        options = example.get("options", [])
        opts_parts = []
        for opt in options:
            opt = str(opt).strip()
            m = re.match(r'^([A-Ea-e])\)(.*)', opt)
            if m:
                letter, text = m.group(1).upper(), m.group(2).strip()
                opts_parts.append(f"({letter}) {text}")
            else:
                opts_parts.append(opt)
        opts_str = "\n".join(opts_parts)
        full_q = f"Question: {question_text}\n{opts_str}"
        gold = example.get("correct", "A").strip().upper()
        return full_q, gold

    # ---- RACE-High (ehovy/race) ----
    # Columns: example_id, article (passage), answer (letter A-D), question, options (list of 4 strings)
    if "race" in dataset_name.lower():
        article = example.get("article", "")[:1500].rstrip()  # truncate long passages
        question_text = example.get("question", "")
        options = example.get("options", [])
        letters = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"({letters[i]}) {options[i]}" for i in range(min(len(options), 4)))
        full_q = f"Passage: {article}\n\nQuestion: {question_text}\n{opts_str}"
        gold = example.get("answer", "A").strip().upper()
        return full_q, gold

    # ---- ReClor ----
    # Columns: id_string, context (passage), question, answers (list of 4 strings), label (int 0-3)
    if "reclor" in dataset_name.lower():
        context = example.get("context", "")[:1500].rstrip()  # truncate long passages
        question_text = example.get("question", "")
        answers = example.get("answers", [])
        letters = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"({letters[i]}) {answers[i]}" for i in range(min(len(answers), 4)))
        full_q = f"Context: {context}\n\nQuestion: {question_text}\n{opts_str}"
        label_int = int(example.get("label", 0))
        gold = letters[label_int] if 0 <= label_int < 4 else "A"
        return full_q, gold

    # ---- GPQA (hendrydong/gpqa_diamond_mc) ----
    # Columns: problem (full MCQ text with (A)-(D) embedded and a trailing
    #   instruction line), solution (\boxed{X}), domain
    # We strip the trailing "Please write your final answer..." instruction line
    # so that format_prompt can add its own instruction.
    if "gpqa" in dataset_name.lower():
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        domain = example.get("domain", "")
        # Strip trailing answer-format instruction from problem text
        lines = problem.strip().splitlines()
        cleaned_lines = [l for l in lines
                         if "please write your final answer" not in l.lower()
                         and "\\boxed{" not in l.lower()]
        cleaned_problem = "\n".join(cleaned_lines).strip()
        if domain:
            cleaned_problem = f"[{domain}] {cleaned_problem}"
        # Extract gold letter from \boxed{X} in solution
        m = re.search(r'\\boxed\{([A-Da-d])\}', solution)
        gold_letter = m.group(1).upper() if m else "A"
        return cleaned_problem, gold_letter

    # ---- MBPP (google-research-datasets/mbpp) ----
    # Columns: task_id, text (description), code (reference), test_list,
    #          test_setup_code, challenge_test_list
    # Gold = JSON blob with reference code + tests so both conformal canonicalization
    # and execution-based verification can use it.
    if "mbpp" in dataset_name.lower():
        description = example.get("text", "")
        ref_code = example.get("code", "")
        test_list = example.get("test_list", [])
        setup = example.get("test_setup_code", "")
        gold = json.dumps({"ref_code": ref_code, "tests": test_list, "setup": setup})
        # Extract function signature (first line of ref_code) so the model knows
        # the required function name and parameter names.  Without this the model
        # invents a different name and every test assertion fails with NameError.
        sig_line = ""
        if ref_code:
            first_line = ref_code.strip().splitlines()[0].strip()
            if first_line.startswith("def "):
                sig_line = first_line.rstrip(":")
        if not sig_line and test_list:
            # Fallback: parse function name from first assertion
            import re as _re
            m = _re.match(r"assert\s+(\w+)\s*\(", test_list[0].strip())
            if m:
                sig_line = m.group(1)
        # Build enriched description that includes the required signature and
        # example test cases so the model can verify its implementation inline.
        tests_preview = "\n".join(f"  {t}" for t in test_list[:3])
        enriched = description
        if sig_line:
            enriched += f"\n\nRequired function signature:\n  {sig_line}"
        if tests_preview:
            enriched += f"\n\nExample tests:\n{tests_preview}"
        return enriched, gold

    # ---- HumanEval (openai/openai_humaneval) ----
    # Columns: task_id, prompt (fn signature + docstring), canonical_solution,
    #          test (check() function), entry_point
    # The model receives the prompt and must complete the function body.
    if "humaneval" in dataset_name.lower():
        prompt = example.get("prompt", "")
        entry_point = example.get("entry_point", "")
        test_fn = example.get("test", "")
        ref_solution = example.get("canonical_solution", "")
        gold = json.dumps({
            "prompt": prompt,
            "ref_solution": ref_solution,
            "test_fn": test_fn,
            "entry_point": entry_point,
        })
        # Return just the prompt text as the question (model completes it)
        return prompt, gold

    # ---- MATH-500 (HuggingFaceH4/MATH-500) ----
    # Columns: problem, solution, answer, subject, level
    # The "answer" field is the already-extracted answer (no \boxed wrapper).
    # Must be checked BEFORE hendrycks_math to avoid the generic "math" check.
    if "math-500" in dataset_name.lower() or "math500" in dataset_name.lower():
        question_text = example.get("problem", "")
        gold = example.get("answer", "").strip()
        return question_text, gold

    # ---- MATH (EleutherAI/hendrycks_math) ----
    # Columns: problem, level, type, solution
    # Gold answer is inside \\boxed{} in solution field.
    if "hendrycks_math" in dataset_name.lower():
        question_text = example.get("problem", "")
        solution = example.get("solution", "")
        # Extract \boxed{...} content as gold answer
        boxed_match = re.search(r'\\boxed\{([^{}]*)\}', solution)
        if boxed_match:
            gold = boxed_match.group(1).strip()
        else:
            # Fallback: last line of solution
            gold = solution.strip().split("\n")[-1].strip()
        return question_text, gold

    # ---- CommonsenseQA ----
    # Columns: question (str), choices {'label': [...], 'text': [...]}, answerKey (str 'A'-'E')
    if "commonsense_qa" in dataset_name.lower():
        question_text = example.get("question", "")
        choices = example.get("choices", {})
        labels = choices.get("label", [])
        texts  = choices.get("text", [])
        choices_str = "\n".join(f"({lbl}) {txt}" for lbl, txt in zip(labels, texts))
        full_q = f"Question: {question_text}\n{choices_str}"
        answer_key = example.get("answerKey", "").strip().upper()
        return full_q, answer_key   # gold is 'A', 'B', 'C', 'D', or 'E'

    # ---- BIG-Bench Hard (lukaemon/bbh) — MUST be before generic fallbacks ----
    # Each example: 'input' (full question w/ embedded choices for MCQ), 'target'
    # (gold answer: '(A)'/'(B)', 'True'/'False', 'Yes'/'No', integer string).
    if "lukaemon/bbh" in dataset_name.lower() or ("bbh" in dataset_name.lower() and "lukaemon" in dataset_name.lower()):
        question_text = example.get("input", "")
        gold = str(example.get("target", "")).strip()
        return question_text, gold

    # ---- GSM8K / standard math ----
    if "question" in example:
        return example["question"], example.get("answer", example.get("label", ""))
    if "problem" in example:
        return example["problem"], example.get("answer", example.get("label", ""))

    # ---- Generic text classification ----
    if "text" in example:
        return example["text"], str(example.get("label", ""))
    if "sentence" in example:
        return example["sentence"], str(example.get("label", ""))

    # ---- Fallback ----
    for key, val in example.items():
        if isinstance(val, str):
            return val, str(example.get("label", ""))
        return question_text, gold

    raise KeyError(f"Could not extract text from example with keys: {list(example.keys())}")


def load_gsm8k_splits(dataset_name: str, dataset_config: str, n_train: int, n_cal: int, n_eval: int, seed: int):
    """Load and split a HuggingFace dataset into train / calibration / eval subsets.

    Handles datasets with different eval split names:
    - Most datasets: "test"
    - commonsense_qa: test set is unlabeled → use "validation" instead
    - No test/validation: carve off 20% of train
    - lukaemon/bbh with config 'mix6': concatenates 6 representative subtasks
    """
    assert os.environ.get("HF_DATASETS_CACHE"), "HF_DATASETS_CACHE must be set."

    # ---- BBH multi-task mix (lukaemon/bbh, config='mix6') -------------------
    # Load each subtask's test split and concatenate into one flat dataset.
    # Each example gets an extra 'task' field for debugging.
    _ds_lower_ld = dataset_name.lower()
    if ("lukaemon/bbh" in _ds_lower_ld or ("bbh" in _ds_lower_ld and "lukaemon" in _ds_lower_ld)) \
            and str(dataset_config).strip().lower() in ("", "mix6", "none"):
        from datasets import concatenate_datasets, Dataset as HFDataset
        task_dsets = []
        for _task in _BBH_MIX6_TASKS:
            _td = load_dataset("lukaemon/bbh", _task)["test"]
            # Add a 'task' field to each example so debugging is easy
            _td = _td.map(lambda ex, t=_task: {**ex, "task": t})
            task_dsets.append(_td)
        combined = concatenate_datasets(task_dsets)
        rng0 = np.random.default_rng(seed)
        all_idx = rng0.permutation(len(combined))
        n_eval_actual = min(n_eval, len(combined) // 3)
        eval_split  = combined.select(all_idx[:n_eval_actual].tolist())
        train_pool  = combined.select(all_idx[n_eval_actual:].tolist())
        rng = np.random.default_rng(seed + 1)
        n_avail = len(train_pool)
        n_want  = min(n_train + n_cal, n_avail)
        tc_idx  = rng.permutation(n_avail)[:n_want]
        n_cal_actual = min(n_cal, n_want // 2)
        cal_idx = tc_idx[:n_cal_actual]
        tr_idx  = tc_idx[n_cal_actual:]
        eval_idx = rng.permutation(len(eval_split))[:n_eval]
        return (
            train_pool.select(tr_idx.tolist()),
            train_pool.select(cal_idx.tolist()),
            eval_split.select(eval_idx.tolist()),
        )

    # Pass config as None when empty/None — some datasets (e.g. HumanEval)
    # have no sub-configs and error when an empty string is provided.
    _config = dataset_config if dataset_config and str(dataset_config).strip() else None
    try:
        ds = load_dataset(dataset_name, _config)
    except ValueError as _e:
        # Offline cache may use a different config name than the default 'main'
        # e.g. openai/openai_humaneval is cached as config 'openai_humaneval'.
        # Parse the available config from the error message and retry once.
        # Handles both: "Available configs in the cache: ['x']"
        # and:          "BuilderConfig 'y' not found. Available: ['x']"
        _m = re.search(r"Available(?:[^:]*): \['([^']+)'", str(_e))
        if _m:
            ds = load_dataset(dataset_name, _m.group(1))
        else:
            raise

    # ---- HumanEval and other test-only datasets: synthesise a train split ----
    # These datasets ship only a "test" split, so we randomly partition it.
    ds_lower = dataset_name.lower()
    if any(k in ds_lower for k in _TEST_ONLY_DATASETS) or (
        "train" not in ds and "test" in ds
    ):
        rng0 = np.random.default_rng(seed)
        all_examples = ds["test"]
        all_idx = rng0.permutation(len(all_examples))
        n_eval_actual = min(n_eval, len(all_examples) // 3)
        eval_split  = all_examples.select(all_idx[:n_eval_actual].tolist())
        train_split = all_examples.select(all_idx[n_eval_actual:].tolist())
    else:
        train_split = ds["train"]

        # For hendrycks_math: filter train/test to hard levels only (Level 3-5)
        # so GRPO training is non-trivial for a 7B model.
        if "hendrycks_math" in dataset_name.lower():
            ds = ds.filter(lambda ex: ex.get("level", "") in _MATH_HARD_LEVELS)
            train_split = ds["train"]

        # Decide which split to use for eval
        use_validation = any(k in ds_lower for k in _VALIDATION_ONLY_DATASETS)

        if use_validation and "validation" in ds:
            eval_split = ds["validation"]
        elif "test" in ds:
            eval_split = ds["test"]
        elif "validation" in ds:
            eval_split = ds["validation"]
        else:
            # No separate eval split: hold out 20% of train
            rng0 = np.random.default_rng(seed)
            all_idx = rng0.permutation(len(train_split))
            n_hold = max(n_eval, len(train_split) // 5)
            eval_split  = train_split.select(all_idx[:n_hold].tolist())
            train_split = train_split.select(all_idx[n_hold:].tolist())

    rng = np.random.default_rng(seed)
    # Gracefully handle datasets smaller than n_train + n_cal
    n_avail = len(train_split)
    n_want  = min(n_train + n_cal, n_avail)
    train_idx = rng.permutation(n_avail)[:n_want]
    # Split into cal (first n_cal, capped) and train (remainder)
    n_cal_actual = min(n_cal, n_want // 2)   # ensure at least half goes to training
    cal_idx = train_idx[:n_cal_actual]
    tr_idx  = train_idx[n_cal_actual:]

    eval_idx = rng.permutation(len(eval_split))[:n_eval]

    return (
        train_split.select(tr_idx.tolist()),
        train_split.select(cal_idx.tolist()),
        eval_split.select(eval_idx.tolist()),
    )
