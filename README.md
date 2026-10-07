# Synthetic Survey

Run synthetic survey experiments with pre-constructed, demographically grounded South Korean personas from [NVIDIA Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea). An OpenAI model answers a validated questionnaire as each sampled persona. Every source field—including the travel, family, food, sports, and arts narratives—is supplied to the model.

The tool produces **synthetic model responses, not measured human opinions or population estimates**. Demographic grounding does not establish behavioral validity. Use results to explore hypotheses and test questionnaires; compare them with human survey data before drawing substantive conclusions. NVIDIA documents independence assumptions and missing interactions in the source dataset.

## Install

Python 3.10+ on macOS or Linux (run locking uses `fcntl`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For model calls, set `OPENAI_API_KEY` in your environment, or put it in the git-ignored `api_key.txt`. `--api-key-file PATH` selects another file; the environment takes precedence. Credentials are never included in run artifacts. You do not need a key for planning, summaries, or tests.

## Try it without an API call or dataset download

The three bundled personas are **handwritten test fixtures**, not a sample of NVIDIA's population.

```bash
python synthetic_survey.py --n 3 \
  --personas-file examples/personas.jsonl \
  --seed 42 --survey-date 2026-10-07 \
  --dry-run --run-dir out/demo
```

Inspect `out/demo/manifest.json` and `out/demo/cohort.json`. The latter contains each persona and the exact messages that will be sent. A dry run creates a new immutable experiment plan, with all respondents marked pending. It does not fabricate answers.

Execute that plan after configuring your key:

```bash
python synthetic_survey.py --resume out/demo
```

## Survey the Nemotron population

Nemotron is the default source; `--nemotron-korea` remains a supported explicit alias. The first real dataset load downloads and caches the source through Hugging Face. It can require substantial disk space and time; cached loads and filtering still involve work.

```bash
# Korean questionnaire and prompts are the default.
python synthetic_survey.py --nemotron-korea --n 100 --seed 42

# Inspect a filtered cohort before making model calls.
python synthetic_survey.py --n 20 --sex 여자 --age 25-40 \
  --province 서울 --education "Bachelor's" \
  --seed 42 --dry-run --run-dir out/seoul-women

# An English-language experiment uses questions.yaml by default.
python synthetic_survey.py --n 10 --lang en --seed 42
```

Sampling is uniform over eligible **rows**, without replacement by default. It is not stratified or weighted. Filtering defines a subgroup; its results must not be interpreted as national estimates. The loader filters Arrow-backed demographic columns and materializes only sampled records, rather than converting all narrative columns to pandas.

The dataset's mutable revision (default `main`) is resolved to a commit **before** loading and saved in the manifest. Use `--dataset-revision COMMIT` to repeat a particular source version. Local JSON/JSONL input must use Nemotron field names; its file hash and path are recorded. Required fields are `uuid`, `age`, `sex`, `education_level`, and `persona`, plus any fields used by your filters. Ages 19–120 are accepted without excluding the dataset's oldest adults. Records are validated before any model calls.

### Filters and experimental overrides

| Argument | Behavior |
| --- | --- |
| `--age 30` / `--age 25-40` | Exact age or inclusive ascending range; invalid ranges are rejected, not clamped |
| `--sex male` / `female` / `남자` / `여자` | Exact source sex category; non-binary is unavailable and explicitly rejected in Nemotron mode |
| `--education` | Korean categories including `무학`, `초등학교`, `중학교`, `고등학교`, `2~3년제 전문대학`, `4년제 대학교`, `대학원`; English aliases also supported |
| `--marital-status` | Korean value or `married`, `single`, `unmarried`, `widowed`, `divorced` |
| `--housing-type` | Korean value or `apartment`, `villa`, `multi-family`, `house`, `single-family` |
| `--occupation`, `--province`, `--district` | Literal, case-insensitive substrings; never regular expressions |
| `--politics TEXT` | Assign the same explicit experimental political orientation to each persona; not a population filter |
| `--with-replacement` | Sample with replacement explicitly; unique-persona counts are reported |

English education aliases are `High school`, `Associate's`, `Bachelor's`, `Master's`, and `Doctorate`. The last two both select `대학원`; the dataset does not distinguish those degrees in this field. The source sex field is not a gender-identity measure.

No political orientation is randomly invented. An omitted orientation stays unspecified. Political overrides are saved separately from the original source fields so their experimental origin is visible. Repeated draws of the same persona are identifiable by the retained source UUID and must not be treated as independent human respondents.

## Questionnaires

The defaults are `questions.ko.yaml` for Korean and `questions.yaml` for English. `--questions-file PATH` accepts YAML or JSON, either a list or an object containing only a `questions` list. Custom question and option text is used verbatim; `--lang` does not translate it.

```yaml
questions:
  - id: satisfaction
    text: "현재 삶에 얼마나 만족하십니까? (0~10)"
    type: scale
    minimum: 0
    maximum: 10
  - id: priorities
    text: "중요한 분야를 최대 두 개 선택하세요."
    type: multi
    options: ["교육", "보건의료", "환경"]
    min_choices: 1
    max_choices: 2
  - id: trips
    text: "지난 12개월 동안 해외여행을 몇 번 다녀오셨습니까?"
    type: number
    minimum: 0
    allow_na: true
```

| Type | Validation |
| --- | --- |
| `free` | Nonblank string |
| `single` | One exact option string |
| `multi` | Unique option strings; default minimum 1, maximum the smaller of 3 or the number of options |
| `scale` | Integer within bounds, default 1–5; booleans and strings are rejected |
| `number` | Finite numeric value within optional bounds; booleans are rejected |

IDs must be unique and begin with a letter, followed by letters, digits, underscores or hyphens, up to 64 characters. Unknown fields and invalid definitions fail before model calls. `allow_na: true` permits JSON `null`; otherwise every question requires a typed answer. Literal `N/A` is not a universal escape from validation. Missing answers, extra IDs, duplicate JSON keys, invalid options and duplicate multi-select selections are rejected.

The API receives a per-question [Structured Outputs schema](https://developers.openai.com/api/docs/guides/structured-outputs). Local validation independently checks the returned JSON before a response is marked successful. The model returns answers keyed by ID; original question text and demographic data are attached by the program, not regenerated by the model.

## Execution, recovery, and reproducibility

```bash
# Resume only pending respondents; successful and failed respondents are skipped.
python synthetic_survey.py --resume out/seoul-women

# Explicitly give failed respondents another attempt budget, preserving earlier history.
python synthetic_survey.py --resume out/seoul-women --retry-failed

# Rebuild CSV/JSONL/summary files from durable records, without a key or network calls.
python synthetic_survey.py --summarize out/seoul-women
```

- Every new run uses a unique directory, or a new directory supplied with `--run-dir`. Existing directories are never overwritten.
- The plan freezes the cohort, exact prompts, question schema, survey date, source revision, settings, seeds, dependency versions and implementation hash. Checksums detect accidental edits. Resume rejects setting overrides and changed implementation code; use the original checkout to continue an experiment after code changes.
- Each respondent has a full unique run ID and retains the source persona UUID. Request seeds are derived independently per respondent and attempt; a retry does not shift later respondents' seeds.
- Completed attempts are atomically saved, including response content, model identifier, token usage, request ID and system fingerprint when provided. CSV/JSONL exports are derived from these records and can be rebuilt.
- Invalid answers, truncated output, connection failures and transient HTTP errors receive bounded exponential-backoff retries. Refusals are recorded as failures without automatic retry. Non-transient HTTP errors stop the run, preserving unattempted respondents as pending. The SDK's hidden retries are disabled.
- Exhausted respondents are retained as failures and are not replaced with new people. Summaries show planned, successful and failed demographic distributions so attrition is visible.
- A process lock prevents concurrent execution of the same run. An interrupted, uncommitted API request may be repeated on resume; this is not an exactly-once billing guarantee.
- A seed makes local sampling reproducible for the same source and software environment. Model output is not guaranteed to be identical, even with a request seed and model snapshot. The tool saves provenance rather than promising deterministic model behavior.

Exit codes: `0` for completed execution/planning/summarization, `1` for a run with failed or pending respondents, `2` for input/artifact errors, and `130` for interruption.

### Model and runtime options

| Argument | Default / meaning |
| --- | --- |
| `--model` | `OPENAI_MODEL`, or `gpt-4o-mini-2024-07-18`; use a Chat Completions model that supports strict structured output |
| `--temperature` | `0.8`; pass `default` to omit the parameter for models that do not support it |
| `--no-model-seed` | Omit the optional API seed for models that do not support it; sampling still uses the saved seed |
| `--max-tokens` | `4096` completion-token budget; increase for long questionnaires or models using reasoning tokens |
| `--max-attempts` | `3` per respondent; interrupted pending work retains the remaining budget |
| `--timeout` | `90` seconds per API request |
| `--seed` | Generated and saved if omitted |
| `--survey-date` | Local current date in ISO format; fixes the reference date for relative questions, not the model's knowledge freshness |
| `--out-dir` | `out` parent directory |

Incompatible model parameters fail visibly; the program does not silently switch models or downgrade validation. Requests are sequential. Large surveys should first be piloted to check context length, token budget, cost, refusals, and answer distributions.

## Outputs

```text
out/run-.../
  manifest.json       # Immutable experiment definition and provenance
  cohort.json         # All sampled personas, exact messages and request seeds
  records/<id>.json   # Durable per-respondent status and attempt history
  respondents.csv     # One row per planned respondent, including failures/pending
  answers.csv         # One row per validated answer; lists encoded as JSON
  results.jsonl       # Successful responses with full persona context and metadata
  summary.json        # Completion, unique-persona, demographic and answer summaries
```

CSV text is written verbatim; JSONL retains native value types. Null answers are blank in CSV and marked with `is_na`. Demographic summaries count respondents once, not once per answer. Question summaries report denominators explicitly; multi-select counts are selection counts, and numeric means exclude null answers. There are no population weights, confidence intervals, or claims of representativeness. Free-text answers are retained without automated interpretation.

## Custom persona experiments

Custom mode is a convenience for controlled hypothetical profiles, not a demographic population generator.

```bash
python synthetic_survey.py --custom --n 5 --age 29 --sex female \
  --nationality "South Korea" --education "Bachelor's" --mbti INTP \
  --politics 진보 --seed 42

python synthetic_survey.py --custom --n 20 --randomize --seed 42 --dry-run
```

Without `--randomize`, all draws share a fixed profile (including one sampled age when a range is given). Defaults are age 35, female, South Korea, university education, unspecified MBTI and politics. With `--randomize`, unspecified age, sex, education and MBTI vary independently; nationality defaults to South Korea and politics remains unspecified. These arbitrary distributions are not calibrated to census data.

## Tests and layout

```bash
python -m unittest discover -v
```

Tests use small Arrow datasets, fixtures and a mocked HTTP transport through the real OpenAI SDK. They make no paid API calls and do not download the Nemotron dataset. CI runs on Python 3.10 and 3.12.

```text
synthetic_survey.py   # CLI entry point
survey/
  cli.py             # Arguments and experiment orchestration
  personas.py        # Source loading, filters, sampling and persona validation
  questions.py       # Questionnaire/schema/answer validation
  prompts.py         # Korean/English prompt rendering
  provider.py        # OpenAI adapter and error classification
  runner.py          # Planning, retries and resumable execution
  storage.py         # Atomic artifacts, locking, exports and summaries
questions*.yaml       # English and Korean questionnaires
examples/             # Handwritten offline fixtures
tests/                # Offline regression and integration tests
```

### Migration from the original script

Nemotron and Korean are now defaults; custom generation requires `--custom`. `--randomize` applies only to custom profiles and no longer assigns random politics to Nemotron personas. Existing `--n`, filtering, model, seed and questionnaire options remain where compatible. Output files now live in per-run directories; old result files are left untouched and cannot be resumed as v2 runs. Legacy model-shaped `answers` arrays are replaced internally by an ID-keyed object, while exported JSONL retains an answer array with original question text.

The persona dataset remains subject to its own [CC BY 4.0 attribution and dataset documentation](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea). Cite the exact dataset revision used in an experiment.
