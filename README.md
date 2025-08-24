# Synthetic Survey (OpenAI ChatGPT API, Python)

This tiny project runs a **synthetic survey** by asking OpenAI's models to role‑play respondents with configurable demographics.

## What it does

- Lets you **manually set or randomly generate** respondent traits: MBTI, age, sex, nationality, education, and political opinion  
  (US: `Democrat`/`Republican`; South Korea: `진보`/`보수`).
- Sends your **questionnaire** (default in `questions.yaml`) to the model **with the persona description**.
- Collects and saves results as both **CSV** and **JSONL** and prints quick summary tables.
- Keeps your API key **in a separate file** (`api_key.txt`) or reads `OPENAI_API_KEY` from env.
- Extra niceties: deterministic `--seed`, retry/backoff, strict JSON responses, and schema validation.

> ⚠️ You’re responsible for checking that creating synthetic datasets is acceptable for your use case. These answers come from a model, not real people. The results doesn't represent the human population in any case. Please refer to: *"Synthetic Replacements for Human Survey Data? The Perils of Large Language Models"* by Bisbee, et al.

## Install & run

```bash
pip install -r requirements.txt
# Put your key into api_key.txt (single line), or export OPENAI_API_KEY
python synthetic_survey.py --n 10 --randomize
```

### Common options

```bash
# Random personas (override any field you want)
python synthetic_survey.py --n 50 --randomize

# Manual persona for everyone
python synthetic_survey.py --n 5 \
  --mbti INTP --age 29 --sex female \
  --nationality "United States" --education "Bachelor's" \
  --politics Democrat

# Age range
python synthetic_survey.py --n 50 --randomize --age 25-40

# Mix: set some, randomize the rest
python synthetic_survey.py --n 20 --randomize --nationality "South Korea" --politics 진보

# Use your own questionnaire file (YAML or JSON)
python synthetic_survey.py --n 30 --questions-file my_questions.yaml

# 한국어 프롬프트
python3 survey.py --n 5 --questions-file questions.yaml --lang ko
```

Outputs are written to `./out/` as timestamped `csv` and `jsonl` files.

## Notes on the OpenAI API

- This script uses the **Chat Completions** endpoint through the official Python SDK.  
- We request **JSON output** from the model to make parsing robust.  
- You can change the model with `--model MODEL_NAME` (e.g., `gpt-4o-mini`) depending on what you have access to.

See official docs for the latest Python usage and structured outputs:
- API Reference (Chat/Responses): https://platform.openai.com/docs/api-reference  
- Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs  
- Libraries (Python SDK): https://platform.openai.com/docs/libraries

## File layout

```
synthetic_survey/
├─ synthetic_survey.py      # main script (CLI)
├─ questions.yaml           # default questionnaire
├─ requirements.txt         # python deps
├─ api_key.txt              # put your API key here (one line)
└─ out/                     # results will be written here
```

---

**License:** MIT (do whatever, just don't hold me liable).