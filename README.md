# Synthetic Survey (OpenAI Chat API, Python)

This project runs a **synthetic survey** by asking OpenAI's LLMs to role-play survey respondents with configurable or dataset-grounded demographics. 

It supports both generating randomized custom personas and loading real-world census-grounded South Korean personas from NVIDIA's large-scale dataset.

---

## 🌟 Key Features

*   **Dual Persona Modes:**
    *   **Custom / Randomized Mode:** Generate personas on-the-fly specifying MBTI, age, sex, nationality, education, and political orientation.
    *   **South Korea Census Mode:** Ground your survey using high-fidelity personas from NVIDIA's [nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) dataset.
*   **Rich Roleplay Context:** When using the South Korean dataset, the models are fed detailed profiles containing occupation, cultural background, marital status, housing details, hobbies, and career goals.
*   **Localized Prompting:** Fully supports prompting and response evaluation in both English and Korean (`--lang ko`).
*   **Robust Execution:** Automatically validates LLM responses against a strict JSON schema, implements exponential backoff/retry, and supports reproducible runs via a random seed (`--seed`).
*   **Clean Outputs:** Saves survey responses as flattened **CSV** and structured **JSONL** files inside an `./out/` directory, automatically dynamically including any attributes unique to the persona.

> [!WARNING]
> Synthetic surveys represent language model priors, not real human responses. These results should be checked carefully against real human data. For details on potential biases, refer to: *"Synthetic Replacements for Human Survey Data? The Perils of Large Language Models"* by Bisbee, et al.

---

## 🚀 Installation & Setup

1.  **Clone the Repository & Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure your OpenAI API Key:**
    Export it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
    *Alternatively, you can save it as a single-line plaintext file named `api_key.txt` in the root of the project.*

---

## 📖 Command-Line Usage

### 1. South Korea Census Mode (`--nemotron-korea`)
Use NVIDIA's synthetic South Korean population to back your survey. This mode loads the [nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) dataset.

#### Dataset Columns

The dataset contains the following columns for each persona:

*   **Structured Demographics & Location:**
    *   `uuid`: Unique identifier for the persona.
    *   `age`: Integer age (adults aged 19 and older).
    *   `sex`: Biological sex (`"남자"` or `"여자"`).
    *   `marital_status`: Marital status (e.g. `"배우자있음"`, `"미혼"`, `"사별"`, `"이혼"`).
    *   `military_status`: Military status (e.g. `"비현역"`).
    *   `family_type`: Family size/structure (e.g. `"배우자와 거주"`, `"배우자·자녀와 거주"`).
    *   `housing_type`: Housing type (e.g. `"아파트"`, `"단독주택"`).
    *   `education_level`: Education attainment (e.g. `"고등학교"`, `"4년제 대학교"`, `"대학원"`).
    *   `bachelors_field`: Academic major field (if college-educated).
    *   `occupation`: Specific job role or profession.
    *   `province` / `district`: Province and local district (e.g. `"광주"` / `"광주-서구"`).
    *   `country`: Country of residence (always `"대한민국"`).
*   **Narrative Persona Facets:**
    *   `persona`: Baseline textual summary of the persona.
    *   `cultural_background`: Regional dialects, regional values, and generational details.
    *   `professional_persona`: Detailed workplace dynamics and professional context.
    *   `career_goals_and_ambitions`: Narrative detailing career drive and financial goals.
    *   `skills_and_expertise` / `skills_and_expertise_list`: Overview and list of professional capabilities.
    *   `hobbies_and_interests` / `hobbies_and_interests_list`: Overview and list of leisure interests.
    *   `sports_persona` / `arts_persona` / `travel_persona` / `culinary_persona` / `family_persona`: Detailed behavioral facets.

```bash
# Run a survey of 10 South Korean personas with Korean prompts
python synthetic_survey.py --nemotron-korea --n 10 --lang ko

# Filter the dataset by sex and age range, and assign a political stance
python synthetic_survey.py --nemotron-korea --n 15 --sex female --age 25-40 --politics "진보" --lang ko

# Filter by occupation, marital status, and province (supports English mapping & substring matching)
python synthetic_survey.py --nemotron-korea --n 5 \
  --sex male \
  --marital-status married \
  --province "서울" \
  --occupation "개발자" \
  --lang ko
```

*Note: The first run of `--nemotron-korea` will download and cache the 2.0GB dataset from Hugging Face. Subsequent runs will load instantly from cache.*

#### Nemotron Option Compatibility

When running with `--nemotron-korea`, some command-line options behave differently or are ignored:

| Option | Supported? | Details / Behavior in Nemotron Mode |
| :--- | :--- | :--- |
| `--n` | **Yes** | Determines how many personas to sample from the dataset. |
| `--age` | **Yes** | Filters the dataset by age (supports single integers or ranges, e.g., `25-40`). |
| `--sex` | **Yes** | Filters the dataset by sex. Maps `male` to `"남자"`, `female` to `"여자"`, and `non-binary` to `"남자"`. |
| `--education`| **Yes** | Filters the dataset by education level. Maps English levels (e.g. `Bachelor's`) to Korean levels (e.g. `4년제 대학교`). |
| `--politics` | **Yes** | Manually sets a political stance on the sampled personas (since politics is not present in the dataset columns). |
| `--randomize`| **Yes** | If specified, randomly assigns a political stance (`진보` / `보수`) to personas when `--politics` is omitted. |
| `--seed` | **Yes** | Ensures deterministic, reproducible sampling from the dataset. |
| `--marital-status` | **Yes** | Filters the dataset by marital status. Maps English (e.g. `married`, `single`) to Korean values (e.g. `배우자있음`, `미혼`). |
| `--housing-type` | **Yes** | Filters the dataset by housing type. Maps English (e.g. `apartment`, `house`) to Korean values (e.g. `아파트`, `단독주택`). |
| `--occupation`| **Yes** | Filters by occupation using a case-insensitive substring search (e.g. `개발자` or `회사원`). |
| `--province` | **Yes** | Filters by province using a case-insensitive substring search (e.g. `서울` or `광주`). |
| `--district` | **Yes** | Filters by district using a case-insensitive substring search (e.g. `서구` or `강남구`). |
| `--mbti` | **No** | *Ignored.* Nemotron personas do not include MBTI traits. |
| `--nationality`| **No** | *Ignored.* Nationality is always fixed to `"South Korea"`. |

### 2. Custom / Randomized Persona Mode
Generate personas dynamically by specifying constraints.

```bash
# Generate 50 randomized personas across different nationalities
python synthetic_survey.py --n 50 --randomize

# Force a specific persona profile for everyone
python synthetic_survey.py --n 5 \
  --mbti INTP --age 29 --sex female \
  --nationality "United States" --education "Bachelor's" \
  --politics Democrat

# Constraint mix: set some properties, randomize the rest
python synthetic_survey.py --n 20 --randomize --nationality "South Korea" --politics 진보
```

### 3. Customizing the Questionnaire
You can define your questions in a YAML or JSON file. By default, the script loads `questions.yaml`.

```bash
python synthetic_survey.py --n 10 --questions-file my_questions.yaml
```

---

## 🛠️ Command-Line Arguments Reference

| Option | Type | Description |
| :--- | :--- | :--- |
| `--n` | `int` | **Required.** Number of synthetic respondents to generate. |
| `--nemotron-korea` | `flag` | Use NVIDIA's South Korean persona dataset instead of random generation. |
| `--lang` | `ko` \| `en` | Prompt language (`en` or `ko`). Default is `en`. |
| `--questions-file` | `str` | Path to the YAML/JSON questions file (default: `questions.yaml`). |
| `--api-key-file` | `str` | Path to your API key file (default: `api_key.txt`). |
| `--model` | `str` | OpenAI Chat model to use (default: `gpt-4o-mini`). |
| `--randomize` | `flag` | Randomize unspecified persona fields for each respondent (politics in Nemotron mode). |
| `--seed` | `int` | Random seed for reproducibility. |
| `--age` | `str` | Age filter. Can be an integer (e.g. `30`) or range (e.g. `25-40`). |
| `--sex` | `str` | Sex filter (`male`, `female`, or `non-binary` / `"남자"`, `"여자"`). |
| `--education` | `str` | Education level filter (e.g. `Bachelor's` / `4년제 대학교`). |
| `--politics` | `str` | Political orientation filter (e.g. `Democrat` / `진보`). |
| `--marital-status`| `str` | **Nemotron Mode Only.** Marital status filter (e.g. `married`, `single`, `widowed`, `divorced` or Korean). |
| `--housing-type` | `str` | **Nemotron Mode Only.** Housing type filter (e.g. `apartment`, `villa`, `house` or Korean). |
| `--occupation` | `str` | **Nemotron Mode Only.** Occupation substring filter (e.g. `개발자`). |
| `--province` | `str` | **Nemotron Mode Only.** Province substring filter (e.g. `서울`). |
| `--district` | `str` | **Nemotron Mode Only.** Local district substring filter (e.g. `강남구`). |
| `--temperature` | `float` | Sampling temperature for the model (default: `0.8`). |

---

## 📁 Repository Layout

```
synthetic-survey/
├── synthetic_survey.py  # Main execution script
├── questions.yaml       # Default survey questions
├── requirements.txt     # Python dependencies
├── api_key.txt          # API Key storage (optional)
└── out/                 # CSV and JSONL survey outputs
```

---

## 📚 References

If you use the South Korea persona dataset in your research or application, please reference the official work:

> Kim, H., Ryu, J., Lee, J., Ryu, H., Praveen, K., Prayaga, S., Thadaka, K., Jennings, W., Sadeghi, B., Sharabiani, A., Choi, Y., & Meyer, Y. (2024). *Nemotron-Personas-Korea*. NVIDIA. Hugging Face Dataset: [nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)

---

## 📄 License
This project is licensed under the MIT License.
