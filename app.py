import json
import re

import torch
import gradio as gr
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import spaces  # provided automatically on HF Spaces

# -----------------------------
# 1. Constants
# -----------------------------
PEFT_MODEL_ID = "LlamaFactoryAI/cv-job-description-matching"
BASE_MODEL_NAME = "akjindal53244/Llama-3.1-Storm-8B"

SYSTEM_PROMPT = (
    "You analyze how well a CV matches a job description for No Skill Jobs. "
    "education is not much relevant unless specified."
    "Your ONLY output must be a single JSON object with EXACTLY these keys: "
    "matching_analysis, description, Total score, recommendation, name, email adress, phone number, home address.\n\n"
    "Constraints:\n"
    "- matching_analysis: at most 3 short bullet-like points, max 20 words each.\n"
    "- description: at most 2 sentences, max 35 words total.\n"
    "- score: integer from 0 to 100.\n"
    "- recommendation: at most 2 sentences, max 35 words total.\n\n"
    "Very important:\n"
    "- Do NOT include the full CV or job description text.\n"
    "- Do NOT wrap the JSON in backticks or any extra text.\n"
    "- Output ONLY raw JSON, nothing before or after."
)

# -----------------------------
# 2. Download & patch adapter (CPU only, safe in main process)
# -----------------------------
print("Downloading adapter...")
adapter_path = snapshot_download(PEFT_MODEL_ID)

config_path = adapter_path + "/adapter_config.json"
with open(config_path, "r") as f:
    cfg = json.load(f)

cfg["task_type"] = "CAUSAL_LM"

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("Patched adapter_config.json → task_type = CAUSAL_LM")
print("Adapter path:", adapter_path)

# -----------------------------
# 3. Globals for lazy GPU init
# -----------------------------
tokenizer = None
model = None


def build_messages(cv: str, job_description: str):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"<CV> {cv} </CV>\n<job_description> {job_description} </job_description>",
        },
    ]


def extract_json_from_text(text: str):
    """
    Try to pull a JSON object out of the model's output.
    If it fails, wrap the raw text in a fallback JSON structure.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text

    try:
        return json.loads(candidate)
    except Exception:
        return {
            "matching_analysis": [
                "Model output could not be parsed as JSON.",
            ],
            "description": text[:200],
            "score": 0,
            "recommendation": "Please try again; the model returned non-JSON output.",
        }


# -----------------------------
# 4. Main inference function (GPU)
# -----------------------------
@spaces.GPU  # required for Stateless GPU Spaces
def match_cv_job(cv: str, job_description: str):
    global tokenizer, model

    if not cv.strip() or not job_description.strip():
        return {
            "matching_analysis": ["Please provide both a CV and a job description."],
            "description": "",
            "score": 0,
            "recommendation": "Fill both text boxes and run again.",
        }

    # Lazy GPU initialization: all CUDA-related stuff happens ONLY here
    if tokenizer is None or model is None:
        print("Initializing tokenizer + model on GPU...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )

        base_model.config.pad_token_id = tokenizer.pad_token_id

        model_ = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="auto",
        )
        model_.eval()
        torch.set_grad_enabled(False)

        model = model_
        print("Model + LoRA adapter loaded successfully on GPU.")

    messages = build_messages(cv, job_description)

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    # Move tensors to the same device as the model
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = encoded["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    result = extract_json_from_text(generated_text)
    return result


# -----------------------------
# 5. Gradio interface
# -----------------------------
cv_input = gr.Textbox(
    label="CV",
    placeholder="Paste the candidate's CV here...",
    lines=18,
)

jd_input = gr.Textbox(
    label="Job Description",
    placeholder="Paste the job description here...",
    lines=8,
)

output_json = gr.JSON(label="Matching result (JSON)")

demo = gr.Interface(
    fn=match_cv_job,
    inputs=[cv_input, jd_input],
    outputs=output_json,
    title="CV–Job Description Matching API",
    description=(
        "Paste a CV and a job description. The model returns a JSON object with "
        "`matching_analysis`, `description`, `score`, and `recommendation`."
    ),
)

if __name__ == "__main__":
    demo.launch()
