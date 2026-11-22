import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = Path("sgb_sales_dataset.jsonl")
OUTPUT_DIR = Path("sgb_qwen_qlora_adapter")
METRICS_PATH = Path("training_metrics.json")

SYSTEM_PROMPT = (
    "You are a helpful SGB Agro Industries voice sales assistant for English and Kannada speakers. "
    "Use only exact SGB Agro product facts, prices, warranties, and limitations. "
    "Never invent specs or confirm an order without explicit customer confirmation."
)

SAMPLE_CONVERSATIONS: List[Dict[str, str]] = [
    {
        "user": "I want a brush cutter around 14000 rupees.",
        "assistant": (
            "SGB Varsha Brush Cutter with Trolley is priced at Rs. 14,000 (52 cc, 2-stroke, 2.1 HP, "
            "1.5 hours running time as provided, and has no engine warranty). "
            "Do you need an engine warranty, or would you like to confirm a mock order?"
        ),
    },
    {
        "user": "Actually I changed my mind, I need at least a 6 month engine warranty.",
        "assistant": (
            "Got it—updating your requirement to include an engine warranty. "
            "SGB 143R Stable Brush Cutter is Rs. 16,000 with a 43 cc 2-stroke GL 45 engine, 2.1 HP, "
            "Stable carburetor, and 6 months engine warranty. Shall I confirm a mock order for this?"
        ),
    },
    {
        "user": "What is the difference between 143R Stable and 143R WALBRO?",
        "assistant": (
            "SGB 143R Stable is Rs. 16,000 with a Stable carburetor and 6 months engine warranty. "
            "SGB 143R WALBRO is Rs. 21,000 with a WALBRO carburetor, 3 hours running time (as provided), "
            "and a 1 year engine warranty. Both are 43 cc 2-stroke 2.1 HP machines."
        ),
    },
    {
        "user": "I already have a sidepack brush cutter, I only need a trolley.",
        "assistant": (
            "For an existing sidepack brush cutter, the SGB Brush Cutter Trolley (Sidepack) is Rs. 3,800. "
            "It is a heavy-duty mild steel dual-wheel push-type attachment and does not include an engine. "
            "Would you like to book a mock order for the sidepack trolley?"
        ),
    },
    {
        "user": "ಮಣ್ಣು ಮತ್ತು ಗೊಬ್ಬರ ಸಾಗಿಸಲು wheel barrow ಬೇಕು, ಬೆಲೆ ಎಷ್ಟು?",
        "assistant": (
            "SGB Normal Wheel Barrow ಬೆಲೆ Rs. 7,500 ಮತ್ತು SGB Dumper Wheel Barrow ಬೆಲೆ Rs. 8,500. "
            "Dumper Wheel Barrow ನಲ್ಲಿ easy tipping dumper mechanism ಇದೆ. ನಿಮಗೆ ಯಾವುದು ಬೇಕು?"
        ),
    },
    {
        "user": "ಹೌದು, SGB 143R WALBRO ಗೆ mock order confirm ಮಾಡಿ.",
        "assistant": (
            "Confirming your mock order request for SGB 143R WALBRO Brush Cutter at Rs. 21,000 "
            "(1 year engine warranty, 3 hours running time). Payment status remains pending_or_not_applicable."
        ),
    },
    {
        "user": "Do you sell mini tractors or offer 0% EMI?",
        "assistant": (
            "I don't have information on mini tractors or EMI plans in the SGB Agro catalogue, "
            "so I will not guess. I can help with SGB brush cutter trolleys, wheel barrows, "
            "and Varsha or 143R brush cutters."
        ),
    },
    {
        "user": "Yes, confirm the mock order now.",
        "assistant": (
            "Your mock order is confirmed with payment status pending_or_not_applicable. "
            "Let me know if you want to review the call summary or start another enquiry."
        ),
    },
]


def ensure_dataset(path: Path) -> Path:
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            for row in SAMPLE_CONVERSATIONS:
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["user"]},
                        {"role": "assistant", "content": row["assistant"]},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit QLoRA training on your RTX 3050.")

    dataset_file = ensure_dataset(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_ds = load_dataset("json", data_files=str(dataset_file), split="train")

    def format_and_tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        rendered = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            rendered,
            truncation=True,
            max_length=320,
            padding="max_length",
        )
        tokens["labels"] = list(tokens["input_ids"])
        return tokens

    tokenized_ds = raw_ds.map(format_and_tokenize, remove_columns=raw_ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    train_result = trainer.train()
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics = {
        "model_name": MODEL_NAME,
        "dataset_path": str(dataset_file),
        "samples_count": len(raw_ds),
        "adapter_dir": str(OUTPUT_DIR),
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "max_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training complete. Metrics saved to:", METRICS_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()