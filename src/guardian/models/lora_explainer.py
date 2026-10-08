"""LoRA Explainer module (M11 / Exp 7).

Wraps small open-weights instruct model with PEFT LoRA adapter
for generating structured engineering explanations from battery assessments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from guardian.utils.config import get_config
from guardian.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are the AI Battery Safety Guardian diagnostic engine. "
    "Analyze the provided structured battery telemetry assessment and generate a "
    "strictly grounded, non-hallucinatory engineering explanation in valid JSON format."
)


class LoRAExplainer:
    """Inference wrapper for base LLM + LoRA adapter fine-tuned on battery explanations."""

    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.base_model_name = (
            base_model_name
            or cfg.get("lora", {}).get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
        )
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing LoRAExplainer on {self.device} (base={self.base_model_name})")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
        )

        if self.adapter_path and self.adapter_path.exists() and (self.adapter_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
        else:
            logger.info("No adapter found or specified; running unadapted base model.")

        self.model.to(self.device)
        self.model.eval()

    def format_input(self, assessment: Dict[str, Any]) -> str:
        """Format structured assessment JSON into chat prompt."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(assessment)},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{SYSTEM_PROMPT}\n\nAssessment:\n{json.dumps(assessment)}\n\nResponse:"

    def generate(
        self,
        messages_or_text: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 350,
        temperature: float = 0.2,
    ) -> str:
        """Generate response text from input prompt or chat messages."""
        if isinstance(messages_or_text, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages_or_text, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages_or_text) + "\nassistant:"
        else:
            prompt = messages_or_text

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.95
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
