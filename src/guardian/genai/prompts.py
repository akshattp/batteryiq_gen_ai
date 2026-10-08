"""Versioned prompt template manager (Section 8.2).

Loads YAML prompts from config/prompts/*.yaml and populates structured parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class PromptTemplate:
    """Represents a versioned prompt loaded from YAML."""

    def __init__(
        self,
        prompt_id: str,
        version: str,
        system: str,
        user_template: str,
        few_shot: Optional[List[Dict[str, str]]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt_id = prompt_id
        self.version = version
        self.system = system
        self.user_template = user_template
        self.few_shot = few_shot or []
        self.output_schema = output_schema or {}

    def render_messages(self, **kwargs: Any) -> List[Dict[str, str]]:
        """Format the system and user templates with given kwargs into message dictionaries."""
        user_content = self.user_template
        for k, v in kwargs.items():
            user_content = user_content.replace(f"{{{k}}}", str(v))
        messages = [{"role": "system", "content": self.system}]
        for example in self.few_shot:
            messages.append(dict(example))
        messages.append({"role": "user", "content": user_content})
        return messages


def load_prompt(prompt_name: str, prompts_dir: str | Path = "config/prompts") -> PromptTemplate:
    """Load a prompt definition from a YAML file."""
    path = Path(prompts_dir) / f"{prompt_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template {prompt_name} not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return PromptTemplate(
        prompt_id=data["id"],
        version=data["version"],
        system=data["system"],
        user_template=data["user_template"],
        few_shot=data.get("few_shot", []),
        output_schema=data.get("output_schema", {}),
    )
