"""
Model generation pipeline for the political bias experiment.

- base rows use completion/prefix-style prompts
- instruct rows use instruction/chat-style prompts

"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    BASE_MODEL,
    DO_SAMPLE,
    INSTRUCT_MODEL,
    MAX_NEW_TOKENS,
    MODEL_FAMILY,
    RANDOM_SEED,
    TEMPERATURE,
    ModelType,
)


@dataclass(frozen=True)
class GenerationSettings:
    """Generation settings shared by both models."""

    seed: int = RANDOM_SEED
    max_new_tokens: int = MAX_NEW_TOKENS
    do_sample: bool = DO_SAMPLE
    temperature: float = TEMPERATURE


@dataclass(frozen=True)
class ModelSpec:
    """Description of one model used in the experiment."""

    model_name: str
    model_type: ModelType
    model_family: str = MODEL_FAMILY


class ModelGenerator:
    """Generate responses for one causal language model."""

    def __init__(self, spec: ModelSpec, settings: GenerationSettings | None = None) -> None:
        self.spec = spec
        self.settings = settings or GenerationSettings()
        self.model = None
        self.tokenizer = None

    @staticmethod
    def set_global_seed(seed: int) -> None:
        """Set random seeds for reproducible generation."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load(self) -> None:
        """Load the model and tokenizer from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_name,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.model.to("cpu")
        self.model.eval()

    def format_prompt(self, prompt_text: str) -> str:
        """
        Format the prompt according to model type.

        Base models receive the raw completion-style prompt.
        Instruction-tuned models receive the instruction prompt wrapped in the
        tokenizer's chat template when the tokenizer provides one.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded before formatting prompts.")

        if self.spec.model_type == "base":
            return prompt_text

        messages = [{"role": "user", "content": prompt_text}]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback for instruction-tuned tokenizers without a chat template.
        return prompt_text

    def generate_one(self, prompt_text: str) -> str:
        """Generate one model response for one prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ModelGenerator.load() must be called before generation.")

        formatted_prompt = self.format_prompt(prompt_text)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.settings.do_sample:
            generation_kwargs["temperature"] = self.settings.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def unload(self) -> None:
        """Release model objects from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_for_prompts(self, prompts: pd.DataFrame) -> pd.DataFrame:
        """
        Run every prompt assigned to this model and return raw output rows.

        The input dataframe must contain only rows where `model_type` equals
        this generator's model type.
        """
        if not prompts["model_type"].eq(self.spec.model_type).all():
            raise ValueError(
                f"All prompt rows must have model_type={self.spec.model_type!r} "
                f"when generating with this model."
            )

        self.set_global_seed(self.settings.seed)
        self.load()

        rows = []
        try:
            iterator: Iterable = prompts.itertuples(index=False)
            for row in tqdm(iterator, total=len(prompts), desc=f"Generating with {self.spec.model_type}"):
                response_text = self.generate_one(row.prompt_text)

                rows.append(
                    {
                        "generation_prompt_id": row.generation_prompt_id,
                        "prompt_id": row.prompt_id,
                        "domain": row.domain,
                        "topic": row.topic,
                        "prompt_type": row.prompt_type,
                        "input_format": row.input_format,
                        "ideological_axis": row.ideological_axis,
                        "prompt_text": row.prompt_text,
                        "model_family": self.spec.model_family,
                        "model_type": self.spec.model_type,
                        "model_name": self.spec.model_name,
                        "response_text": response_text,
                        "generation_seed": self.settings.seed,
                        "temperature": self.settings.temperature,
                        "do_sample": self.settings.do_sample,
                        "max_new_tokens": self.settings.max_new_tokens,
                    }
                )
        finally:
            self.unload()

        return pd.DataFrame(rows)


def validate_generation_prompts(prompts: pd.DataFrame, expected_prompt_cases: int = 200) -> None:
    """Validate that the prompt dataframe has the columns needed for generation."""
    required_columns = {
        "generation_prompt_id",
        "prompt_id",
        "model_type",
        "input_format",
        "domain",
        "topic",
        "prompt_type",
        "prompt_text",
        "ideological_axis",
    }
    missing = required_columns.difference(prompts.columns)
    if missing:
        raise ValueError(f"Missing required prompt columns: {sorted(missing)}")

    if prompts.isna().sum().sum() != 0:
        raise ValueError("Generation prompt dataset contains missing values.")

    if not prompts["generation_prompt_id"].is_unique:
        raise ValueError("generation_prompt_id values must be unique.")

    if set(prompts["model_type"]) != {"base", "instruct"}:
        raise ValueError("Expected model_type values to be exactly {'base', 'instruct'}.")

    if set(prompts["input_format"]) != {"completion", "instruction"}:
        raise ValueError("Expected input_format values to be exactly {'completion', 'instruction'}.")

    if not prompts.groupby("prompt_id")["model_type"].nunique().eq(2).all():
        raise ValueError("Every prompt_id must have both base and instruct prompt rows.")

    expected_rows = expected_prompt_cases * 2
    if len(prompts) != expected_rows:
        raise ValueError(f"Expected {expected_rows} generation prompts, got {len(prompts)}.")


def run_generation_pipeline(
    generation_prompts: pd.DataFrame,
    output_path: Path,
    base_model: str = BASE_MODEL,
    instruct_model: str = INSTRUCT_MODEL,
    model_family: str = MODEL_FAMILY,
    settings: GenerationSettings | None = None,
    expected_prompt_cases: int = 200,
) -> pd.DataFrame:
    """
    Run the base and instruction-tuned models on their appropriate prompt rows.

    Unlike the previous version, this function does not duplicate one shared
    prompt dataframe across both models. It expects a 400-row generation prompt
    dataframe: 200 completion prompts for the base model and 200 instruction
    prompts for the instruction-tuned model.
    """
    validate_generation_prompts(generation_prompts, expected_prompt_cases=expected_prompt_cases)
    settings = settings or GenerationSettings()

    specs = [
        ModelSpec(base_model, "base", model_family),
        ModelSpec(instruct_model, "instruct", model_family),
    ]

    outputs = []
    for spec in specs:
        prompts_for_model = generation_prompts.loc[
            generation_prompts["model_type"].eq(spec.model_type)
        ].copy()

        model_outputs = ModelGenerator(spec=spec, settings=settings).generate_for_prompts(prompts_for_model)
        outputs.append(model_outputs)

    raw_outputs = pd.concat(outputs, ignore_index=True)
    expected_rows = len(generation_prompts)

    if len(raw_outputs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(raw_outputs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_outputs.to_csv(output_path, index=False)

    return raw_outputs


def validate_raw_outputs(raw_outputs: pd.DataFrame, expected_prompt_cases: int = 200) -> None:
    """Validate the generated raw outputs."""
    required_columns = {
        "generation_prompt_id",
        "prompt_id",
        "domain",
        "topic",
        "prompt_type",
        "input_format",
        "ideological_axis",
        "prompt_text",
        "model_family",
        "model_type",
        "model_name",
        "response_text",
    }
    missing = required_columns.difference(raw_outputs.columns)
    if missing:
        raise ValueError(f"Missing required raw output columns: {sorted(missing)}")

    expected_rows = expected_prompt_cases * 2
    if len(raw_outputs) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(raw_outputs)}.")

    if set(raw_outputs["model_type"]) != {"base", "instruct"}:
        raise ValueError("Expected model_type values to be exactly {'base', 'instruct'}.")

    if set(raw_outputs["input_format"]) != {"completion", "instruction"}:
        raise ValueError("Expected input_format values to be exactly {'completion', 'instruction'}.")

    if not raw_outputs["generation_prompt_id"].is_unique:
        raise ValueError("generation_prompt_id values must be unique.")

    if not raw_outputs.groupby("prompt_id")["model_type"].nunique().eq(2).all():
        raise ValueError("Every prompt_id must have both base and instruct outputs.")

    if raw_outputs["response_text"].isna().any():
        raise ValueError("response_text contains missing values.")
