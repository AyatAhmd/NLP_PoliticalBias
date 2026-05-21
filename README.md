# NLP Final Project: Instruction Tuning and Political Bias

This project studies how instruction tuning changes the behaviour of a language model on political prompts.

The experiment compares two versions of the same model family:

- Base model: `Qwen/Qwen2.5-0.5B`
- Instruction-tuned model: `Qwen/Qwen2.5-0.5B-Instruct`

Both models are given the same set of political prompts. Their responses are scored and compared using a paired treatment-effect design.

## Structure

```text
src/
  config.py          Shared constants and paths
  dataset.py         Prompt dataset construction
  generation.py      Model loading and response generation
  scoring.py         Rule-based response scoring
  statistics.py      Paired treatment-effect analysis
  visualization.py   Plotting functions

notebooks/
  01_generate_dataset.ipynb
  02_generate_outputs.ipynb
  03_score_outputs.ipynb
  04_analysis.ipynb

data/
  prompts_200.csv
  raw_outputs.csv
  scored_outputs.csv

results/
  tables/
  figures/
```

## Pipeline

1. `01_generate_dataset.ipynb` creates `data/prompts_200.csv`.
2. `02_generate_outputs.ipynb` runs both models and creates `data/raw_outputs.csv`.
3. `03_score_outputs.ipynb` scores responses and creates `data/scored_outputs.csv`.
4. `04_analysis.ipynb` computes treatment effects, tables, plots, and qualitative examples.

## Research Question

What is the total causal effect of instruction tuning on political neutrality, ideological leaning, cautiousness, and refusal behaviour in LLM responses?

## Main Finding

Instruction tuning increased cautious language and refusal/avoidance behaviour more clearly than it changed political leaning or neutrality.

## AI Usage

Parts of this project were developed with assistance from ChatGPT. The final code, results, and report were reviewed and edited by the student.
