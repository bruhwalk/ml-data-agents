"""Run full Active Learning experiment: LLM selects model, compare strategies.

Usage::

    python run_al.py
    python run_al.py data/labeled/labeled.parquet
    python run_al.py data/labeled/labeled.parquet 20 5
    python run_al.py data/labeled/labeled.parquet 20 5 --no-llm

LLM (Claude API) analyzes the dataset and chooses:
  - Which model to use (logreg, logreg_balanced, svm, nb)
  - Optimal seed size

Pass --no-llm to skip LLM selection and use defaults (logreg, seed=50).
"""

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd
from agents.active_learning_agent import ActiveLearningAgent, MODELS


def main(
    parquet_path: str = "data/labeled/labeled.parquet",
    batch_size: int = 20,
    n_iterations: int = 5,
    use_llm: bool = True,
) -> None:
    df = pd.read_parquet(ROOT / parquet_path)

    # Filter to rows with labels
    label_col = "auto_label"
    df_labeled = df[df[label_col].notna()].copy()

    # Read task type from config
    task_type = "text classification"
    try:
        with open(ROOT / "config.yaml") as f:
            config = yaml.safe_load(f)
        task_type = config.get("task", {}).get("type", task_type)
    except Exception:
        pass

    print(f"Dataset: {len(df_labeled)} labeled rows")
    print(f"Labels: {df_labeled[label_col].nunique()} classes")
    print(f"Task: {task_type}")
    print()

    # Step 1: LLM selects model and seed size
    if use_llm:
        print("Asking Claude API to select model and seed size...")
        agent = ActiveLearningAgent()  # temporary, will recreate with chosen model
        recommendation = agent.select_model(df_labeled, task_type=task_type)

        model_key = recommendation["model"]
        seed_size = recommendation["seed_size"]
        reasoning = recommendation.get("reasoning", "")

        print(f"  Model: {MODELS[model_key]['name']} ({model_key})")
        print(f"  Seed size: {seed_size}")
        print(f"  Reasoning: {reasoning}")
        print()
    else:
        model_key = "logreg"
        seed_size = 50
        recommendation = None
        print(f"Using defaults: model={model_key}, seed_size={seed_size}")
        print()

    # Create agent with chosen model
    agent = ActiveLearningAgent(model=model_key)
    if recommendation:
        agent._llm_recommendation = recommendation

    print(f"Seed size: {seed_size}")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {n_iterations}")
    print(f"Strategies: entropy, margin, random")
    print()

    # Step 2: Run all strategies
    strategies = ["entropy", "margin", "random"]
    print("Running AL experiments...")
    results = agent.compare_strategies(
        df_labeled,
        strategies=strategies,
        seed_size=seed_size,
        n_iterations=n_iterations,
        batch_size=batch_size,
    )

    # Print per-strategy results
    for strategy in strategies:
        history = results[strategy]
        final = history[-1]
        print(f"  {strategy}: accuracy={final['accuracy']:.4f}, F1={final['f1']:.4f}")

    # Step 3: Generate report and charts
    print("\nGenerating report and charts...")
    paths = agent.report(results)

    print(f"\nArtifacts saved:")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if recommendation:
        print(f"\nLLM chose: {MODELS[model_key]['name']} (seed={seed_size})")
        print(f"  Reason: {reasoning}")

    for strategy in strategies:
        h = results[strategy]
        print(f"\n{strategy}:")
        for step in h:
            print(f"  iter {step['iteration']}: n={step['n_labeled']}, "
                  f"acc={step['accuracy']:.4f}, f1={step['f1']:.4f}")

    # Savings
    if "random" in results:
        random_final = results["random"][-1]
        print(f"\nRandom baseline final F1: {random_final['f1']:.4f} "
              f"(at {random_final['n_labeled']} examples)")

        for strategy in strategies:
            if strategy == "random":
                continue
            for h in results[strategy]:
                if h["f1"] >= random_final["f1"]:
                    saved = random_final["n_labeled"] - h["n_labeled"]
                    pct = saved / random_final["n_labeled"] * 100
                    print(f"  {strategy} reaches this at {h['n_labeled']} "
                          f"examples (saves {saved}, {pct:.1f}%)")
                    break


if __name__ == "__main__":
    no_llm = "--no-llm" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-llm"]
    parquet = args[0] if len(args) > 0 else "data/labeled/labeled.parquet"
    batch = int(args[1]) if len(args) > 1 else 20
    iters = int(args[2]) if len(args) > 2 else 5
    main(parquet, batch, iters, use_llm=not no_llm)
