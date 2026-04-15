import argparse
import json
import math
import pathlib
from typing import Optional

import numpy as np
import scipy  # type: ignore


def sigdig(value, CI):
    def num_lead_zeros(x):
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = n_lead_zeros_CI + CI_sigdigs
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        return str(f"{round(value, decimals - 1):.{decimals - 1}f}"), str(
            f"{round(CI, decimals - 1):.{decimals - 1}f}"
        )
    else:
        return str(f"{rounded_value:.{decimals}f}"), str(f"{rounded_CI:.{decimals}f}")


# tests to ensure sigdig behavior
value = 0.084011111
CI = 0.0010011111
assert sigdig(value, CI) == ("0.0840", "0.0010")

value2 = 0.083999999
CI2 = 0.0009999999
assert sigdig(value2, CI2) == ("0.0840", "0.0010")


def confidence_interval(values, sizes):
    """99% bootstrap CI half-width for weighted mean."""
    if len(values) <= 1:
        return 0.0

    identifiers = [i for i in range(len(values))]
    dict_x_w = {
        identifier: (value, weight)
        for identifier, (value, weight) in enumerate(zip(values, sizes))
    }

    def weighted_mean(z, axis):
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    try:
        ci = scipy.stats.bootstrap(
            (identifiers,),
            statistic=weighted_mean,
            confidence_level=0.99,
            axis=0,
            method="BCa",
            random_state=42,
        )
        low = float(ci.confidence_interval.low)
        high = float(ci.confidence_interval.high)
        return (high - low) / 2
    except Exception:
        return 0.0


def weighted_avg_and_std(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    average = np.average(values, weights=weights)
    n_eff = np.square(np.sum(weights)) / np.sum(np.square(weights))
    if n_eff <= 1:
        return float(average), 0.0

    variance = np.average((values - average) ** 2, weights=weights) * (
        n_eff / (n_eff - 1)
    )
    return float(average), float(np.sqrt(max(variance, 0.0)))


def parse_models_arg(models_arg: str) -> list[str]:
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def load_results_for_model(model: str) -> dict[int, dict]:
    """
    Returns mapping user_id -> result dict.
    Fails on duplicate user entries in the same file.
    """
    result_file = pathlib.Path(f"./result/{model}.jsonl")
    if not result_file.exists():
        return {}

    with open(result_file, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f.readlines()]

    user_map: dict[int, dict] = {}
    for r in rows:
        if "user" not in r:
            continue
        u = int(r["user"])
        if u in user_map:
            raise ValueError(f"Duplicate user {u} in {result_file}")
        user_map[u] = r
    return user_map


def extract_param_vectors(result: dict) -> list[list[float]]:
    out: list[list[float]] = []
    if "parameters" not in result:
        return out

    p = result["parameters"]
    if isinstance(p, list):
        out.append(p)
    elif isinstance(p, dict):
        for v in p.values():
            if isinstance(v, list):
                out.append(v)
    return out


def common_users_across_models(model_to_user_map: dict[str, dict[int, dict]]) -> set[int]:
    if not model_to_user_map:
        return set()
    sets = [set(user_map.keys()) for user_map in model_to_user_map.values()]
    common = sets[0].copy()
    for s in sets[1:]:
        common &= s
    return common


def sanity_check_sizes(
    model_to_user_map: dict[str, dict[int, dict]],
    common_users: set[int],
) -> tuple[int, dict[int, int]]:
    """
    Ensures for every common user, result['size'] is identical across all model files.
    Returns:
      total_common_reviews, size_by_user
    """
    models = list(model_to_user_map.keys())
    size_by_user: dict[int, int] = {}
    mismatches = []

    for u in sorted(common_users):
        sizes = []
        for m in models:
            r = model_to_user_map[m][u]
            if "size" not in r:
                sizes.append(None)
            else:
                sizes.append(int(r["size"]))

        # Check missing/None
        if any(s is None for s in sizes):
            mismatches.append((u, sizes))
            continue

        if len(set(sizes)) != 1:
            mismatches.append((u, sizes))
            continue

        size_by_user[u] = sizes[0]

    if mismatches:
        print("Sanity check failed: 'size' mismatch across files for some common users.")
        print("First mismatches (user -> sizes in model order):")
        for u, sz in mismatches[:20]:
            print(f"  {u} -> {sz}")
        print("Model order:", models)
        raise SystemExit(1)

    total_common_reviews = int(sum(size_by_user.values()))
    return total_common_reviews, size_by_user


def summarize_model(
    model: str,
    user_map: dict[int, dict],
    metric_key: str,
    common_users_sorted: list[int],
    common_sizes: np.ndarray,
):
    metrics = []
    params = []

    for u in common_users_sorted:
        result = user_map[u]
        m = result.get("metrics", {}).get(metric_key, None)
        if m is None:
            return None
        metrics.append(float(m))
        params.extend(extract_param_vectors(result))

    metrics_arr = np.array(metrics, dtype=np.float64)
    return {
        "model": model,
        "metrics": metrics_arr,
        "sizes": common_sizes,  # same per model after sanity check
        "params": params,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--models",
        type=str,
        default="FSRS-7-4class-method1,FSRS-7-4class-method2",
        help="Comma-separated model result names (without .jsonl)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="CrossEntropy4",
        help="Metric key inside result['metrics']",
    )
    args = parser.parse_args()

    models = parse_models_arg(args.models)
    if not models:
        raise ValueError("No models provided.")

    model_to_user_map: dict[str, dict[int, dict]] = {}
    for model in models:
        user_map = load_results_for_model(model)
        if len(user_map) == 0:
            print(f"Skipping {model}: result file missing or empty.")
            continue
        model_to_user_map[model] = user_map

    if len(model_to_user_map) == 0:
        print("No valid result files found.")
        raise SystemExit(0)

    common_users = common_users_across_models(model_to_user_map)
    if len(common_users) == 0:
        print("No common users across selected model files.")
        raise SystemExit(0)

    total_common_reviews, size_by_user = sanity_check_sizes(model_to_user_map, common_users)
    common_users_sorted = sorted(common_users)
    common_sizes = np.array([size_by_user[u] for u in common_users_sorted], dtype=np.float64)

    print(f"Common users across models: {len(common_users_sorted)}")
    print(f"Common reviews across models: {total_common_reviews}")

    summaries = []
    for model, user_map in model_to_user_map.items():
        s = summarize_model(
            model=model,
            user_map=user_map,
            metric_key=args.metric,
            common_users_sorted=common_users_sorted,
            common_sizes=common_sizes,
        )
        if s is not None:
            summaries.append(s)

    if len(summaries) == 0:
        print("No valid metrics found for common users.")
        raise SystemExit(0)

    if args.fast:
        # Sort by review-weighted metric descending
        scored = []
        for s in summaries:
            mean_reviews, std_reviews = weighted_avg_and_std(s["metrics"], s["sizes"])
            scored.append((mean_reviews, std_reviews, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        for mean_reviews, std_reviews, s in scored:
            model = s["model"]
            metrics = s["metrics"]
            sizes = s["sizes"]
            params = s["params"]

            mean_users, std_users = weighted_avg_and_std(metrics, np.ones_like(sizes))

            print(f"Model: {model}")
            print(
                f"Weighted by reviews -> {args.metric} (mean±std): "
                f"{mean_reviews:.6f}±{std_reviews:.6f}"
            )
            print(
                f"Weighted by users   -> {args.metric} (mean±std): "
                f"{mean_users:.6f}±{std_users:.6f}"
            )

            if len(params) > 0:
                print(f"parameters (median): {np.median(params, axis=0).round(6).tolist()}")
            print()
    else:
        for scale in ("users", "reviews"):
            print(f"Weighted by number of {scale}")
            print(f"| Model | {args.metric} |")
            print("| --- | --- |")

            rows = []
            for s in summaries:
                model = s["model"]
                metrics = s["metrics"]
                sizes = s["sizes"]

                size_base = sizes if scale == "reviews" else np.ones_like(sizes)
                wmean, _wstd = weighted_avg_and_std(metrics, size_base)
                CI = confidence_interval(metrics, size_base)
                rounded_mean, rounded_CI = sigdig(wmean, CI)
                rows.append((wmean, f"| {model} | {rounded_mean}±{rounded_CI} |"))

            # 3) sort ascending by CrossEntropy4
            rows.sort(key=lambda x: x[0], reverse=False)

            for _, line in rows:
                print(line)
            print()