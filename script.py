"""
FSRS-7 minimalistic benchmark script with 4-class augmentation.

Usage:

    python script.py --data ../anki-revlogs-10k --processes 8 --four-class-method method1
    python script.py --data ../anki-revlogs-10k --processes 8 --four-class-method method2 --calibration-bins 20

Binary FSRS-7 is trained as-is (Again vs Pass).
Then a post-hoc layer maps recall probability R to 4-class probabilities:
Again / Hard / Good / Easy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from scipy.optimize import curve_fit  # type: ignore
from sklearn.metrics import log_loss  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from torch import Tensor, nn
from tqdm.auto import tqdm  # type: ignore

from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader  # type: ignore
from fsrs_v7 import FSRS7
from data import load_user_data

warnings.filterwarnings("ignore", category=UserWarning)


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass
class Config:
    # Data
    data_path: Path = Path("../anki-revlogs-10k")
    max_user_id: Optional[int] = None

    # Model / training
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 512
    n_splits: int = 5
    seed: int = 42
    default_params: bool = False       # skip training, use default weights
    use_recency_weighting: bool = False

    # FSRS-7-specific data flags (always on in this version)
    use_secs_intervals: bool = True
    include_short_term: bool = True
    max_seq_len: int = 64

    # Train / test split options
    train_equals_test: bool = False
    no_test_same_day: bool = False
    no_train_same_day: bool = False

    # 4-class layer (on top of FSRS recall R)
    four_class_method: str = "method1"   # "method1" | "method2"
    calibration_bins: int = 20           # used by method2
    eps: float = 1e-8

    # S0 limits
    s_min: float = 0.0001
    init_s_max: float = 100.0

    # Output
    save_evaluation_file: bool = False
    save_raw_output: bool = False
    generate_plots: bool = False
    save_weights: bool = False
    verbose_inadequate_data: bool = False

    # Parallelism
    num_processes: int = 1

    def get_evaluation_file_name(self) -> str:
        return f"FSRS-7-4class-{self.four_class_method}"


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_path=Path(args.data),
        max_user_id=args.max_user_id,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        seed=args.seed,
        default_params=args.default_params,
        use_recency_weighting=args.recency_weighting,
        four_class_method=args.four_class_method,
        calibration_bins=args.calibration_bins,
        save_evaluation_file=args.save_evaluation_file,
        save_raw_output=args.save_raw_output,
        save_weights=args.save_weights,
        verbose_inadequate_data=args.verbose,
        num_processes=args.processes,
        no_test_same_day=args.no_test_same_day,
        no_train_same_day=args.no_train_same_day,
    )


# ── trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(
        self,
        model: FSRS7,
        train_set: pd.DataFrame,
        batch_size: int = 512,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model
        self.batch_size = getattr(model, "batch_size", batch_size)
        self.betas = getattr(model, "betas", (0.9, 0.999))
        self.n_epoch = model.n_epoch
        self.loss_fn = nn.BCELoss(reduction="none")

        model.initialize_parameters(train_set)
        filtered = model.filter_training_data(train_set)

        self.train_dataset = BatchDataset(
            filtered.copy(), self.batch_size, max_seq_len=max_seq_len
        )
        self.train_loader = BatchLoader(self.train_dataset)

        self.optimizer = model.get_optimizer(
            lr=model.lr, wd=model.wd, betas=self.betas
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_loader.batch_nums * self.n_epoch,
        )

    def _batch_process(self, batch: tuple) -> dict[str, Tensor]:
        sequences, delta_ts, labels, seq_lens, weights = batch
        real_batch_size = seq_lens.shape[0]
        result = self.model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
        result["labels"] = labels
        result["weights"] = weights
        return result

    def train(self) -> list:
        best_loss = np.inf
        best_w = self.model.state_dict()
        epoch_len = len(self.train_dataset.y_train)

        for _ in range(self.n_epoch):
            loss_val, w = self._eval()
            if loss_val < best_loss:
                best_loss = loss_val
                best_w = w

            for batch in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                result = self._batch_process(batch)
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                self.model.apply_gradient_constraints()
                self.optimizer.step()
                self.scheduler.step()
                self.model.apply_parameter_clipper()

        loss_val, w = self._eval()
        if loss_val < best_loss:
            best_w = w
        return best_w

    def _eval(self) -> tuple[float, list]:
        self.model.eval()
        self.train_loader.shuffle = False
        total_loss = 0.0
        total_items = 0
        epoch_len = len(self.train_dataset.y_train)
        with torch.no_grad():
            for batch in self.train_loader:
                result = self._batch_process(batch)
                total_loss += (
                    (
                        self.loss_fn(result["retentions"], result["labels"])
                        * result["weights"]
                    )
                    .sum()
                    .item()
                )
                if "penalty" in result:
                    total_loss += (result["penalty"] / epoch_len).item()
                total_items += batch[3].shape[0]
        self.train_loader.shuffle = True
        w = self.model.state_dict()
        return total_loss / max(total_items, 1), w


# ── prediction helpers ────────────────────────────────────────────────────────


def batch_predict(
    model: FSRS7, dataset: pd.DataFrame, config: Config
) -> tuple[list, list, list]:
    """Run model over dataset and return (retentions, stabilities, difficulties)."""
    model.eval()
    ds = BatchDataset(dataset, batch_size=8192, sort_by_length=False)
    loader = BatchLoader(ds, shuffle=False)
    dev_loader = DevicePrefetchLoader(loader, target_device=config.device)

    retentions, stabilities, difficulties = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            sequences, delta_ts, labels, seq_lens, weights = batch
            real_batch_size = seq_lens.shape[0]
            result = model.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
            retentions.extend(result["retentions"].cpu().tolist())
            if "stabilities" in result:
                stabilities.extend(result["stabilities"].cpu().tolist())
            if "difficulties" in result:
                difficulties.extend(result["difficulties"].cpu().tolist())

    return retentions, stabilities, difficulties


# ── 4-class layer (on top of R) ──────────────────────────────────────────────


def _piecewise_linear_clamped(
    r: np.ndarray, p_min: float, r_at_min: float, p_max: float, r_at_max: float
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    p_min = float(np.clip(p_min, 0.0, 1.0))
    p_max = float(np.clip(p_max, 0.0, 1.0))
    r_at_min = float(np.clip(r_at_min, 0.0, 1.0))
    r_at_max = float(np.clip(r_at_max, 0.0, 1.0))

    if abs(r_at_max - r_at_min) < 1e-8:
        return np.full_like(r, (p_min + p_max) / 2.0)

    slope = (p_max - p_min) / (r_at_max - r_at_min)
    y = p_min + slope * (r - r_at_min)
    y = np.clip(y, min(p_min, p_max), max(p_min, p_max))
    return np.clip(y, 0.0, 1.0)


def _fit_method1(r_train: np.ndarray, rating_train: np.ndarray) -> dict:
    """Constant p(Hard|pass), p(Good|pass), p(Easy|pass) on training split."""
    pass_mask = rating_train > 1
    pass_ratings = rating_train[pass_mask]

    c_h = int(np.sum(pass_ratings == 2))
    c_g = int(np.sum(pass_ratings == 3))
    c_e = int(np.sum(pass_ratings == 4))
    total = c_h + c_g + c_e

    # Laplace smoothing for robustness on tiny splits.
    if total == 0:
        p_h, p_g, p_e = 1 / 3, 1 / 3, 1 / 3
    else:
        p_h = (c_h + 1) / (total + 3)
        p_g = (c_g + 1) / (total + 3)
        p_e = (c_e + 1) / (total + 3)

    s = p_h + p_g + p_e
    return {
        "method": "method1",
        "p_h": float(p_h / s),
        "p_g": float(p_g / s),
        "p_e": float(p_e / s),
    }


def _binned_pass_frequencies(
    r_train: np.ndarray, rating_train: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_bin_mean, hard_freq, easy_freq, counts) among pass-only samples."""
    pass_mask = rating_train > 1
    r = r_train[pass_mask]
    y = rating_train[pass_mask]

    if len(r) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(r, edges[1:-1], right=False)

    x_means, hard_freq, easy_freq, counts = [], [], [], []
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(np.sum(mask))
        if n == 0:
            continue

        rr = r[mask]
        yy = y[mask]
        x_means.append(float(np.mean(rr)))
        hard_freq.append(float(np.mean(yy == 2)))
        easy_freq.append(float(np.mean(yy == 4)))
        counts.append(n)

    return (
        np.array(x_means, dtype=float),
        np.array(hard_freq, dtype=float),
        np.array(easy_freq, dtype=float),
        np.array(counts, dtype=float),
    )


def _fit_method2(r_train: np.ndarray, rating_train: np.ndarray, n_bins: int) -> dict:
    """Fit 8 params with curve_fit over binned frequencies (Hard|pass, Easy|pass)."""
    x, h, e, cnt = _binned_pass_frequencies(r_train, rating_train, n_bins=n_bins)

    # Fallback if insufficient pass/bin data.
    if len(x) < 2:
        m1 = _fit_method1(r_train, rating_train)
        return {
            "method": "method2",
            "hard_params": (m1["p_h"], 0.0, m1["p_h"], 1.0),
            "easy_params": (m1["p_e"], 0.0, m1["p_e"], 1.0),
        }

    sigma = 1.0 / np.sqrt(np.maximum(cnt, 1.0))

    h0 = [
        float(np.min(h)),
        float(x[np.argmin(h)]),
        float(np.max(h)),
        float(x[np.argmax(h)]),
    ]
    e0 = [
        float(np.min(e)),
        float(x[np.argmin(e)]),
        float(np.max(e)),
        float(x[np.argmax(e)]),
    ]

    bounds = ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])

    try:
        hard_params, _ = curve_fit(
            _piecewise_linear_clamped,
            x,
            h,
            p0=h0,
            sigma=sigma,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        hard_params = np.array(h0, dtype=float)

    try:
        easy_params, _ = curve_fit(
            _piecewise_linear_clamped,
            x,
            e,
            p0=e0,
            sigma=sigma,
            bounds=bounds,
            maxfev=20000,
        )
    except Exception:
        easy_params = np.array(e0, dtype=float)

    return {
        "method": "method2",
        "hard_params": tuple(map(float, hard_params)),
        "easy_params": tuple(map(float, easy_params)),
    }


def _r_to_p4(r: np.ndarray, calib: dict, eps: float = 1e-8) -> np.ndarray:
    """
    Map FSRS recall probability r to 4-class probs:
      p(Again)=1-r
      p(Hard)=r*p(Hard|pass)
      p(Good)=r*p(Good|pass)
      p(Easy)=r*p(Easy|pass)
    """
    r = np.asarray(r, dtype=float)
    r = np.clip(r, 1e-4, 1 - 1e-4)

    if calib["method"] == "method1":
        p_h_cond = np.full_like(r, calib["p_h"], dtype=float)
        p_g_cond = np.full_like(r, calib["p_g"], dtype=float)
        p_e_cond = np.full_like(r, calib["p_e"], dtype=float)
    else:
        hp = calib["hard_params"]
        ep = calib["easy_params"]

        p_h_cond = _piecewise_linear_clamped(r, *hp)
        p_e_cond = _piecewise_linear_clamped(r, *ep)

        p_h_cond = np.clip(p_h_cond, 0.0, 1.0)
        p_e_cond = np.clip(p_e_cond, 0.0, 1.0)

        # enforce p_h + p_e <= 1, remainder goes to good
        sum_he = p_h_cond + p_e_cond
        over = sum_he > (1.0 - eps)
        if np.any(over):
            scale = (1.0 - eps) / sum_he[over]
            p_h_cond[over] *= scale
            p_e_cond[over] *= scale

        p_g_cond = 1.0 - p_h_cond - p_e_cond

    p_again = 1.0 - r
    p_hard = r * p_h_cond
    p_good = r * p_g_cond
    p_easy = r * p_e_cond

    p4 = np.stack([p_again, p_hard, p_good, p_easy], axis=1)
    p4 = np.clip(p4, eps, 1.0)
    p4 = p4 / p4.sum(axis=1, keepdims=True)
    return p4


# ── evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    y4: list[int],
    p4: np.ndarray,
    df: pd.DataFrame,
    user_id: int,
    config: Config,
    w_list: list[list[float]],
) -> tuple[dict, Optional[dict]]:
    ce4 = float(log_loss(y4, p4, labels=[0, 1, 2, 3]))

    stats: dict = {
        "metrics": {
            "CrossEntropy4": round(ce4, 6),
        },
        "user": int(user_id),
        "size": len(y4),
    }

    if w_list and isinstance(w_list[-1], list):
        stats["parameters"] = [round(float(x), 6) for x in w_list[-1]]
    elif config.save_weights and w_list:
        Path(f"weights/{config.get_evaluation_file_name()}").mkdir(
            parents=True, exist_ok=True
        )
        torch.save(w_list[-1], f"weights/{config.get_evaluation_file_name()}/{user_id}.pth")

    raw: Optional[dict] = None
    if config.save_raw_output:
        raw = {
            "user": int(user_id),
            "p4": [[round(float(v), 6) for v in row] for row in p4.tolist()],
            "y4": [int(v) for v in y4],
        }

    return stats, raw


def save_evaluation_file(user_id: int, df: pd.DataFrame, config: Config) -> None:
    if config.save_evaluation_file:
        df.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )


def sort_jsonl(file: Path) -> list:
    data = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return data


# ── per-user processing ───────────────────────────────────────────────────────


def _catch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            user_id = args[0] if args else kwargs.get("user_id")
            msg = traceback.format_exc()
            if user_id is not None:
                msg = f"User {user_id}:\n{msg}"
            return None, msg

    return wrapper


@_catch
def process(user_id: int, config: Config) -> tuple[dict, Optional[dict]]:
    dataset = load_user_data(user_id, config)

    w_list: list[list[float]] = []
    calibrators: list[dict] = []
    testsets: list[pd.DataFrame] = []

    tscv = TimeSeriesSplit(n_splits=config.n_splits)

    for train_index, test_index in tscv.split(dataset):
        if not config.train_equals_test:
            train_set = dataset.iloc[train_index].copy()
            test_set = dataset.iloc[test_index].copy()
        else:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[test_index]["review_th"].min()
            ].copy()

        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()

        testsets.append(test_set)

        try:
            train_local = train_set.copy()

            if not config.train_equals_test:
                assert train_local["review_th"].max() < test_set["review_th"].min()

            if config.use_recency_weighting:
                x = np.linspace(0, 1, len(train_local))
                train_local["weights"] = 0.25 + 0.75 * np.power(x, 3)

            model = FSRS7(config).to(config.device)

            if config.default_params:
                weights = model.state_dict()
            else:
                trainer = Trainer(
                    model=model,
                    train_set=train_local,
                    batch_size=config.batch_size,
                )
                weights = trainer.train()

        except Exception as e:
            if str(e).endswith("inadequate."):
                if config.verbose_inadequate_data:
                    print("Skipping - Inadequate data")
            else:
                print(f"User: {user_id}")
                raise
            weights = FSRS7(config).state_dict()

        # Fit 4-class layer on SAME train split (no future leakage)
        train_model = FSRS7(config, w=weights).to(config.device)
        r_train, _, _ = batch_predict(train_model, train_set, config)
        r_train_arr = np.array(r_train, dtype=float)
        rating_train_arr = train_set["rating"].to_numpy(dtype=int)

        if config.four_class_method == "method1":
            calib = _fit_method1(r_train_arr, rating_train_arr)
        else:
            calib = _fit_method2(
                r_train_arr, rating_train_arr, n_bins=config.calibration_bins
            )

        w_list.append(weights)
        calibrators.append(calib)

        if config.train_equals_test:
            break

    p4_all: list[np.ndarray] = []
    y4_all: list[int] = []
    save_tmp: list[pd.DataFrame] = []

    for weights, calib, testset in zip(w_list, calibrators, testsets):
        part_test = testset.copy()
        model = FSRS7(config, w=weights).to(config.device)
        retentions, stabilities, difficulties = batch_predict(model, part_test, config)

        p4 = _r_to_p4(np.array(retentions, dtype=float), calib=calib, eps=config.eps)

        part_test["p_again"] = p4[:, 0]
        part_test["p_hard"] = p4[:, 1]
        part_test["p_good"] = p4[:, 2]
        part_test["p_easy"] = p4[:, 3]

        if stabilities:
            part_test["s"] = stabilities
        if difficulties:
            part_test["d"] = difficulties

        p4_all.append(p4)
        y4_all.extend(part_test["y4"].tolist())
        save_tmp.append(part_test)

    save_df = pd.concat(save_tmp)
    if "tensor" in save_df.columns:
        del save_df["tensor"]
    save_evaluation_file(user_id, save_df, config)

    p4_mat = np.vstack(p4_all) if p4_all else np.zeros((0, 4), dtype=float)
    return evaluate(y4_all, p4_mat, save_df, user_id, config, w_list)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSRS-7 benchmark with 4-class augmentation")
    p.add_argument("--data", default="../anki-revlogs-10k", help="Path to dataset")
    p.add_argument("--processes", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--default-params",
        dest="default_params",
        action="store_true",
        help="Skip training, use default FSRS-7 parameters",
    )
    p.add_argument("--recency-weighting", dest="recency_weighting", action="store_true")
    p.add_argument("--no-test-same-day", dest="no_test_same_day", action="store_true")
    p.add_argument("--no-train-same-day", dest="no_train_same_day", action="store_true")

    p.add_argument(
        "--four-class-method",
        dest="four_class_method",
        choices=["method1", "method2"],
        default="method1",
        help="Method to map FSRS recall R -> Again/Hard/Good/Easy",
    )
    p.add_argument(
        "--calibration-bins",
        dest="calibration_bins",
        type=int,
        default=20,
        help="Number of bins used by method2",
    )

    p.add_argument("--save-evaluation-file", dest="save_evaluation_file", action="store_true")
    p.add_argument("--save-raw", dest="save_raw_output", action="store_true")
    p.add_argument("--save-weights", dest="save_weights", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max-user-id", dest="max_user_id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = _parse_args()
    config = build_config(args)
    torch.manual_seed(config.seed)

    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    file_name = config.get_evaluation_file_name()

    Path(f"evaluation/{file_name}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)

    result_file = Path(f"result/{file_name}.jsonl")
    raw_file = Path(f"raw/{file_name}.jsonl")

    processed_users: set[int] = set()
    if result_file.exists():
        processed_users = {d["user"] for d in sort_jsonl(result_file)}
    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    unprocessed = []
    for user_id in dataset.partitioning.dictionaries[0]:
        uid = user_id.as_py()
        if config.max_user_id is not None and uid > config.max_user_id:
            continue
        if uid not in processed_users:
            unprocessed.append(uid)
    unprocessed.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, uid, config) for uid in unprocessed]
        pbar = tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        for future in pbar:
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, raw = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    if raw:
                        with open(raw_file, "a", encoding="utf-8", newline="\n") as f:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    sort_jsonl(result_file)
    if config.save_raw_output:
        sort_jsonl(raw_file)


if __name__ == "__main__":
    main()