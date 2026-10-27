"""NSL-KDD: download, contaminated training sets and preprocessing.

Protocol (paper, Sec. 5.1): the training set holds every normal record of
KDDTrain+ plus a random sample of its attacks, so that attacks are a fraction
c in {0, 5, 10, 15}% of the training set; the test set is the full KDDTest+.

Two choices differ from the course code and are deliberate:

* the min-max scaler is fitted on the contaminated training set actually used
  for training, not on all of KDDTrain+: fitted on everything, the feature
  ranges of attacks the model never sees would leak into the preprocessing;
* the one-hot vocabulary comes from the whole KDDTrain+ file. It is a schema
  (which protocols, services and flags exist), not a statistic, and it keeps
  the input at 122 dimensions for every contamination level, as in the paper.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Mirror of the official files (University of New Brunswick); checksums match the originals.
_MIRROR = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]  # fmt: skip
CATEGORICAL = ["protocol_type", "service", "flag"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: str | Path) -> Path:
    """Download KDDTrain+ and KDDTest+ into `raw_dir` (skipped if already there) and verify them."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = raw_dir / name
        if not path.exists():
            url = _MIRROR + name.replace("+", "%2B")
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        if _sha256(path) != digest:
            raise ValueError(f"{path} does not match the expected SHA-256 checksum")
    return raw_dir


def load(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and binary labels (1 = attack) of an NSL-KDD file."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    y = (df["label"] != "normal").to_numpy(dtype=np.int64)
    return df.drop(columns=["label", "difficulty"]), y


def contaminate(
    X: pd.DataFrame, y: np.ndarray, contamination: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """All normal records plus random attacks, attacks being `contamination` of the result."""
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must be in [0, 1)")
    normal, attacks = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    n_attacks = min(round(contamination / (1.0 - contamination) * normal.size), attacks.size)
    keep = np.concatenate([normal, rng.choice(attacks, size=n_attacks, replace=False)])
    keep = rng.permutation(keep)
    return X.iloc[keep].reset_index(drop=True), y[keep]


class Preprocessor:
    """Min-max scaling of numeric features and one-hot encoding of categorical ones.

    Test values outside the training range are not clipped: an out-of-range
    value is itself evidence of an anomaly, and clipping would hide it.
    """

    def __init__(self, vocabulary: pd.DataFrame):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(vocabulary[CATEGORICAL])
        self.scaler = MinMaxScaler()
        self.numeric: list[str] = [c for c in vocabulary.columns if c not in CATEGORICAL]

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        self.scaler.fit(X[self.numeric].to_numpy(dtype=np.float64))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        numeric = self.scaler.transform(X[self.numeric].to_numpy(dtype=np.float64))
        onehot = self.encoder.transform(X[CATEGORICAL])
        return np.hstack([numeric, onehot]).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + list(self.encoder.get_feature_names_out(CATEGORICAL))


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray  # used only for diagnostics, never for training
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def prepare(raw_dir: str | Path, contamination: float, seed: int) -> Dataset:
    """Contaminated training set and test set, preprocessed (fit on the training set only)."""
    raw_dir = Path(raw_dir)
    X_full, y_full = load(raw_dir / "KDDTrain+.txt")
    X_test, y_test = load(raw_dir / "KDDTest+.txt")
    X_train, y_train = contaminate(X_full, y_full, contamination, np.random.default_rng(seed))
    pre = Preprocessor(vocabulary=X_full).fit(X_train)
    return Dataset(pre.transform(X_train), y_train, pre.transform(X_test), y_test, pre.feature_names)
