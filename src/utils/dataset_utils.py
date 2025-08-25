import numpy as np

from pathlib import Path


def save_npz_dict(d, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # flatten first level of dict into a single compressed NPZ
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in d.items()})


def load_npz_dict(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
