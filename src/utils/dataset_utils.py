import logging
import numpy as np
import bayesflow as bf

from pathlib import Path
from typing import Dict, Union
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_npz_dict(d: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """Save a dictionary to a compressed .npz file.

    Parameters
    ----------
    d : Dict[str, np.ndarray]
        Dictionary with numpy arrays to save.
    path : Union[str, Path]
        Path to save the .npz file.

    Notes
    -----
    Creates parent directories if they do not exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: np.asarray(v) for k, v in d.items()})

def load_npz_dict(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load a dictionary from a compressed .npz file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the .npz file.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with loaded arrays.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    """
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def generate_dataset(
    simulator: MoleculeSimulator,
    batch_size: int,
    num_molecules: int,
    out_path: Union[str, Path]
) -> Dict[str, np.ndarray]:
    """Generate or load a dataset of molecular simulations.

    If a dataset exists at `out_path` with the correct batch size, it is loaded.
    Otherwise, a new dataset is generated using the simulator and saved.

    Parameters
    ----------
    simulator : MoleculeSimulator
        Simulator for generating molecular data.
    adapter : bayesflow.adapters.Adapter
        Adapter for transforming simulation outputs (unused in this version).
    batch_size : int
        Number of samples to generate or load.
    num_molecules : int
        Number of molecules per simulation.
    out_path : Union[str, Path]
        Path to save or load the dataset.

    Returns
    -------
    Dict[str, np.ndarray]
        Dataset with keys like 'nuc_potential', 'overlap', 'coordinates', 't1'.

    Raises
    ------
    FileNotFoundError
        If saving the dataset fails due to invalid path.
    ValueError
        If the loaded dataset has incorrect batch size or missing keys.
    """
    out_path = Path(out_path)
    if out_path.exists():
        logging.info(f"Found existing dataset at {out_path}")
        try:
            data = load_npz_dict(out_path)
            if data["nuc_potential"].shape[0] == batch_size:
                logging.info(f"Loaded dataset with {batch_size} samples")
                logging.info(f"Dataset keys: {list(data.keys())}")
                logging.info(f"Dataset shapes: {[(k, data[k].shape) for k in data.keys()]}")
                return data
            else:
                logging.warning(f"Existing dataset has {data['nuc_potential'].shape[0]} samples, expected {batch_size}. Regenerating...")
        except Exception as e:
            logging.warning(f"Failed to load dataset: {e}. Regenerating...")

    logging.info(f"Generating dataset ({batch_size} samples, {num_molecules} molecules)...")
    try:
        data = simulator.sample(batch_size=batch_size, num_molecules=num_molecules, show_progress=True)
        logging.info(f"Dataset keys: {list(data.keys())}")
        logging.info(f"Dataset shapes: {[(k, data[k].shape) for k in data.keys()]}")
        save_npz_dict(data, out_path)
        logging.info(f"Dataset saved to {out_path}")
        return data
    except Exception as e:
        logging.error(f"Dataset generation failed: {e}")
        raise

def verify_dataset(
    data: Dict[str, np.ndarray],
    expected_keys: Sequence[str] = ["nuc_potential", "overlap", "coordinates", "t1"]
) -> None:
    """Verify dataset structure and shapes.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dataset to verify.
    expected_keys : List[str], optional
        Expected keys in the dataset (default: ["nuc_potential", "overlap", "coordinates", "t1"]).

    Raises
    ------
    ValueError
        If keys are missing or shapes are inconsistent.
    """
    if not all(k in data for k in expected_keys):
        raise ValueError(f"Dataset missing keys: {set(expected_keys) - set(data.keys())}")
    shapes = [data[k].shape[0] for k in expected_keys]
    if len(set(shapes)) > 1:
        raise ValueError(f"Inconsistent batch sizes: {shapes}")
    for key, value in data.items():
        logging.info(f"  {key}: shape {value.shape}")
