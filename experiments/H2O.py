import os
import sys
os.environ["KERAS_BACKEND"] = "jax"
sys.path.append('../src')

import tqdm
import logging
import numpy as np
import bayesflow as bf

from pathlib import Path
from simulations.molecules import *
from utils.dataset_utils import *
from utils.simulator_utils import *

np.set_printoptions(suppress=True)


def h2o_ccsd():

    h2o_molecules = make_molecules(species="H2O", N=2)
    ccsd = compute_ccsd(h2o_molecules["species"], h2o_molecules["pos"])

    return ccsd


if __name__ == "__main__":

    # define simulator
    simulator = bf.make_simulator([h2o_ccsd])

    # define adapter
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate([
            'V_mat',
            'atoms_flat',
            # 's_flat'
        ], into="inference_conditions")
        .concatenate(['T1_flat'], into="inference_variables")
    )

    # define inference network
    dm = bf.networks.DiffusionModel()

    # define workflow
    dm_workflow = bf.workflows.BasicWorkflow(
        adapter = adapter,
        inference_network=dm,
        checkpoint_filepath="./checkpoints/h2o_diffusion.ckpt",
    )

    # define path for storing generated dataset
    out_dir = Path("data")
    train_path = out_dir / "h2o_train.npz"
    val_path = out_dir / "h2o_val.npz"

    # simulate dataset
    train_set = simulate_with_tqdm(dm_workflow, n=50)
    logging.info("Done.")
