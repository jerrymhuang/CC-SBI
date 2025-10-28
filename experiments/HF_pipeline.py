import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
sys.path.append("src")

import keras
import logging
import argparse
import bayesflow as bf
import matplotlib.pyplot as plt

from pathlib import Path
from simulations.benchmarks.hf import hf
from simulations.molecules import MoleculeSimulator
from utils.dataset_utils import generate_dataset, verify_dataset, load_npz_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments for the HF training pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with training parameters.
    """
    parser = argparse.ArgumentParser(description="HF training pipeline for BayesFlow")
    parser.add_argument("--train-samples", type=int, default=10, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=2, help="Number of validation samples")
    parser.add_argument("--num-molecules", type=int, default=1, help="Number of molecules per simulation")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory for model")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Output directory for diagnostic figures")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--basis", type=str, default="cc-pVTZ", help="Basis fucntion for simulation")
    return parser.parse_args()

def main():
    """
    H2O training pipeline for BayesFlow.

    Generates or loads training and validation datasets for water molecule chains,
    trains a DiffusionModel using BayesFlow, and saves the trained model.
    """
    args = parse_args()

    # Ensure output directories exist
    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)


    # Define simulator
    simulator = MoleculeSimulator(
        molecule_fun=hf,
        basis=args.basis,
        coord_scale=0.1,
    )

    # Define adapter
    adapter = (
        bf.adapters.Adapter()
        .drop(["atoms", "kinetic_energy", "occupancies", "determinant", "hf_energy"])
        .convert_dtype("float64", "float32")
        .concatenate(
            ["overlaps", "nuc_attraction"],
            into="inference_conditions"
        )
        .concatenate(["t1"], into="inference_variables")
    )

    # Define networks
    dm = bf.networks.DiffusionModel()

    # Set up workflow
    dm_workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=dm,
        checkpoint_filepath=checkpoint_dir / "hf_diffusion.ckpt",
    )

    include_kwargs = {
        "include_all": False,
        "include_integrals": True,
        "include_hartree_fock": True,
        "include_cc": True,
        "include_coordinates": False
    }

    # Generate and verify datasets, train, and visualize diagnostics
    try:
        train_set = generate_dataset(
            simulator=simulator,
            batch_size=args.train_samples,
            num_molecules=args.num_molecules,
            include_kwargs=include_kwargs,
            out_path=out_dir / f"hf_{args.num_molecules}_{args.basis}_train.npz"
        )
        val_set = generate_dataset(
            simulator=simulator,
            batch_size=args.val_samples,
            num_molecules=args.num_molecules,
            include_kwargs=include_kwargs,
            out_path=out_dir / f"hf_{args.num_molecules}_{args.basis}_val.npz"
        )
        logging.info("Verifying dataset structure...")
        train_data = load_npz_dict(out_dir / f"hf_{args.num_molecules}_{args.basis}_train.npz")
        verify_dataset(train_data)

        # Check batch size
        if train_data["nuc_attraction"].shape[0] != args.train_samples:
            logging.warning(f"Expected {args.train_samples} train samples, got {train_data['nuc_attraction'].shape[0]}")

        # Train offline
        logging.info("Starting offline training...")
        history = dm_workflow.fit_offline(
            data=train_data,
            val_data=val_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        logging.info("Training completed.")

        # Test reloading checkpoint (This step is optional)
        logging.info("Testing model reload...")
        model_path = checkpoint_dir / "hf_diffusion.ckpt" / "model.keras"
        reloaded_model = keras.saving.load_model(model_path)
        logging.info(f"Model reloaded successfully from {model_path}")

        # Generate and save diagnostics
        logging.info("Generating diagnostics...")
        fig_size = (18, 72)
        legend_fontsize = 6
        label_fontsize = 10
        figures = dm_workflow.plot_default_diagnostics(
            test_data=val_set,
            loss_kwargs={"figsize": (15, 3), "label_fontsize": label_fontsize},
            recovery_kwargs={"figsize": fig_size, "label_fontsize": label_fontsize},
            calibration_ecdf_kwargs={
                "figsize": fig_size,
                "legend_fontsize": legend_fontsize,
                "difference": True,
                "label_fontsize": label_fontsize
            },
            z_score_contraction_kwargs={"figsize": fig_size, "label_fontsize": label_fontsize}
        )
        for plot_name, fig in figures.items():
            fig_path = figures_dir / f"hf_{args.num_molecules}_{args.basis}_{plot_name}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logging.info(f"Saved diagnostic plot to {fig_path}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
