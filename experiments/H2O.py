import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
sys.path.append("src")

import logging
import argparse
from pathlib import Path
import bayesflow as bf
import keras
from simulations.molecules import MoleculeSimulator
from simulations.benchmarks.water import water
from utils.dataset_utils import generate_dataset, verify_dataset, load_npz_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_args():
    """Parse command-line arguments for the H2O training pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with training parameters.
    """
    parser = argparse.ArgumentParser(description="H2O training pipeline for BayesFlow")
    parser.add_argument("--train-samples", type=int, default=10, help="Number of training samples")
    parser.add_argument("--val-samples", type=int, default=2, help="Number of validation samples")
    parser.add_argument("--num-molecules", type=int, default=2, help="Number of molecules per simulation")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory for datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory for model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    return parser.parse_args()

def main():
    """H2O training pipeline for BayesFlow.

    Generates or loads training and validation datasets for water molecule chains,
    trains a DiffusionModel using BayesFlow, and saves the trained model.
    """
    args = parse_args()

    # Ensure output directories exist
    out_dir = Path(args.out_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Define simulator
    simulator = MoleculeSimulator(
        species=water,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1
    )

    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(
            ["overlap", "nuc_potential"],
            into="inference_conditions"
        )
        .concatenate(["t1"], into="inference_variables")
    )

    dm = bf.networks.DiffusionModel()
    dm_workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=dm,
        checkpoint_filepath=checkpoint_dir / "h2o_test_diffusion.ckpt",
    )

    # Generate and verify datasets
    try:
        train_set = generate_dataset(
            simulator, args.train_samples, args.num_molecules, out_dir / "h2o_train.npz"
        )
        val_set = generate_dataset(
            simulator, args.val_samples, args.num_molecules, out_dir / "h2o_val.npz"
        )
        logging.info("Verifying dataset structure...")
        train_data = load_npz_dict(out_dir / "h2o_train.npz")
        verify_dataset(train_data)

        # Check batch size
        if train_data["nuc_potential"].shape[0] != args.train_samples:
            logging.warning(f"Expected {args.train_samples} train samples, got {train_data['nuc_potential'].shape[0]}")

        # Train offline
        logging.info("Starting offline training...")
        history = dm_workflow.fit_offline(
            data=train_data,
            val_data=val_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        logging.info("Training completed.")

        # Test reloading checkpoint
        logging.info("Testing model reload...")
        model_path = checkpoint_dir / "h2o_test_diffusion.ckpt" / "model.keras"
        reloaded_model = keras.saving.load_model(model_path)
        logging.info(f"Model reloaded successfully from {model_path}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()