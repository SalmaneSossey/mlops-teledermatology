import argparse
import tempfile
import unittest
from pathlib import Path

from src.training.run_hparam_sweep import (
    select_trials,
    trial_output_dir,
    trial_to_training_config,
    write_sweep_results,
)


class RunHparamSweepTest(unittest.TestCase):
    def make_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            images_dir=Path("/images"),
            splits_dir=Path("/splits"),
            output_dir=Path("/runs"),
            epochs=2,
            batch_size=8,
            image_size=224,
            seed=42,
            num_workers=0,
            experiment_name="experiment",
            tracking_uri=None,
            hf_dataset_repo="dataset/repo",
            allow_cpu=True,
            focal_gamma=2.0,
        )

    def test_select_trials_caps_default_plan(self):
        trials = select_trials(3)

        self.assertEqual([trial.name for trial in trials], ["baseline", "lr_1e-4", "lr_1e-3"])

    def test_trial_to_training_config_maps_sweep_knobs(self):
        trial = select_trials(6)[-1]
        config = trial_to_training_config(trial, self.make_args())

        self.assertEqual(config.output_dir, Path("/runs/hparam_sweep/focal_loss"))
        self.assertEqual(config.loss_type, "focal_loss")
        self.assertFalse(config.require_gpu)
        self.assertEqual(config.hf_dataset_repo, "dataset/repo")

    def test_write_sweep_results_sorts_by_selection_score(self):
        rows = [
            {"name": "low", "selection_score": 0.1},
            {"name": "high", "selection_score": 0.9},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = write_sweep_results(rows, Path(tmpdir))

            self.assertTrue((Path(tmpdir) / "hparam_sweep_results.csv").exists())
            self.assertEqual(frame.iloc[0]["name"], "high")

    def test_trial_output_dir_is_stable(self):
        trial = select_trials(1)[0]

        self.assertEqual(trial_output_dir(Path("/runs"), trial), Path("/runs/hparam_sweep/baseline"))


if __name__ == "__main__":
    unittest.main()
