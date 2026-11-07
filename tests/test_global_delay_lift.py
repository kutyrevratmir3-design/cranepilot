import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import linalg

from scripts.global_delay_lift import (
    FEATURE_DIMS, features, fit_path, load_model, predict_residual, save_model,
)


def packet(n=36, seed=73):
    """Independent synthetic past/future-command arrays, no measured future."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, .02, (n, 41))
    x = rng.normal(0, .005, (n, 41))
    u = rng.normal(0, .3, (n, 41))
    relative = x - x[:, :1]
    t = -.01 * np.arange(21)
    velocity = x[:, :21] @ ((t - t.mean()) / np.sum((t - t.mean())**2))
    raw = {
        "angular_context": np.column_stack((theta / .1, relative / .1,
                                            u / 2., np.zeros((n, 2)))),
        "actuator_context": np.column_stack((relative / .1, u / 2.)),
        "omega": rng.uniform(5., 10., n),
        "initial_velocity": velocity,
        "initial_x": x[:, 0],
        "future_u": rng.normal(0, .3, (n, 30)),
    }
    return raw, rng.normal(0, .2, (n, 30))


class GlobalDelayLiftTests(unittest.TestCase):
    def test_shapes_target_poisoning_and_translation(self):
        raw, acc = packet()
        poisoned = {**raw, "target_q": np.full((36, 30), np.nan),
                    "target_x": np.full((36, 30), np.inf),
                    "harmonic": np.full((36, 7), np.nan),
                    "initial_q": np.full(36, np.nan),
                    "initial_p": np.full(36, np.nan),
                    "initial_x": raw["initial_x"] + 1000.}
        poisoned["angular_context"] = raw["angular_context"].copy()
        poisoned["angular_context"][:, -2:] = np.nan
        shifted = {**raw, "actuator_context": raw["actuator_context"].copy()}
        shifted["actuator_context"][:, :41] += 17.
        for basis, dim in FEATURE_DIMS.items():
            for mode in ("actor", "command"):
                with self.subTest(basis=basis, mode=mode):
                    x = features(raw, acc, mode, basis)
                    self.assertEqual(x.shape, (36, dim))
                    np.testing.assert_array_equal(x, features(poisoned, acc, mode, basis))
                    np.testing.assert_allclose(x, features(shifted, acc, mode, basis),
                                               atol=2e-13, rtol=2e-12)

    def test_modes_use_only_selected_future(self):
        raw, acc = packet()
        changed = {**raw, "future_u": raw["future_u"] + 7.}
        np.testing.assert_array_equal(features(raw, acc), features(changed, acc))
        np.testing.assert_array_equal(features(raw, None, "command"),
                                      features(raw, acc * 700., "command"))
        self.assertGreater(np.linalg.norm(features(raw, acc) - features(raw, acc + 1.)), 1.)
        self.assertGreater(np.linalg.norm(features(raw, None, "command") -
                                          features(changed, None, "command")), 1.)

    def test_zero_response_has_no_length_bias(self):
        raw, acc = packet()
        for key in ("angular_context", "actuator_context", "initial_velocity", "future_u"):
            raw[key] = np.zeros_like(raw[key])
        for basis in FEATURE_DIMS:
            np.testing.assert_array_equal(features(raw, np.zeros_like(acc), basis=basis), 0.)

    def test_ridge_path_uses_one_svd_and_exact_weighted_objective(self):
        raw, acc = packet(50)
        rng = np.random.default_rng(5)
        y = rng.normal(0, .01, (50, 30))
        weights = np.arange(50, dtype=float)
        alphas = (1e-4, 1e-2, 1.)
        with patch("scripts.global_delay_lift.linalg.svd", wraps=linalg.svd) as svd:
            models = fit_path(raw, acc, y, alphas, weights=weights)
        self.assertEqual(svd.call_count, 1)
        self.assertEqual(set(models), set(alphas))
        x = features(raw, acc)
        w = weights / weights.sum()
        for alpha, model in models.items():
            scaled = x / model.scale
            gram = scaled.T @ (w[:, None] * scaled) + 30. * alpha * np.eye(x.shape[1])
            rhs = scaled.T @ (w[:, None] * y)
            expected = np.linalg.solve(gram, rhs)
            np.testing.assert_allclose(model.coef, expected, atol=2e-12, rtol=2e-10)
            self.assertEqual(predict_residual(model, raw, acc).shape, (50, 30))

    def test_linear_synthetic_recovery_on_new_rows(self):
        train, acc = packet(300, seed=4)
        heldout, heldout_acc = packet(45, seed=19)
        x = features(train, acc)
        truth = np.zeros((x.shape[1], 30))
        truth[[0, 4, 18, 27, 30, 60, 90]] = np.random.default_rng(91).normal(0, .01, (7, 30))
        model = fit_path(train, acc, x @ truth, [1e-12])[1e-12]
        scale_before = model.scale.copy()
        prediction = predict_residual(model, heldout, heldout_acc)
        np.testing.assert_allclose(prediction, features(heldout, heldout_acc) @ truth,
                                   atol=1e-7, rtol=1e-5)
        np.testing.assert_array_equal(model.scale, scale_before)

    def test_save_load_and_training_scales_are_retained(self):
        raw, acc = packet(24)
        y = np.random.default_rng(37).normal(0, .02, (24, 30))
        with tempfile.TemporaryDirectory() as directory:
            for basis in FEATURE_DIMS:
                for mode in ("actor", "command"):
                    model = fit_path(raw, acc, y, [.01], basis=basis, input_mode=mode)[.01]
                    path = Path(directory) / f"{basis}_{mode}.npz"
                    save_model(model, path)
                    restored = load_model(path)
                    np.testing.assert_array_equal(predict_residual(model, raw, acc),
                                                  predict_residual(restored, raw, acc))
                    self.assertEqual(model.metadata, restored.metadata)
                    with self.assertRaises(FileExistsError):
                        save_model(model, path)

    def test_invalid_packets_and_fit_arguments_are_rejected(self):
        raw, acc = packet(4)
        y = np.zeros((4, 30))
        with self.assertRaises(ValueError):
            features({**raw, "omega": -raw["omega"]}, acc)
        with self.assertRaises(ValueError):
            features(raw, np.zeros((4, 31)))
        with self.assertRaises(ValueError):
            fit_path(raw, acc, np.zeros((4, 7)))
        with self.assertRaises(ValueError):
            fit_path(raw, acc, y, [0.])
        with self.assertRaises(ValueError):
            fit_path(raw, acc, y, weights=np.zeros(4))
        with self.assertRaises(ValueError):
            fit_path(raw, acc, y, weights=-np.ones(4))


if __name__ == "__main__":
    unittest.main()
