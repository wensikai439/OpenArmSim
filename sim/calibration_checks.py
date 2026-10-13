"""Calibration guards, temporal alignment and unchanged production parameters."""
import copy
import json
import unittest

import numpy as np

from .calibration import PROTOCOL, ControllerStudy, metrics, select_candidate, validate_protocol, report
from .environment import ROOT


class CalibrationChecks(unittest.TestCase):
    def test_split_rejects_same_episode_and_short_data(self):
        reference = json.loads((ROOT / "reference/real_trajectory.json").read_text())["episodes"]
        bad = {**PROTOCOL, "holdout_episodes": [0]}
        with self.assertRaises(ValueError):
            validate_protocol(bad, reference)
        bad = {**PROTOCOL, "frames": 10000}
        with self.assertRaises(ValueError):
            validate_protocol(bad, reference)

    def test_candidate_selection_cannot_optimize_holdout(self):
        a = {"fit_rmse_rad": .1, "holdout_rmse_rad": 99.}
        b = {"fit_rmse_rad": .2, "holdout_rmse_rad": .001}
        self.assertIs(select_candidate([b, a]), a)

    def test_fit_rollout_is_deterministic_and_does_not_change_profile(self):
        before = (ROOT / "sim/profile.json").read_bytes()
        study = ControllerStudy({**PROTOCOL, "frames": 20})
        try:
            a = study.rollout(0, [1., 1.], 0)
            study.rollout(0, [.5, 2.], 2)
            b = study.rollout(0, [1., 1.], 0)
            np.testing.assert_array_equal(a["prediction"], b["prediction"])
            np.testing.assert_array_equal(a["prediction"][0], a["target"][0])
            self.assertEqual((ROOT / "sim/profile.json").read_bytes(), before)
            self.assertEqual(a["robot_external_contact_steps"], 0)
        finally:
            study.close()

    def test_error_metric_is_time_aligned_not_difference_of_rms(self):
        target = np.array([[1.]*7, [-1.]*7])
        prediction = -target
        self.assertEqual(metrics(prediction, target, 0)["rmse_rad"], 2.)
        self.assertEqual(metrics(target, target, 0)["rmse_rad"], 0.)

    def test_missing_measurements_cannot_report_calibrated(self):
        data = report()
        self.assertFalse(data["transfer_ready"])
        self.assertFalse(data["auto_apply"])
        camera = next(p for p in data["parameters"] if p["id"] == "cameras")
        self.assertEqual(camera["status"], "missing")
        self.assertIsNone(camera["value"])

    def test_recorded_clock_uses_verified_nonuniform_times_and_four_second_window(self):
        study = ControllerStudy()
        try:
            row = study.rollout(0, [1., 1.], 0)
            times = row["recorded_seconds"]
            self.assertLessEqual(times[-1], 4.)
            self.assertLess(len(times), 120)
            self.assertGreater(np.ptp(np.diff(times)), .01)
            self.assertLessEqual(np.max(abs(np.asarray(row["simulated_seconds"])-times)), .000501)
            self.assertEqual(len(row["prediction"]), len(times))
        finally:
            study.close()
