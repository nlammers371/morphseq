import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from analyze.difference_detection.classification_test_multiclass import (
    _make_logistic_classifier,
    run_classification_test,
)


class ClassWeightPolicyTests(unittest.TestCase):
    def test_make_logistic_classifier_uses_balanced_weights(self):
        clf_binary = _make_logistic_classifier(n_classes=2, random_state=42)
        clf_multi = _make_logistic_classifier(n_classes=3, random_state=42)

        self.assertEqual(clf_binary.class_weight, "balanced")
        self.assertEqual(clf_multi.class_weight, "balanced")

    def test_run_classification_test_verbose_reports_balanced_policy(self):
        df = pd.DataFrame(
            {
                "embryo_id": ["e1", "e2", "e3", "e4"],
                "cluster": ["A", "A", "WT", "WT"],
                "predicted_stage_hpf": [10.0, 10.5, 10.2, 10.7],
                "z_mu_b_0": [0.1, 0.2, -0.1, -0.2],
            }
        )

        fake_result = {
            "ovr_classification": {
                "A": pd.DataFrame(
                    [
                        {
                            "time_bin": 8,
                            "time_bin_center": 10.0,
                            "auroc_observed": 0.75,
                            "pval": 0.05,
                        }
                    ]
                )
            }
        }

        with patch(
            "analyze.difference_detection.classification_test_multiclass.run_multiclass_classification_test",
            return_value=fake_result,
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                run_classification_test(
                    df=df,
                    groupby="cluster",
                    groups=["A"],
                    reference="WT",
                    features=["z_mu_b_0"],
                    n_permutations=0,
                    verbose=True,
                )

            output = buf.getvalue()

        self.assertIn("Classifier policy: class_weight='balanced'", output)


if __name__ == "__main__":
    unittest.main()
