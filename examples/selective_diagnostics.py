"""Run with: python examples/selective_diagnostics.py (after installing FLARE)."""
import json

from flare.selective_metrics import selective_classification_report


# Synthetic labels only: all five true TDEs are correctly selected, but 62
# other objects are also labelled TDE. True-class protection is not precision.
classes = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
truth = [4] * 5 + [0] * 62 + [1] * 3
prediction = [4] * 67 + [-1] * 3
report = selective_classification_report(truth, prediction, class_names=classes)
print(json.dumps({"true_TDE": report["by_true_class"]["TDE"],
                  "emitted_TDE": report["by_predicted_class"]["TDE"]},
                 indent=2, allow_nan=False))
