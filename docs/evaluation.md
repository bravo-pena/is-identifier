# Evaluation Summary

IS Identifier 1.0 was validated on a private regulatory annotation dataset.
The dataset is not included in the public repository or model repository.

| Metric | Value | Target | Status |
| --- | ---: | ---: | --- |
| `count_macro_f1` | 0.5345 | 0.520 | PASS |
| `span_f1_partial` | 0.6723 | 0.650 | PASS |
| `recall_aim0` | 0.5739 | 0.550 | PASS |
| `recall_aim_ge1` | 0.9225 | 0.850 | PASS |

Evaluation rules:

- Rows with missing or invalid AIM count annotations were excluded from scoring.
- Empty denominators were excluded from the affected macro-average denominator.
- Reported metrics are aggregate validation metrics only; private row-level
  predictions and source sentences are not published.
