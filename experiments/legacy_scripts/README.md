# Historical standalone scripts

These files were moved from `scripts/` to keep current preflight, audit,
export, and verification tools easy to find. They are preserved as source
history, not validated entry points for the current manuscript.

| File | Historical purpose |
| --- | --- |
| `paper_run_tmp.py` | Early temporary experiment dispatcher |
| `spring_main.py` | Early standalone spring simulation |
| `spring_confidence.py`, `multi_spring_confidence_ellipse.py` | Earlier confidence-ellipse prototypes |
| `open_source_data_helper.py` | Earlier source-data preparation helper |
| `model_comparison.py` | Earlier comparison script |
| `run_direction_metrics_tests.py` | Print-oriented direction metric diagnostic |

No maintained experiment runner imports these modules. Do not use their
outputs as manuscript evidence without checking the historical data and
parameter assumptions against the current pipeline.
