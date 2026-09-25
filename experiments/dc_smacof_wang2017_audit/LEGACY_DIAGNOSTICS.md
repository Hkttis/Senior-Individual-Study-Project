# Archived DC-SMACOF diagnostics

`legacy_v3_diagnostic.py` was formerly `tests/test_dc_smacof.py`. It is a
manual, print-oriented diagnostic from April 2026. Its T6 scanner expects a
four-column graph row, while the current data loader returns a different
shape. T7 takes arguments from T6 and cannot be collected as a standalone
pytest test. T5 has no assertions. These are reasons to preserve the file as
historical evidence, not to treat it as a passing current regression test.

`snapshots/dc_smacof_copy_20260403.py` was formerly
`tests/dc_smacof_copy.py` and is likewise a historical implementation copy.

For maintained coverage, use `tests/test_dc_smacof_direction_preprocessing.py`,
`tests/test_dc_smacof_direction_target.py`, `tests/test_dc_smacof_hpo.py`,
and `tests/test_dc_smacof_wang2017_audit.py`. Formal results are produced by
the current runner and verified from its saved output, not by these archived
diagnostics.
