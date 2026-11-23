# Case-level results and replay order

Each case contributes once within the analysis named below. The table IDs apply within a study. The three TN-SCUI2020 transfer tables share a record number; their scan number groups records with identical decoded ultrasound pixels. The paired-reference tables use P01 to P76 and order the two reference masks by ascending source-file number. Source filenames and patient linkage are unnecessary for recalculating the published score comparisons.

| Study | Published cases |
|---|---|
| TN3K selective correction and controls | [576 validation records across methods, decoding rules and tolerances](tn3k-margin.json) |
| TNUI reference encoding | [48 PNG/JPEG comparisons](tnui-codec.json) · [Resolution comparison on the same 48 scans](../tnui-control/scale-results.json) |
| TNUI localization and display borders | [96 localization comparisons](tnui-localization.json) · [96 display-field measurements](tnui-border-discovery.json) · [48 fresh three-way comparisons](tnui-border-repeat.json) |
| TN-SCUI2020 filter transfer | [3,644 reference-mask checks](tnscui-field.json) · [160 paired model evaluations](tnscui-model.json) · [24 boundary records](tnscui-boundary.json) |
| TN-SCUI2020 alternative references | [76 pairs with baseline and edited scores](paired-edit-scores.json) · [Paired changes and target review](paired-changes.csv) |
| Two-model ranking | [Scores, per-image differences and reference assignments](model-ranking.csv) · [152 packed binary model predictions](model-ranking-predictions.npz) |

Ranking differences are three-channel minus grayscale in percentage points. Within each group the two extreme assignments in the CSV attain the reported bounds. Prediction arrays are keyed by pair and model, such as P01_gray, and stored as NumPy packbits along the width. Unpack along axis 1 and trim to the width in the CSV.

## Replay

For a check requiring only the Python standard library, run `python3 verify_public_release.py` from the repository root. For inference, [restore the checkpoints](../../../README.md#restore-the-four-original-checkpoints), obtain TN3K from its documented release, and follow the study instructions below from the thyroid project directory.

1. [Prepare the source development data](../../README.md#reproduction), then [run the cross-fitted margin comparison and offset controls](../README.md#replay). Use `tracing_packet.py` to regenerate the local investigator key for the [blinded tracing protocol](../READER-COMPARISON.md).
2. [Fetch the first TNUI training sample and run its encoding and resolution controls](../tnui-control/README.md#replay). Then [run localization, field detection and the disjoint replication](../tnui-location/README.md#reproduction).
3. Obtain the pinned TN-SCUI2020 training archive, create `local/tnscui-transfer`, then [run the transfer audit and boundary follow-up](../tnscui-transfer/README.md#reproduction).
4. [Recreate the two-outline evaluations](../paired-references/README.md#reproduction-and-verification), then [run and verify the two-model comparison](../paired-references/MODEL-RANKING.md#reproduction). The plotting scripts in those directories recreate the ultrasound panels from the source data.
5. Regenerate the numeric tables with `python margin-study/export_case_results.py`. Use `--out` to choose a separate directory for comparison.

The original result receipts retain hashes of the inputs and byte-for-byte local outputs used in each experiment. The public score tables remove source identifiers for the new cross-source cohorts. The TN3K margin table and packed ranking predictions retain their original hashes.
