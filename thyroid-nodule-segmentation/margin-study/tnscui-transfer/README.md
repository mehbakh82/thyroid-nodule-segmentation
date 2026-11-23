# Does the TNUI display-border filter transfer to TN-SCUI2020?

**Use the unfiltered TN3K grayscale model as the general baseline.** The frozen filter raised mean Dice by 32.19 points on 48 fresh TNUI development images. On 160 TN-SCUI2020 development images, the gain was 0.087 points and the paired 95% interval included zero. A full reference-mask audit found several nodules where the filter suppressed a substantial part of the annotation. [TNUI replication](../tnui-location/border-repeat.json) · [Transfer protocol](PROTOCOL.md) · [Primary results](results.json)

## Reference tissue retained

We checked all 3,644 image/mask pairs at their released resolution. All had aligned dimensions and binary masks with foreground value 255. The series contained 1,601 distinct width/height combinations. The filter trimmed at least one row on 3,139 records.

| Reference-mask outcome | All 3,644 records | 24 boundary records | 3,620 remaining records |
|---|---:|---:|---:|
| Any labeled area removed | 15 | 15 | 0 |
| More than 1% removed | 14 | 14 | 0 |
| More than 5% removed | 9 | 9 | 0 |
| Pooled labeled area retained | 99.824% | 90.428% | 100% |
| Largest loss on one record | 44.01% | 44.01% | 0% |

The boundary group includes labels within 2% of image height of a retained band limit and labels extending past it. Its definition puts every tissue-loss case in this group by construction. The two safety criteria required at least 99% pooled retention and zero records with more than 5% loss. Only the pooled criterion passed. [Generate overlays of the two largest losses and a boundary control](examples.py)

## Frozen model on the 160 prespecified cases

All arms used the same whole-image prediction, flip averaging and native scoring grid. The two filters simply suppressed predictions outside their retained rows. The fixed control used band positions estimated in the earlier 96-image TNUI study. Bootstrap intervals use paired records and a frozen checkpoint.

| Postprocessing | Mean Dice | Mean sensitivity | Complete misses | False-positive pixels removed | True-positive pixels removed |
|---|---:|---:|---:|---:|---:|
| None | .79664 | .82960 | 7 | 0 | 0 |
| Image-specific filter | .79750 | .82960 | 7 | 8,100 | 0 |
| Fixed TNUI band | .79755 | .82956 | 7 | 11,169 | 803 |

Versus the unfiltered baseline, the image-specific filter gained **0.087 Dice points** (95% paired image-bootstrap interval **0.000 to 0.260**), with two gains and 158 ties. It failed the prespecified requirement that the lower limit exceed zero. The fixed band gained 0.092 points (interval 0.0004 to 0.268). A higher average Dice can coexist with removal of correctly predicted tissue.

## What happened in the boundary cases?

Only one of the primary 160 images belonged to the boundary group. After the primary analysis, we reran the frozen model on all 24 boundary records, representing 22 distinct rasters. This is a descriptive follow-up selected using the reference labels. Six records (five distinct images) lost correct foreground and worsened in Dice, 18 tied, and none improved. Across these records the filter removed 54,541 true-positive pixels and 114 false-positive pixels. Mean Dice fell from .6744 to .6526, with no new complete misses. The largest individual drop was .8430 to .5622 on the scan with the greatest reference-area loss. The sample overlap reproduced the primary scores exactly. [Boundary results](boundary-results.json)

## Provenance and interpretation

The challenge organizers describe 3,644 training images from 3,644 patients with identifiers and binary nodule annotations. The pinned mirror supplies 3,644 distinct CSV IDs, but 76 pairs of IDs share identical decoded grayscale pixels and have different masks. This leaves 3,568 distinct rasters. The nine records losing over 5% of the reference represent seven distinct images. We scored one deterministically selected mask per distinct raster for the primary comparison and kept every reference in the safety audit. Patient linkage beyond the release documentation is unverified. [Organizer description](https://tn-scui2020.grand-challenge.org/Dataset/) · [Pinned data mirror](https://huggingface.co/datasets/us-segmentator/us-segmentation-dataset/tree/6e7dadb8681fa5a137a1a607127510e36235610d/zips/Thyroid)

We found zero exact native grayscale matches against 2,879 TN3K development images or 192 previously sampled TNUI records. This screen cannot establish patient independence across sources or detect resized copies. Conclusions apply to this distributed training release and frozen checkpoint. DDTI and the official test partitions remain available for later evaluation.

For the supervisor meeting, inspect the two largest losses and a retained boundary case alongside a successful TNUI example. Identify which dark regions a reader can confidently distinguish from real nodule tissue. Keep this rule specific to the tested TNUI format while that distinction remains unresolved.

## Reproduction

Create `local/tnscui-transfer/`, then place the pinned archive at `local/tnscui-transfer/TNSC2020.zip` and retain the original TN3K source checkpoint, native training images, and three prior TNUI development sample manifests and images. Run from the thyroid project directory in the CPU PyTorch environment. The runner verifies the archive, checkpoint, frozen rule and discovery bounds. Sampling ranks the full, case-sensitive PNG filename (including its extension) by SHA-256 with the registered salt.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/tnscui-transfer/transfer.py /path/to/datasets/tn3k
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/tnscui-transfer/boundary_followup.py
python margin-study/tnscui-transfer/examples.py
```
