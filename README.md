# Thyroid nodule segmentation on ultrasound

An MSc thesis feasibility study by Mehran Bakhtiari comparing grayscale, contrast-enhanced and gradient inputs for automatic nodule segmentation on TN3K. The experiments use a matched ResNet34 U-Net, 2,303 training images and 576 fold-0 validation images, with one optimization seed. The accompanying DDTI work covers cohort preparation. External model evaluation is pending.

| TN3K development comparison | Mean foreground Dice |
|---|---:|
| Grayscale | 0.8150 |
| Grayscale + contrast | 0.8156 |
| Grayscale + gradient magnitude | 0.8147 |
| Grayscale + contrast + gradient | 0.8167 |
| Average of grayscale and contrast models, two passes | 0.8200 |
| Grayscale with horizontal-flip averaging, two passes | 0.8221 |

Flip averaging improved Dice by 0.71 points over single-pass grayscale (paired 95% image-bootstrap interval +0.38 to +1.10). The audit also found that different decoding rules for the JPEG reference masks particularly affect small nodules. These are exploratory, image-level development results. Reliable patient identifiers and external performance are open questions. [Stage audit](thyroid-nodule-segmentation/STAGE-AUDIT.md) · [One-page summary](thyroid-nodule-segmentation/results/supervisor-meeting-summary.pdf)

## Explore the study

- Read the [thesis evidence brief](msc-readiness/thyroid-evidence/README.md), [dataset atlas](msc-readiness/thyroid-evidence/DATASETS.md), [prior-art comparison](msc-readiness/thyroid-evidence/PRIOR-ART.md), and [four-paper reading note](msc-readiness/THYROID-NODULE-READING.md).
- Inspect the [initial experiment](thyroid-nodule-segmentation/RESULTS.md), [four-arm ablation](thyroid-nodule-segmentation/ABLATION-RESULTS.md), [boundary sensitivity analysis](thyroid-nodule-segmentation/BOUNDARY-AUDIT.md), and [DDTI preflight](thyroid-nodule-segmentation/DDTI-PREFLIGHT.md).
- Review the [blinded case packet](thyroid-nodule-segmentation/runs/ablation-fold0/contrast-case-review/blinded-cases.pdf) and [blank response form](thyroid-nodule-segmentation/runs/ablation-fold0/contrast-case-review/response-sheet.csv). Readers who want to remain blinded should record their judgments before consulting the assignment keys and scored examples.
- Follow the [study reproduction instructions](thyroid-nodule-segmentation/README.md) for data preparation, training, checkpoint replay, per-image scoring, and PDF generation. The repository includes all four binary validation prediction archives, training histories, paired scores, bootstrap results, unblinding keys, aggregate and per-case audits, and DDTI reference mappings.

## Restore the four original checkpoints

From the repository root run `python3 restore_checkpoints.py`. The script downloads four files (about 1.01 GiB total) from the [v0.1.0 release](https://github.com/mehbakh82/thyroid-nodule-segmentation/releases/tag/v0.1.0), verifies the original evaluation SHA-256s and restores each training checkpoint to its expected location. For example, the grayscale file belongs at `thyroid-nodule-segmentation/runs/source-fold0/gray/checkpoint.pt`. Model weights, optimizer state and epoch history are included in each checkpoint.

Acquire the source TN3K and DDTI archives from the [pinned dataset URLs and checksums](msc-readiness/thyroid-evidence/sources.json) and place them under `msc-readiness/thyroid-evidence/local-data/`. The evidence directory contains scripts for auditing the releases and reproducing their figures. The diagnostic marks, sample manifests and visualizations refer to public source images. [Dataset attribution](THIRD-PARTY-NOTICES.md)

Run `python3 verify_public_release.py` from the repository root for a data-free integrity check. For the research tests, install the dependencies described in the study instructions and run `python -m unittest discover -s tests` from `thyroid-nodule-segmentation/`.
