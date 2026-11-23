# Thyroid nodule segmentation on TN3K

A paired source-development comparison of a grayscale ResNet34 U-Net and the same model with local contrast and Sobel channels. [Study protocol](PROTOCOL.md) · [Development results](RESULTS.md) · [Supervisor meeting brief](runs/source-fold0/meeting-brief.pdf)

The [pre-meeting research audit](STAGE-AUDIT.md) covers the implemented baselines, CPU fusion control, marker intervention, reproducibility checks and remaining thesis decisions.

The [follow-up experiment index](../README.md#follow-up-experiments) covers selective correction, reference encoding, localization, display-field transfer and model ranking. [Case tables and replay order](margin-study/case-results/README.md)

## Reproduction

Use Python 3.12. The cache in this workspace was produced using NumPy 2.5.5, SciPy 1.18.1, scikit-image 0.26.0 and Pillow 13.0.0. Training used PyTorch 2.9.1, torchvision 0.24.1, NumPy 2.2.6, Pillow 11.3.0 and OpenCV headless 4.12.0.88. Evaluation uses NumPy, SciPy and Pillow. The PDF generator also uses ReportLab. Per-image predictions, review materials and audit records are included here. To restore the original model parameters, use the [checkpoint download instructions](../README.md#restore-the-four-original-checkpoints).

1. Extract the verified TN3K archive from the thesis evidence folder. Prepare the author's numeric filename order and fold 0 with `python prepare.py /path/to/datasets/tn3k local/data`. The SHA-256 of the source archive is recorded in the [data audit](../msc-readiness/thyroid-evidence/sources.json).
2. Obtain the official torchvision ResNet34 ImageNet V1 weights, `resnet34-b627a593.pth`, from download.pytorch.org/models. The frozen run record contains the full file hash. Use `python train.py --data local/data --weights local/resnet34-b627a593.pth --run runs/source-fold0 --device mps` on the original device or start a fresh run directory on CUDA. Each arm resumes at its most recent completed epoch. The run rejects mismatched inputs. Use a single training writer per directory.
3. The image-only marker review was saved in `local/blind/`. Regenerate its image order with `python review.py blind --data local/data --dir local/blind`. The hash-selected 36-image panel had clear markers in slots 1, 5, 8, 9, 12, 16, 20, 22, 23 and 26. Slots 4, 13, 24, 27, 30 and 33 were uncertain; the others had no obvious caliper. Slot to image-index mapping is in `blind-order.json`.
4. Run `python evaluate.py --data local/data --run runs/source-fold0 --markers local/blind/marker-review.json`, followed by `python marker_proximity.py --data local/data --run runs/source-fold0 --order local/blind/blind-order.json`. Use `python replay.py --data local/data --run runs/source-fold0` for fresh prediction verification on the original device. The two-page discussion PDF is generated with `python meeting.py local/data runs/source-fold0`.
5. Scoring fixtures run with `python -m unittest discover -s tests`. [Aggregate result and run fingerprints](results/summary.json) · [Checkpoint replay](results/replay.json)

The primary dataset was published by Gong et al. [Author release](https://huggingface.co/datasets/haifan-gong/TN3K) · [Source paper](https://doi.org/10.1016/j.compbiomed.2022.106389)

## Contrast and gradient ablation

The [follow-up protocol](ABLATION-PROTOCOL.md) fixes the original controls and isolates the two input changes. From the project directory, run `python channel_ablation.py` in the original MPS training environment with OpenCV available. The runner verifies the initial recipe, checks AC power and macOS thermal state during work, and runs contrast-only followed by gradient-only. It reloads each completed checkpoint and compares every predicted mask with the saved validation output. Data, weights and the original controls occupy the same paths as the first experiment. Use one writer per run directory.

In the scoring environment, `python ablation_report.py` verifies the receipts, rescoring all four arms on paired images. In the ReportLab environment, `python ablation_brief.py` renders a meeting addendum that uses the two original rescue/harm examples selected before the ablation. The [aggregate results](results/channel-ablation.json) contain paired intervals, the factorial contrasts, subgroup sizes, training durations and thermal-state counts.
