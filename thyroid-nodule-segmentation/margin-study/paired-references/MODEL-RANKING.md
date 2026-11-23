# Can the reference outline change which trained model ranks first?

**Yes for mean surface Dice on the 69 apparent same-target scans.** The grayscale-plus-contrast-and-gradient model led by 1.43 points when every scan used reference A, and by 0.94 points when every scan used B. Allowing either recorded outline for each scan yielded an attainable mean difference from −2.46 to +4.84 points. For region Dice, even the least favorable selection gave the three-channel model a small lead of 0.16 points on these 69 scans. [Prespecified protocol](MODEL-RANKING-PROTOCOL.md) · [Aggregate results](model-ranking-results.json) · [Per-scan differences and endpoint assignments](../case-results/model-ranking.csv)

## Cohort rankings

Each value is a mean paired difference in percentage points, three-channel model minus grayscale. A and B are file-ID conventions within each pair. Their ordering carries no verified information about readers. Each range endpoint is achieved by choosing one of the two observed outlines per image and using that same outline to score both models. The range measures sensitivity to this choice on the specified scans. Both endpoint assignments appear in the case-level CSV.

| Scans | Metric | All A | All B | Equal reference weight | Attainable range |
|---|---|---:|---:|---:|---:|
| 69 apparent same-target | Surface Dice, 2 pixels | +1.43 | +0.94 | +1.19 | −2.46 to +4.84 |
| 69 apparent same-target | Region Dice | +0.81 | +0.67 | +0.74 | +0.16 to +1.32 |
| 7 flagged | Surface Dice, 2 pixels | −5.03 | −2.38 | −3.71 | −7.74 to +0.32 |
| 7 flagged | Region Dice | −6.39 | −7.16 | −6.77 | −8.95 to −4.60 |
| All 76 | Surface Dice, 2 pixels | +0.83 | +0.64 | +0.74 | −2.95 to +4.42 |
| All 76 | Region Dice | +0.15 | −0.05 | +0.05 | −0.68 to +0.78 |

Reference choices at the endpoints use knowledge of both models' scores. They establish what is possible in these data. The frequency of such choices by an independent expert remains unknown. The finite-set ranges carry no sampling-confidence or patient-population interpretation. Including the seven scans flagged for target correspondence or extent also removes the robustness of the region-Dice lead.

## Individual scans

| Group | Metric | Three-channel wins twice | Grayscale wins twice | Opposite winners | Tie involved | Opposite by ≥2 points each way |
|---|---|---:|---:|---:|---:|---:|
| Apparent same-target | Surface Dice | 30 | 22 | 13 | 4 | 8 |
| Apparent same-target | Region Dice | 25 | 26 | 15 | 3 | 0 |
| Flagged | Surface Dice | 2 | 3 | 2 | 0 | 1 |
| Flagged | Region Dice | 1 | 3 | 3 | 0 | 2 |

Five same-target scans changed surface-Dice winner by at least five points in each direction. In P39, the same two predictions yielded **+10.99** points under A and **−9.92** under B. In P20, both references supported the three-channel prediction, by **+24.83** and **+27.91** points. The examples were selected by the smaller magnitude across the two references, as specified before scoring. [Generate the meeting examples](model_examples.py)

## Meeting decision

Keep grayscale as the working cross-source baseline. These 69 selected scans show how a small claimed boundary improvement can depend on which available reference is used. They also contain cases where both references agree strongly. Ask Dr. Amiri which margin distinctions matter and arrange independent tracing before selecting a model on a small surface-Dice lead. The prepared blinded kit uses separate TN3K development cases. [Reader protocol](../READER-COMPARISON.md)

## Reproduction

The two frozen fold-0 U-Nets share architecture, initialization and training budget. Both predictions average the probabilities from an original and a horizontally flipped view. The original training channel function constructs grayscale or grayscale/CLAHE/Sobel channels on the resized unpadded image. Both use the same padding, normalization, threshold and scoring grid. All 76 scans are scored, including misses. This experiment uses a single pair of trained checkpoints and a previously investigated, nonrandom subset of a training release. A/B provenance and patient independence are unverified. DDTI and official test sets were outside the evaluation.

From the thyroid project directory, use the training-compatible CPU environment (PyTorch 2.9.1, OpenCV 4.12.0, NumPy 2.2.6, Pillow 11.3.0 and SciPy). Install OpenCV in the inference environment.

```sh
python margin-study/paired-references/model_ranking.py
python margin-study/paired-references/verify_model_ranking.py
python margin-study/paired-references/model_examples.py
```

The grayscale masks and scores matched the prior experiment on all 76 scans. Independent region overlap and radius-two boundary matching reproduced all 304 model/reference evaluations. A separate replay on six TN3K validation scans per arm matched the archived single-view predictions and two-view predictions computed through the original training loader.
