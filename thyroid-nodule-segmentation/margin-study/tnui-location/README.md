# Target localization in TNUI ultrasound images

**23 November 2026. Target location warrants further investigation.** We applied the frozen TN3K grayscale flip-averaged model to a fresh 96-image sample of the TNUI author training partition. An equal-size crop centered on the annotated nodule improved mean whole-frame Dice by **12.82 points** over a crop at the image center (paired 95% image-bootstrap interval **+6.34 to +19.68**). The median paired gain was **1.59 points**, with 62 improvements and 34 deteriorations. This met the provisional five-point development threshold. The target-centered condition uses the reference to supply a location; its performance is a diagnostic of opportunity.

| Input or output restriction | Mean Dice | Mean target sensitivity | Mean false foreground / target area | Complete misses |
|---|---:|---:|---:|---:|
| Full image | .4438 | .8290 | 7.36 | 3 |
| Fixed image-center crop | .6449 | .6620 | .64 | 10 |
| Reference-centroid crop | .7731 | .8811 | .57 | 0 |
| Automatic crop at first prediction's largest component | .4934 | .7009 | 2.35 | 8 |
| Original whole-frame prediction clipped to reference window | .7761 | .8272 | .37 | 3 |
| Image-only horizontal display-border filter (post-hoc) | .8035 | .8290 | .23 | 3 |

Models saw 224-pixel long-side inputs and an original-plus-flipped probability average. Crops covered 390 × 390 native pixels within the 780 × 780 image. Each crop prediction was projected onto the whole native frame; all annotated tissue outside a crop counted as missed. Baseline false foreground was 7.36 times target area on average and 2.35 times target area at the median. Mean reference area fraction and case mix differ across acquisitions. Numerical comparisons with other datasets should retain their original scoring grid.

## Interpretation

- Removing predictions outside the known reference window, without rerunning the network, yielded .7761 Dice. Rerunning on the reference-centered crop yielded .7731 (paired difference −0.30 points, interval −3.24 to +2.97). The bulk of the aggregate gain can already be obtained by suppressing distant false positives *given the correct window*. A cropped run did improve mean sensitivity and recovered the three remaining complete misses.
- The central window fully contained the target in 46/96 cases, and excluded the entire target in two. It improved mean Dice but produced ten complete misses. The reference-centered window fully contained the target in 91/96 cases. The automatic window did so in 60/96. Its Dice improved over whole-image inference by 4.96 points, yet it lagged the central crop by 15.15 points. Therefore a reliable automatic locator remains an open question.
- Fresh refers to distinct filenames and distinct native grayscale arrays relative to the earlier 48 TNUI images and the 2,879 TN3K development images screened. Patient IDs, other views, and resized copies have not been ruled out. All 96 cases had aligned positive binary masks and remained eligible. There was one frozen TN3K checkpoint. No training, GPU, official TNUI validation/test outcomes, or DDTI scoring was used.
- The results separate useful spatial information from the deployment challenge of obtaining it. For the meeting, ask whether a clinician-supplied approximate center is part of the intended workflow. For unprompted use, compare an image-only acquisition-field filter and an independent automatic locator with the fixed central crop; report missed nodules as well as Dice. Continue expert tracing to establish which of the surviving boundary differences matter anatomically.

[Protocol](PROTOCOL.md) · [All paired results and input fingerprints](results.json) · [Generate the three example overlays](examples.py) · [TNUI release](https://github.com/zxg3017/TNUI-2021-/tree/5fa417629f581d82017eed535cdd62586717b669)

## What explained most of the apparent localization opportunity

The example overlays revealed foreground predictions in the horizontal black display borders. On the 96 images used to discover this issue, a simple rule estimated the top and bottom of the ultrasound field using grayscale intensities only. Clearing predictions outside those rows removed 4,135,748 of 4,363,685 false-positive pixels (94.8%). All annotated nodule pixels were retained. The rule produced .8035 mean Dice and changed 66 images for the better. This initial comparison is post-hoc. [Border rules](BORDER.md) · [Discovery results](border-results.json) · [Generate before/after overlays](border_examples.py)

**Frozen within-source replication.** On a further 48 hash-selected TNUI training images disjoint from both earlier samples, the frozen filter raised Dice from **.5107 to .8326**. The paired gain was **32.19 points** (95% image-bootstrap interval **23.98–40.73**). It removed **84.9% of pooled false-positive pixels**, retained all 1,124,026 annotated nodule pixels and left complete-miss count at zero. Thirty-one images improved, 17 tied and none worsened. A constant band, set once using the discovery cohort's median top and bottom row fractions, achieved .7305. The image-specific band gained **10.21 points** over that control (4.44–16.70). [Replication results](border-repeat.json)

**Decision.** Include the frozen scan-field rule in the conventional baseline before investing in a nodule localizer. This filtering step uses image-only display geometry, has no trained parameters and does not require a nodule center. Its effect was replicated within one data source with a characteristic display format. It establishes removal of an acquisition/display artifact and gives a more honest estimate of transfer performance for this model on these images. It does not by itself establish improved anatomical margins, clinical usefulness, generalization to other display formats, or a novel thesis method. The subsequent [TN-SCUI2020 transfer check](../tnscui-transfer/README.md) found that the rule cut into peripheral nodules and gave a negligible average benefit. Keep its use specific to this TNUI format. Keep the reader tracing exercise for the separate question of contour validity.

## Reproduction

Run from the thyroid project directory in the existing CPU PyTorch environment with Pillow, SciPy and NumPy. Reuse the sample manifest of the earlier 48-case experiment and the verified native TN3K images.

```sh
python margin-study/tnui-location/fetch.py local/tnui-location local/tnui-control/sample.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/tnui-location/pilot.py local/tnui-location local/tnui-control /path/to/datasets/tn3k
python margin-study/tnui-location/examples.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/tnui-location/border_check.py
python margin-study/tnui-location/fetch.py local/tnui-border-repeat local/tnui-control/sample.json --location-sample local/tnui-location/sample.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/tnui-location/border_repeat.py /path/to/datasets/tn3k
python margin-study/tnui-location/border_examples.py
python -m unittest discover -s margin-study/tnui-location -p 'test_*.py'
```
