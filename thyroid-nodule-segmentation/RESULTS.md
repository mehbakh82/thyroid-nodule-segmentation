# Development result for discussion with Dr. Amiri

**The fixed three-channel input showed little average benefit on TN3K.** On fold 0, adding local contrast and gradient magnitude to grayscale changed mean Dice by **+0.18 percentage points**, with a paired 95% image-bootstrap interval of **−0.66 to +1.07 points**. This is a source-development result from two matched 12-epoch fits. It supports retaining grayscale as the working baseline.

| Endpoint on 576 validation images | Grayscale | Grayscale + contrast + gradient |
|---|---:|---:|
| Mean foreground Dice | 0.8150 | 0.8167 |
| Mean IoU | 0.7251 | 0.7263 |
| Mean surface Dice within 2 pixels | 0.4875 | 0.4924 |
| Mean HD95, pixels on 224-pixel long-side grid | 18.33 | 18.76 |
| Zero-overlap predictions | 9 | 8 |
| Empty predictions | 2 | 2 |

Dice improved on 288 images, decreased on 282 and tied on six. Three grayscale misses acquired some overlap, and two images lost all overlap. Under the stricter threshold of Dice ≥0.50, two misses were rescued and one acceptable prediction became a complete miss. Both arms missed the target in six of the same images. Among images with Dice ≥0.50 in both arms, surface Dice improved by at least 0.10 in 67 cases and worsened by at least 0.10 in 70. These are concrete reversals for reviewing what constitutes a meaningful correction.

| Prespecified development subgroup | Images | Grayscale Dice | Three-channel Dice | Paired difference, points (conditional 95% interval) |
|---|---:|---:|---:|---:|
| Small by relative area | 144 | 0.7793 | 0.7866 | +0.73 (−1.11, +2.74) |
| Low-gradient boundary proxy | 148 | 0.8370 | 0.8332 | −0.38 (−1.62, +0.79) |

Both subgroup cutoffs came from training images. The low-gradient subgroup had higher baseline Dice than the remaining 428 images (0.8370 vs 0.8073). Its relationship to a clinician's assessment of an indistinct margin needs direct examination.

**Measurement marks.** In an image-only review of 36 validation images before reading paired predictions, ten showed clear calipers, twenty had no obvious marks and six were uncertain. Mean Dice changes were −4.35, −0.39 and −5.86 points respectively. For seven clearly marked images with 20 manually localized crosses, 586 background pixels were within four pixels of a selected center. The grayscale arm predicted foreground on 120, and the three-channel arm on 167. False foreground increased in five images and tied in two. This small convenience subset, its approximate coordinates and the proximity of real boundaries to calipers preclude a causal interpretation. [Proximity counts](results/marker-proximity.json)

**Execution and inference.** The H100 server timed out during SSH negotiation. Both models ran on the same local Apple GPU, with identical initial parameters, image order, geometric/photometric augmentation, architecture and training budget. There were 2,303 training images and 576 validation images. A conservative near-copy check flagged no cross-fold matches. The release does not supply reliable patient identifiers for this analysis. The interval conditions on the fitted models, treats images as sampling units, and excludes training-seed variation. The comparison tested the combined input change. A separate contrast-only arm would be required to isolate the gradient's effect. The official TN3K test set and DDTI were outside model fitting and scoring.

**Meeting decision.** Use the representative examples to ask whether the important clinical failure is selecting the wrong structure, underdrawing a useful margin, or including adjacent tissue. The current result gives little support for building a larger network around the fixed edge input. The visible-marker pattern merits a clinician-reviewed, matched artifact experiment if it corresponds to the intended use. [Local two-page brief](runs/source-fold0/meeting-brief.pdf) · [Full example panel](runs/source-fold0/meeting-panel.png)

Both checkpoints were reloaded independently and reproduced every validation mask bit for bit. [Aggregate record](results/summary.json) · [Replay record](results/replay.json)
