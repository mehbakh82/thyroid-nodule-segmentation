# One prediction, two thyroid nodule outlines

**A one-pixel shrinkage changed from a gain to a loss, depending on the reference, in 21 of 69 scans whose outlines appeared to trace the same nodule.** In 12 of those scans, both changes were at least two surface-Dice points in opposite directions. Four remained reversals at five points in both directions. The finding passes the project's prespecified threshold of five material reversals to prioritize independent tracing. [Protocol](PROTOCOL.md) · [Results](results.json) · [Generate the meeting examples](meeting.py) · [All 76 paired changes](../case-results/paired-changes.csv)

The [frozen two-model comparison](MODEL-RANKING.md) extended this test to model selection. On the 69 apparent same-target scans either model could lead in mean surface Dice when the available outline was chosen separately per scan.

## Target review

The TN-SCUI2020 training archive contains 76 distinct grayscale scans each released under two IDs with different binary nodule masks. Native reference-to-reference region Dice ranged from .182 to .981, with median .931. Each pair had matching classification labels. Five pairs failed the geometric correspondence screen, and pre-score visual inspection flagged two more for a conspicuous extension present in only one outline. The other 69 pairs appeared to outline the same target. This was a nonclinical review. Mask provenance and grader identity remain unknown. [Transfer and source audit](../tnscui-transfer/README.md) · [Recorded review](visual-review.json)

## Effect of shrinking the predicted contour

All 76 scans were scorable. Each scan supplied one frozen baseline prediction and one erosion on its unpadded 224-long-side grid. The positive direction means that shrinkage increased surface Dice within a two-pixel tolerance. A material reversal requires a gain of at least two points under one reference and a loss of at least two under the other.

| Scan group | Scans | Both improve | Both worsen | Opposite directions | Tie involved | Material reversals |
|---|---:|---:|---:|---:|---:|---:|
| Apparent same target | 69 | 20 | 23 | 21 | 5 | 12 |
| Flagged for target or extent | 7 | 2 | 3 | 2 | 0 | 0 |
| All | 76 | 22 | 26 | 23 | 5 | 12 |
| Baseline localized under both references, within the 69 | 60 | 20 | 22 | 17 | 1 | 10 |

For the 69 apparent same-target scans, the median absolute difference between the two estimates of shrinkage's effect was 5.35 surface-Dice points. Averaging both references equally gave a mean effect of −0.93 points. The number of reversals clearing one, two and five points on both sides was 15, 12 and 4 respectively. On the secondary region-Dice metric, 13 pairs reversed direction but none cleared even one point on both sides. These are observed counts in a selected set of repeated scans. A clinical tolerance in millimeters, expert reader agreement and the frequency of reversals in an unselected population remain open questions.

## Three cases for the meeting

| Scan | Reason for inclusion | Change with A | Change with B |
|---|---|---:|---:|
| P44 | Strongest same-target reversal, ranked by the smaller absolute change | −15.91 | +7.62 |
| P73 | Strongest same-direction result by the same criterion | −26.09 | −20.58 |
| P30 | Lowest overlap of the two references, with markedly different outlined extents | −0.39 | +0.90 |

Reference A has the lower numeric release ID within each pair. P44's reference masks had .926 region Dice with each other. This shows how similar region outlines can give opposite judgments on an edit to the margin. P73 illustrates a case where both outlines support keeping the original prediction. The case-level CSV gives the signed changes for both metrics, correspondence disposition and reversal strength.

For Dr. Amiri, use these three examples to discuss which margin differences are anatomically meaningful, then arrange independent delineation by a reader experienced in thyroid ultrasound. The existing 12-case blinded kit uses separate TN3K development cases. It tests agreement with shrinkage in that source. Direct adjudication of a TN-SCUI2020 example would require a reader who has not seen its overlays. [Existing tracing protocol](../READER-COMPARISON.md)

## Reproduction and verification

From the thyroid project directory, use the CPU PyTorch environment for the preceding transfer experiment. The runner verifies the training archive, prior audit, fixed checkpoint and review fingerprints. The first stage regenerates the five local image-and-reference review sheets in the original order. Replaying the second stage uses the stored pre-score visual dispositions. Images, release identifiers in the case-level export, and weights stay local.

```sh
python margin-study/paired-references/pairs.py prepare
python margin-study/paired-references/pairs.py evaluate
python margin-study/paired-references/verify.py
python margin-study/paired-references/meeting.py
```

The independent scoring check reruns inference using the second release ID of each scan, reconstructs the erosion with explicit neighbor comparisons and computes boundary matches with a radius-two disk. It checks 76 binary prediction hashes, 152 reference evaluations, and the primary counts in all four strata.
