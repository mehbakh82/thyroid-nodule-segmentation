# Selective margin correction on thyroid ultrasound

**23 November 2026. Keep grayscale with flip averaging as the development baseline. Stop this version of the correction rule.** We assumed the intended use was an already localized nodule, preserving well-traced tissue mattered, and visible calipers were incidental. A specialized model that could reliably identify poor contour segments might still help. This pilot tested one concrete version with existing data on CPU.

## Experiment and decision

The source U-Net was frozen. Two lightweight local classifiers learned foreground/background labels in a four-pixel band around its predicted contour. Both received intensity, multiscale gradients, normal-direction samples and flip-based model confidence/disagreement. One learned the >127 reference convention and one learned the ==255 convention. The selective rule edited a pixel only on strong agreement. The predictors and edit region at inference used only the scan and frozen model predictions. Each of the 576 validation images was scored with correction heads fitted on other images. The original reference was used afterwards to identify the 534 cases with baseline region Dice at least .50. This is an image-level operational proxy for localization.

| Method | Mean surface Dice on 534 localized images, >127 | Mean surface Dice on the same images, ==255 | Mean region Dice on all 576, >127 |
|---|---:|---:|---:|
| Frozen two-view grayscale baseline | .52350 | .46934 | .82209 |
| Four iterations of geodesic active contour | .49498 | .51103 | .80880 |
| Local classifier, primary labels | .52340 | .47075 | .82122 |
| Two classifiers, average probability | .51501 | .49434 | .81612 |
| Two classifiers, selective agreement | .52359 | .46942 | .82207 |

The primary paired difference was **+0.009 surface-Dice points** (95% paired image-bootstrap interval **−0.019 to +0.043**). Tolerance was two pixels at 224-pixel long-side resolution. The proposed rule exceeded the generic two-head average by 0.86 points, but that comparator itself reduced the primary score. All five methods had seven complete misses. See the [protocol fixed before scoring](PROTOCOL.md) and [aggregate results](results.json).

The selective rule changed 670 pixels across just 35 of the 534 localized images. Where both reference encodings agreed, 340 changed from wrong to right and 321 from right to wrong. Nine changes landed in decoder-disagreement regions. Only one localized image gained five or more surface-Dice points and none lost five or more. The unguarded two-head average edited 131,104 pixels in the same images. The guarded implementation abstained frequently and achieved little net correction. Clinical readers have not validated whether the edits were anatomically preferable.

## Two different questions about opportunity

1. **Are there correctable pixels nearby?** In the 534 localized images, 326,770 of 684,434 erroneous pixels (47.7%) lay in the permitted band. A deliberately reference-guided upper bound that corrected *every* erroneous in-band pixel would lift mean **region Dice** from .8684 to .9448. It estimates the geometric opportunity under the primary reference. The method's primary metric was **surface Dice**, and the oracle uses unavailable reference information. [Opportunity calculation](opportunity.json)
2. **Can a scoring convention reward an anatomically unproven change?** In a post-hoc diagnostic, shrinking all baseline masks by one pixel changed mean surface Dice by −1.94 points using >127 and +4.03 points using ==255. Growing them by one pixel produced −2.72 and −6.76 points respectively. Under strict decoding, a subset of removed foreground pixels became enclosed background. Filling reference holes reduced the shrink effect to +3.47 points under strict decoding while the primary effect was still −1.94. The source annotations give no independent reader vote for shrinking a nodule. [Offset diagnostic](offset-check.json)

## Interpretation for Dr. Amiri

The two label definitions are technical interpretations of a single JPEG. The intervention tests digital reference stability; it cannot substitute for independent experts. This cohort informed earlier hypothesis selection, so cross-fitting limits direct leakage during refiner fitting but does not provide a fresh confirmatory cohort. Image-bootstrap intervals condition on the fitted predictions and exclude patient clustering, training-seed variation and refitting uncertainty. The two-point investment threshold was an engineering assumption. DDTI remains available for a frozen external test if a later method justifies one.

For the meeting, prioritize the actual boundary that should count as correct, including how calipers and poorly visible arcs should be handled. An original lossless contour/polygon export, or blinded redelineation of a small sample before comparing models, could distinguish reference construction from anatomical improvement. If the lab identifies a reproducible local error, develop a method aimed at that error and compare it against both the frozen flip baseline and the closest published thyroid-specific refiner on comparable data. This pilot included a classical active-contour benchmark and capacity-matched learned controls. [Thyroid-specific residual refinement](https://doi.org/10.3389/fcell.2026.1879094) · [Spatially varying smoothing precedent](https://pmc.ncbi.nlm.nih.gov/articles/PMC9974964/)

## Replay

The [independent tracing protocol](READER-COMPARISON.md) contains the reader kit, the TN3K source check, and the procedure for scoring returned outlines. The scorer accepts native-size lossless binary PNGs and records ambiguous or untraceable cases separately.

From the thyroid project directory, run the cache stage in the original CPU-compatible PyTorch environment and the remaining scripts in the evaluation environment with SciPy, scikit-image and scikit-learn. This run used scikit-learn 1.9.1, SciPy 1.18.1, scikit-image 0.26.0 and NumPy 2.5.5 for fitting and scoring. Use the author archive's extracted train/validation masks.

```sh
python margin-study/cache_baseline.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python margin-study/pilot.py /path/to/tn3k/trainval-mask
python margin-study/opportunity.py
python margin-study/offset_check.py /path/to/tn3k/trainval-mask
python -m unittest discover -s margin-study -p 'test_*.py'
```

The baseline on every scored image was compared to the prior CPU flip prediction's region and surface Dice. Sample counts, image folds, checkpoint and script fingerprints are in the aggregate result. Training data, local probability arrays and per-image records stay in the working directory. [One-page meeting update](meeting-update.pdf)
