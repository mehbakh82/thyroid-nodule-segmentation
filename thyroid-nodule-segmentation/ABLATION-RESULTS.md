# Input channel comparison on TN3K

**Retain grayscale as the baseline.** Across 576 fold-0 validation images, adding CLAHE alone changed mean foreground Dice by +0.07 percentage points; adding Sobel magnitude alone changed it by −0.03 points. The paired 95% intervals span zero for each comparison. The experiment used one initialization seed and image-level grouping, with 2,303 training images per fit. The intervals condition on the fitted models.

| Input | Mean Dice | Gain versus grayscale in points (95% paired image bootstrap) | Surface Dice at 2 pixels | Mean HD95, pixels | Zero overlap |
|---|---:|---:|---:|---:|---:|
| Grayscale | .8150 | Reference | .4875 | 18.33 | 9 |
| Grayscale with contrast | .8156 | +0.07 (−0.74 to +0.95) | .4810 | 19.51 | 8 |
| Grayscale with gradient | .8147 | −0.03 (−0.81 to +0.76) | .4867 | 17.40 | 7 |
| Grayscale with both | .8167 | +0.18 (−0.66 to +1.07) | .4924 | 18.76 | 8 |

The contrast-by-gradient interaction was +0.14 points (−0.85 to +1.08). The results leave both small gains and small harms plausible. Gradient alone had the fewest zero-overlap predictions, but that difference from grayscale amounts to two images.

Among the 148 images in the training-defined low-gradient subgroup, contrast alone gained 0.57 points (−0.57 to +1.97) and gradient alone lost 0.55 points (−2.09 to +0.80). The direct contrast-only minus gradient-only comparison was +1.12 points (−0.20 to +2.81). Among the ten images with clear visual calipers in the previous blinded review, gradient alone lost 10.59 mean Dice points (−25.70 to +0.68). That small, exploratory subset motivates inspecting specific examples. It cannot establish a causal effect of calipers.

The local [meeting addendum](runs/ablation-fold0/meeting-addendum.pdf) revisits two cases chosen before these two fits. On the previous grayscale miss, contrast alone reached .917 Dice, gradient alone .804, and both .918. In the earlier harm example, grayscale reached .689, contrast alone .057, gradient alone .524, and both .000. These cases illustrate why an improvement on a particular nodule does not imply a general preference for that input.

## Execution and verification

An exploratory pre-meeting screen examined whether five whole-image descriptors could select between the fixed grayscale and contrast-only models before inference. Five deterministic image folds held each scored image out of selector fitting. A standardized ridge regression (alpha 100) estimated paired Dice gain from median intensity, intensity range, dark-pixel fraction, high-frequency residual and gradient strength. Mean Dice was .8144 using the selector, .8150 using grayscale and .8156 using contrast. Its paired gain over grayscale was −0.06 points (conditional image interval −0.60 to +0.56). A retrospective choice using the known reference would reach .8357. Contrast gained at least five Dice points on 53 images and lost at least five on 54. This screen characterizes a single source fold and one seed. It leaves local morphological cues and patient-level repeatability unresolved. [Computed results](results/contrast-selection.json)

The [contrast-only blinded packet](runs/ablation-fold0/contrast-case-review/blinded-cases.pdf) samples four gains, four losses and four nearly unchanged cases at random within each stratum, excluding cases previously shown in the two-arm panels. Record appearance on the first two pages before comparing the randomized contours. This is an enriched discussion sample. Use any identified cue to define a follow-up hypothesis for independent cases. [Response sheet](runs/ablation-fold0/contrast-case-review/response-sheet.csv)

Both new fits ran sequentially for 12 epochs on the local Apple GPU with the same original train/validation IDs, augmentation order, loss, optimizer, channel positions, initialization and threshold. The contrast fit took 20 min 04 s of epoch time; the gradient fit took 20 min 22 s. The monitored interval including checkpointing and re-inference was 41 min 15 s. macOS reported nominal thermal state and AC power at all 163 samples. Peak sampled PyTorch Metal driver allocation was 2.34 GiB. These are operating-system status observations and allocator readings. They do not measure chip temperature.

Reloading each checkpoint reproduced all 576 masks exactly. Both original arms were rescored and matched their saved per-image metrics. [Protocol](ABLATION-PROTOCOL.md) · [Full paired results and run fingerprints](results/channel-ablation.json)
