# What a second dataset adds to the thyroid margin study

**23 November 2026.** Use TNUI-2021 training data for acquisition and reference-format development, ThyroidXL for a future patient-grouped external comparison if available, and the agreed DDTI release for its frozen test. Continue independent tracing of TN3K in parallel. Roles should be decided before pooled training or external score inspection.

## Controlled reference experiment

The sample comprised 48 of the 966 public TNUI training images, preselected by salted hash of filename before inspecting image or label contents. All 96 image/mask files matched their pinned source Git blob IDs. All 48 pairs were 780 × 780, with a two-valued source mask, and passed size and decoding checks. There were no exact native grayscale matches within the sample or to the 2,879 TN3K development images. This cannot establish independent patients or rule out near duplicates. No TNUI validation or test images were scored. [Author dataset at the audited revision](https://github.com/zxg3017/TNUI-2021-/tree/5fa417629f581d82017eed535cdd62586717b669)

We started from the released PNG binary mask and deliberately compressed a 0/255 copy with the luminance quantization table shared by all 2,879 TN3K development masks. We held the TN3K flip-averaged grayscale model, image, model output, one-pixel erosion and scoring grid fixed, and changed only the evaluation reference. All 48 eligible images enter the primary analysis, including misses. The two-percentile limits below are from paired image bootstrap draws and condition on these cases and the trained checkpoint. [Pre-specified experiment](PROTOCOL.md) · [Numerical results](results.json)

| Reference used for scoring | Mean region Dice, unchanged model | Effect of shrinking the prediction, surface-Dice points at two pixels |
|---|---:|---:|
| Released binary PNG | .4323 | +2.48 (−0.08 to +5.22) |
| JPEG, foreground >127 | .4323 | +2.48 (−0.08 to +5.22) |
| Same JPEG, foreground exactly 255 | .4231 | +4.27 (+1.89 to +6.89) |

The strict decoding increased the apparent erosion benefit by **1.79 points** (paired interval +1.08 to +2.59). Six images changed sign. The cohort mean favored erosion numerically under both original PNG and strict decoding. The direction reversal at cohort level seen on TN3K did not replicate. This control shows that an encoding choice *can amplify* a claimed improvement. It does not recover the original TN3K annotation, establish how every JPEG was created, or show that a radiologist prefers smaller outlines.

## Exploratory resolution check

The median fraction of image occupied by the nodule was 1.69%, compared with 8.57% in TN3K development. Held the field of view, checkpoint and threshold fixed and doubled the input long side to 448, then bilinearly resampled the foreground probability onto the original unpadded 224 evaluation grid before thresholding. Mean Dice decreased from .4323 to .3525 (−7.98 points, paired image interval −14.25 to −1.88). Mean surface Dice at two pixels decreased from .2846 to .1803. Although complete misses decreased from two to one, the aggregate delineation worsened. This was an outcome-informed follow-up on the same 48 images. The falling score at 448 is specific to this frozen model and inference change. Resolution-aware training and image framing are different interventions. [Follow-up definition](SCALE.md) · [Scale results](scale-results.json)

## Recommended next experiments

1. **Diagnose localization versus contour error on new TNUI development images.** Lock a fresh hash-selected subset from the remaining training files. Compare the same frozen model on full images, a fixed central crop, and a same-size crop centered on the reference centroid. Use the last arm only as a diagnostic for a correctly supplied center. Keep crop size independent of lesion size, project outputs to the common frame, and report full-image misses, sensitivity, excess foreground and boundary agreement. A matched central crop helps isolate magnification from the benefit of a correct center. An oracle crop alone cannot establish deployment performance. Advance to an automatic detector/gland-guided crop only if a supplied center creates a worthwhile and repeatable gain. Keep expert tracing as the test of anatomical correctness.
2. **Assess reference agreement independently of model performance.** Wu et al. describe TNMIX as TN-SCUI and DDTI images labeled by five graders. Verify availability of the *individual* masks, grader identities within image and source mapping before using this route. No individual-reader public file set was verified here. Because DDTI is included, analyzing it for development would compromise the planned external role. Seek the TN-SCUI portion, split by verified patient and retain every grader for each case together. Use the existing 12-case TN3K reader kit if the individual masks cannot be obtained. [Authors' methods, section 4.1](https://arxiv.org/pdf/2208.03016) · [Published paper](https://pubmed.ncbi.nlm.nih.gov/38669168/)
3. **Use one independently acquired patient-linked cohort after method selection.** ThyroidXL has explicit patient partitions (9,541 training images from 3,354 patients; 2,094 test images from 739 patients) and the authors describe a clean, caliper-free release. Use the provider's access process. Adjudication from multiple readers is different from release of all reader-specific contours. A clean-center comparison measures transport across several changes together; to study calipers specifically, use matched images and sham edits, e.g. TUD with its documented inpainting procedure. TN-SCUI training is an alternative for broader development, with organizer-reported binary annotations for 3,644 images of 3,644 patients. [ThyroidXL authors](https://github.com/VNPT-AI-Official/ThyroidXL) · [Access terms](https://huggingface.co/datasets/VNPT-AI/ThyroidXL) · [TN-SCUI organizers](https://github.com/mtv2020/TNSCUI2020) · [TUD](https://pubmed.ncbi.nlm.nih.gov/38267314/)

## Replay

From the thyroid project folder, run the following in the CPU PyTorch environment used for the frozen study. The final positional argument to the codec script is the extracted author TN3K dataset folder. Downloads are staged under the ignored local directory.

```sh
python margin-study/tnui-control/fetch_sample.py local/tnui-control
python margin-study/tnui-control/codec_control.py local/tnui-control /path/to/datasets/tn3k
python margin-study/tnui-control/scale_check.py local/tnui-control
```

Source provenance and sample fingerprint are recorded in the score receipt. Both scripts use the original single fold-0 grayscale checkpoint and CPU inference. There is no fitted TNUI component.
