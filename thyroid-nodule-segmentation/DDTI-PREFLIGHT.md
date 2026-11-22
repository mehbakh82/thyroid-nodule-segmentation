# DDTI external evaluation — proposal for the supervisor meeting

**Decision to make with Dr. Amiri.** Which failure would a useful thesis method need to reduce, and by how much? Use the blinded TN3K cases to choose between wrong-target selection, missed nodule tissue, and inclusion of adjacent tissue. Once the endpoint is chosen, record the method and primary comparison before first external inference.

## Cohort prepared on CPU

I indexed the 256 × 256 US-Segmentator DDTI mirror and verified its archive against the published SHA-256. Its 637 paired entries contain 630 distinct decoded RGB images and 634 distinct image–mask pairs. Of the images, 626 have one distinct mask and four have two. All 637 alternate double-extension image filenames decode identically to their corresponding primary files. Row multiplicities and mask differences are preserved in the local map.

For the primary image-level analysis I propose one prediction per unique image. Score it against each of that image’s distinct references, average those scores within the image, then give each of the 630 images equal weight. This avoids picking whichever mask happens to favor a model. For a sensitivity check, repeat on the 626 single-reference images. The origin of the four mask differences is unknown. Distinct images may still be different views of the same patient. A claim about independent patients needs reliable case or patient linkage. Confirm that the lab intends this mirror; a different release needs its own manifest.

## Decisions to record before scoring

- Specify an automatic, unprompted grayscale reference model and one prespecified challenger. If the reader review shifts the thesis toward localization, define how to count a wrong nodule separately from partial overlap. If it favors boundary accuracy, define the boundary tolerance and how a complete miss is scored. Report the denominator for each endpoint.
- Preserve the source model’s channel order, normalization, aspect-ratio policy, fixed 0.5 threshold and postprocessing. DDTI is already resampled in this mirror. Report boundary distances in pixels on a declared grid. Physical-distance interpretation requires reliable pixel spacing.
- Report paired image-level changes, complete misses and the single-reference sensitivity. Keep all references of a given image together when resampling. Group by verified patient IDs if a future release supports that unit.

The external model predictions and scores are still unopened. The artifact prepared here is a cohort index. Regenerate it from the project directory with `python ddti_manifest.py`. [Audit receipt](results/ddti-preflight.json) · [Local image and reference map](local/ddti-preflight/image-reference-map.json) · [TN3K blinded review packet](runs/ablation-fold0/contrast-case-review/blinded-cases.pdf)
