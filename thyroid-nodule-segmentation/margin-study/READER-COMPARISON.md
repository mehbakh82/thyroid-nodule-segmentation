# Independent tracing comparison

**23 November 2026 · Analysis specified before receipt of reader tracings.** The question is whether the direction of a one-pixel boundary correction agrees with an independent tracing, and how often the scan itself supports a clear target. The kit contains 12 development images balanced by relative nodule size, selected from cases already localized by the baseline. All findings will be descriptive and conditional on that selection.

The two audited public TN3K releases contain byte-identical development masks. The availability of earlier contours or polygons remains unconfirmed. [Source check](reference-source-check.json)

## For the reader

Open only the [image-only tracing kit](reader-kit.zip). A reader with experience delineating thyroid nodules on ultrasound should complete it independently, before seeing model overlays or either reference mask for these cases. For each image, record `traced`, `target_unclear`, or `untraceable` in the supplied CSV. Use `target_unclear` when more than one nodule might be intended, and `untraceable` when a plausible target has insufficient boundary evidence. For a traced case return a native-size binary mask and note the location of any uncertain arc. Please identify any previous exposure to these images or their annotations to the study coordinator.

A lossless grayscale PNG with values 0 and 255 is the simplest return format. Use the supplied IDs, e.g. `case-01-mask.png`. A native-coordinate polygon with its tool and coordinate convention can also be returned: the coordinator should review its rasterization at native resolution before scoring. A PDF screenshot, a hand-selected zoom or a resized mask cannot establish the alignment. Return the observations and masks locally to the investigator; store each reader's originals separately.

## For the investigator

1. Use the private case key only after responses have been recorded. Validate IDs, mask binary encoding, dimensions and image fingerprints. Verify alignment visually for returned masks, particularly after polygon conversion. All 12 rows should have a disposition, including the ambiguous cases. For an interim return, report which rows remain pending.
2. On the native grid, compare each original JPEG decoding (>127 and ==255) against the reader's binary mask. Summarize per-image foreground Dice and signed relative area error for both. Do not interpret one reader's preference as anatomical truth.
3. For the model comparison, use the **same unpadded 224-long-side grid as the original experiment**. Transform the reader mask and the two original references by nearest neighbor. Compare the frozen grayscale-flip prediction and its fixed one-pixel erosion to each of the three references. The primary diagnostic is the paired change in two-pixel surface Dice from erosion, reported per case and averaged equally across traced cases. Report ties and direction reversals. The decoder with the higher native overlap and the reference yielding a higher score for a shrunken prediction answer different questions.
4. Preserve ambiguous, untraceable and pending counts in the denominator. Do not assume an ambiguous case has zero Dice. Report numbers by size stratum and the observed differences descriptively. Keep free-text remarks in the reader's original sheet; they need qualitative interpretation to identify recurrent arcs. Confirm any resulting hypothesis on fresh, patient-linked data before an external comparison.

Investigators can regenerate the kit and its local matching key after the margin study with `python margin-study/tracing_packet.py /path/to/tn3k/trainval-image`. Keep the investigator key separate from the reader's materials.

Run from the thyroid project directory with Python, NumPy, SciPy and Pillow:

```sh
python margin-study/score_tracings.py /path/to/reader-return /path/to/tn3k/trainval-mask --out /path/to/new-private-result.json
```

The return directory contains `observations.csv` and any masks at its top level. The result records hashes of the evaluated originals and each submitted PNG. Each invocation writes to a new output path. Patient-level confidence intervals and clinical preference estimates require a suitable sampling design.
