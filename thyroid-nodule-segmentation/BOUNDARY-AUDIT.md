# Boundary sensitivity on the existing TN3K validation predictions

The average advantage of the contrast-and-gradient model is small under both audited reference-mask conventions. Some individual contour differences persist through every scoring choice tested. Those cases provide a concrete starting point for the supervisor discussion.

## Numerical sensitivity

Both saved models were scored on the same 576 validation images. The reference JPEG was decoded on its native grid, then resized by nearest neighbor onto the saved prediction grid. The alternative retained only pixels equal to 255, matching the effect of dividing by the maximum and casting to integer on this archive. Source mask checksums, prediction checksums, image order and agreement with every original primary score were checked.

| Reference pixels kept | Grayscale mean Dice | Three-channel mean Dice | Gain in region Dice points | Mean surface Dice gain at 1 / 2 / 3 pixels, points |
|---|---:|---:|---:|---:|
| Above 127 (original) | 0.8150 | 0.8167 | +0.18 | +0.73 / +0.50 / +0.26 |
| Exactly 255 | 0.7816 | 0.7835 | +0.19 | +0.38 / +0.48 / +0.49 |

Among the 521 images where both primary region Dice scores reached 0.50, the original two-pixel surface Dice metric showed 67 improvements and 70 deteriorations of at least ten points. Requiring both models to reach 0.50 under both decoders retained 517 images. Requiring the ten-point difference in the same direction at all three tolerances and both decoders retained 15 improvements and 19 deteriorations. With the less stringent requirement of preserving the direction alone, 57 of the original 67 improvements and 56 of the original 70 deteriorations remained. Losing the ten-point classification can mean a smaller effect of the same sign. Pixel tolerances refer to the 224-pixel long-side grid.

## Supervisor review

The [12-case packet](runs/source-fold0/boundary-audit/blinded-cases.pdf) shows the source image and two contours with identical styling. Model assignment is randomized per case and balanced across the packet. Each reviewer can record preference, the main concern, and the location and importance of the correction on paper or in the [response sheet](runs/source-fold0/boundary-audit/response-sheet.csv). Ask which contour the reviewer would keep and why before showing the reference or revealing model identity. Another option is to flag an image as ambiguous.

This was a post-hoc development analysis. The packet was deliberately sampled from persistent improvements and deteriorations, scoring-sensitive cases, little-change cases and shared misses. The nine examples from the earlier meeting panel were excluded. These 12 images suit discussion of error types; they cannot estimate their prevalence or establish a clinical model preference.

## Replay

Run `python boundary_audit.py /path/to/datasets/tn3k/trainval-mask` in the evaluation environment, then `python reader_packet.py` in the PDF environment. The aggregate output is in [boundary-sensitivity.json](results/boundary-sensitivity.json). The local keys and responses belong to the review directory. All computations in this audit use CPU and existing development predictions.
