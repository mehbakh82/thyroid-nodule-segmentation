# Reproduce the thyroid evidence checks

Create `local-data/` in this directory and place the two provider archives there as `TN3K.rar` and `ddti.zip`. The public download URLs, revisions, archive sizes and checksums are in [sources.json](sources.json). This is a data and prior-art audit. No source-study software or model weights were executed.

## File checks and development experiments

Run from this directory using Python with NumPy, SciPy, Pillow and scikit-image. The figure also uses ReportLab. Ensure the archives match the SHA-256 values in the source manifest before extracting.

```sh
mkdir -p local-data/tn local-data/dd
bsdtar -xf local-data/TN3K.rar -C local-data/tn datasets/tn3k
bsdtar -xf local-data/ddti.zip -C local-data/dd
python inspect_release.py --tn3k local-data/tn/datasets/tn3k --ddti local-data/dd/ThyroidNodule-DDTI --out file-audit.json
python near_copies.py local-data/tn/datasets/tn3k local-data/dd/ThyroidNodule-DDTI near-copy-audit.json
python mask_sensitivity.py local-data/tn/datasets/tn3k mask-sensitivity.json
python edge_probe.py local-data/tn/datasets/tn3k development-edge-probe.json
python meeting_figure.py local-data/tn/datasets/tn3k/trainval-mask reference-mask-agreement.pdf
```

Exact image identity includes native dimensions and all decoded RGB pixel bytes. Mask identity in the binary comparison uses per-mask half-maximum. The DDTI alias check compares each double-extension file against its one-extension counterpart. A positive duplicate check shows the same decoded pixels; a negative check cannot establish different patients. The near-copy screening uses a difference hash followed by thumbnail correlation, without flipping, alignment, panel extraction or crop registration. Groups with conflicting masks require a provenance decision before splitting. A duplicate must travel with its group.

The visual marker spot check sorts training image filenames by SHA-256 of the UTF-8 string `thyroid-reading-sample-v1:` followed by the filename and takes the first 24. The reviewer inspected thumbnails. Obvious crosses or measuring lines were visible at positions 6, 10, 15, 20 and 21 in this ordered sample. Faint or ambiguous marks were not counted. No patient-level or site-level prevalence estimate follows.

## Published benchmark

Use the immutable revision linked in [benchmark-check.json](benchmark-check.json). Download `per_image/ood_summary.csv`, `manifests/tucc_tn3k_ddti/ddti_panel_map.csv` and `manifests/tucc_tn3k_ddti/splits.csv`. Verify the SHA-256 values recorded there; the local name `splits-recent.csv` refers to the third file. Filter the first table to source TN3K and target DDTI and retain all strategies and architectures. From the split table filter dataset DDTI and count rows and distinct case IDs per split. Check that the three case-ID sets are disjoint. All results quoted are from aggregate files; no external patient-level model scores were inspected.

## Literature retrieval

`python search.py` queries Europe PMC. `python crossref_search.py` then runs the complementary Crossref search. Both apply the 21 November 2026 publication boundary. The Crossref results are relevance-ranked and truncated at 75 per query. Rerunning later may change counts because databases update indexing. Keep the linked snapshots as the dated record. Screening emphasized direct transfer, annotation provenance, boundary/marker mechanisms, independent sites, public data or code, and consequences for deployment. Detection/classification-only records and incidental matches remain visible in the discovery files. No individual relevance or risk-of-bias ratings are asserted for the entire retrieved set.

The supplemental targeted searches covered dataset aliases and primary author repositories, DDTI XML versus processed derivatives, TN3K→DDTI and reverse transfer, contour/edge-guided models, caliper removal and guidance, augmentation, multireader annotation, SAM/SAMUS/US-43d exposure, clinical negative frames, Iranian thyroid datasets, and institutional external validation. Findings and primary-source links are in the dataset atlas and prior-art map. No gated dataset terms were accepted.
