# Third-party sources

The ultrasound examples and reference contour overlays in the study figures derive from the TN3K dataset of Haifan Gong and colleagues. The author-published dataset card at revision `c0479e921fc9b7afcf1e8ca9b296a408e06a5bb9` designates the MIT license. The accompanying project identifies the copyright holder below. This notice applies to the included TN3K examples and the 12 TN3K images in the blinded reader kit. [Dataset](https://huggingface.co/datasets/haifan-gong/TN3K/tree/c0479e921fc9b7afcf1e8ca9b296a408e06a5bb9) · [Paper](https://doi.org/10.1016/j.compbiomed.2022.106389) · [Publisher's repository license](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation/blob/a2f11b4f3e5cb79b85d1472b6ba489499eef47e7/LICENSE)

MIT License

Copyright (c) 2021 haifan gong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The DDTI material in this repository comprises research metadata, hashes, counts and an image/reference map. For source images, provider terms and lineage see the [dataset audit](msc-readiness/thyroid-evidence/DATASETS.md) and [pinned sources](msc-readiness/thyroid-evidence/sources.json). The four literature readings and other works are cited by DOI in the accompanying notes. The trained models used torchvision's ResNet34 ImageNet V1 initialization. [Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)

The TNUI-2021 development experiments use the authors’ training partition at a pinned commit. The TN-SCUI2020 experiments use the labeled training release and a checksum-verified mirror. The repository publishes study outputs and provides code to read the source images and masks and generate example overlays locally. [TNUI authors](https://github.com/zxg3017/TNUI-2021-/tree/5fa417629f581d82017eed535cdd62586717b669) · [TN-SCUI2020 organizers](https://github.com/mtv2020/TNSCUI2020) · [Pinned archive listing](https://huggingface.co/datasets/us-segmentator/us-segmentation-dataset/tree/6e7dadb8681fa5a137a1a607127510e36235610d/zips/Thyroid)
