# Third party notices

This repository implements anatomical segmentation followed by strict predicted-ROI binary and direct three-class classification. Its model adapters use the components below. Upstream licenses and copyright notices remain applicable to those components; this release does not assign a new blanket license to the authors' study-specific code.

| Component | Pinned source | Use in this repository | License record |
|---|---|---|---|
| Ultralytics YOLO26 | `ultralytics==8.4.138` | Installed dependency called by the YOLO adapter; upstream source archives and weights are not bundled | [AGPL-3.0](licenses/ULTRALYTICS_AGPL-3.0.txt) |
| EMCAD | `SLDGroup/EMCAD`, commit `26c9c31f731f749b62c5fe83f44376dac75f3aa8` | User-provided local source, accessed through `EMCAD_SOURCE_DIR` | [UT Austin Research License](https://github.com/SLDGroup/EMCAD/blob/26c9c31f731f749b62c5fe83f44376dac75f3aa8/LICENSE) |
| SAM2-UNet | `WZH0120/SAM2-UNet`, commit `01598e5e9912ffb23f965ecbebf4d1dfecbaa56e` | Prompt-free adapter, U-shaped decoder blocks and Hiera integration | [Apache 2.0](licenses/SAM2_UNET_APACHE-2.0.txt), with the U-Net ancestry notice below |
| Meta SAM2/Hiera | Source included by the pinned SAM2-UNet revision; independent license reference `facebookresearch/sam2` commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Hiera backbone and utilities | [Apache 2.0](licenses/META_SAM2_APACHE-2.0.txt) |
| PVT-v2 | `whai362/PVT`, commit `57e2dfaa5a46f9050d76f306a4fcd9a7c061f520` | PVT-v2-B0 model family and official pretrained initialization; this release obtains its executed EMCAD-associated implementation from the user's EMCAD source | [PVT Apache 2.0](licenses/PVT_APACHE-2.0.txt); the EMCAD source remains subject to its separate license |
| timm | `timm==1.0.19` | Installed model utilities | [Apache 2.0](licenses/TIMM_APACHE-2.0.txt) |
| torchvision | `torchvision==0.23.0` | Independently fitted ResNet-18 secondary classifiers and image transforms | [BSD 3-Clause](licenses/TORCHVISION_BSD-3-Clause.txt) |

## Attribution and local adaptations

Meta source files retain the notice “Copyright (c) Meta Platforms, Inc. and affiliates.” The timm license retains the Ross Wightman copyright notice. SAM2-UNet adaptations select the prompt-free Hiera-Tiny path and the required decoder blocks; local import and activation-checkpoint changes are identified in the affected files. Original authorship is not transferred by these adaptations. Source identities and acquisition checks are recorded in [third_party_dependencies.json](provenance/third_party_dependencies.json).

The SAM2-UNet `DoubleConv`/`Up` helper lineage includes links to `xiaopeng-liao/Pytorch-UNet` (now `milesial/Pytorch-UNet`), commit `8ebac70e633bac59fc22bb5195e513d5832fb3bd`, and the associated padding-fix example in `HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy`, commit `0e854509c2cea854e247a9c615f175f76fbb2e3a`. These source links are retained in `sam2_unet_blocks.py`. The linked Pytorch-UNet revision carries [GPL-3.0](licenses/PYTORCH_UNET_GPL-3.0.txt). The SAM2-UNet repository-level Apache declaration does not remove those inherited notices or establish an Apache-only license for every inherited portion. This record preserves that ancestry rather than relicensing it.

The custom ViT Method2 port derives from an author-supplied research notebook. The source notebook SHA-256 is recorded in the dependency manifest. Its notebook archive is not distributed, no external pretrained ViT initialization is claimed, and no third-party redistribution grant or new license is inferred from its inclusion in the study.

## Using EMCAD

Obtain the pinned EMCAD checkout directly from its [official repository](https://github.com/SLDGroup/EMCAD/tree/26c9c31f731f749b62c5fe83f44376dac75f3aa8) under terms applicable to your use. Set `EMCAD_SOURCE_DIR` to that checkout's root, which contains `lib/decoders.py` and `lib/pvtv2.py`. The release adapter checks both source hashes before importing them and does not download EMCAD code automatically. The exact expected hashes are in the dependency manifest.

For a PowerShell session, a relative local checkout can be selected with:

```powershell
$env:EMCAD_SOURCE_DIR = (Resolve-Path '../EMCAD').Path
```

The UT Austin Research License permits specified noncommercial research use but restricts redistribution of its source and derivatives. Accordingly, EMCAD decoder/PVT source, original archives and the earlier adapted EMCAD wrapper are not bundled. Retain the upstream license and required notices in your own licensed checkout. The reference to PVT's separate Apache license does not relicense modifications distributed through EMCAD.

## Dependencies and weights

Package versions are recorded in the supplied requirements files. Installed dependencies retain their own licenses; this notice is not a replacement for those terms. In particular, the Ultralytics dependency retains its AGPL obligations. Source and model-weight rights are distinct: official initialization files are obtained separately from their pinned upstream URLs and verified against recorded SHA-256 hashes. Clinical-study-trained checkpoints, private data and original run archives are not part of this source package.
