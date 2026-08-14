'''
HER2 immunohistochemistry extension for LoFi-MedG.

This package builds an image-text-box grounding dataset (`her2`) for HER2 breast
pathology from public sources only. It follows the annotation conventions used by
the existing `tn5000` / `segthor` datasets: a `<name>_512p/` directory holding
letterboxed 512x512 JPEGs under `train/`, `val/`, `test/`, plus one
`<split>.json` manifest per split.

See README_her2.md for data sources, licenses, and an explicit statement of which
annotations are real and which are templated weak supervision.
'''
