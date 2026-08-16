# RESULTS — HER2 Pathology Extension

**Status: pre-registered analysis plan. No training run has been executed yet.**

This file is written *before* training deliberately, so the reporting standard is
fixed in advance and cannot be adjusted to flatter whatever comes out. Sections 3
onward contain no numbers. They will be filled in from the runs described in
`README_her2.md` Sections 7–8, and this notice will be replaced with the run
metadata.

## 1. What was built and verified

| Component | Status |
|---|---|
| Gap analysis over HEROHE / TCGA-BRCA / CAMELYON / BCI | Complete — `README_her2.md` §1 |
| `her2` dataset format, matching the `tn5000`/`segthor` convention | Complete |
| `--dataset her2` / `--her2_dir` wired into `main.py` | Complete |
| BCI IHC conversion pipeline (DAB deconvolution → boxes, templated captions) | Complete, exercised end-to-end on synthetic patches |
| CAMELYON polygon conversion pipeline | Complete, **not yet run against real slides** |
| HEROHE / TCGA-BRCA slide-label export | Complete, **not yet run against real clinical files** |
| Decoder token-budget analysis | Complete — measured against the real gemma-3-270m-it tokenizer |
| Frozen-encoder feature cache + laptop smoke test | Code complete and unit-tested, **never executed against real checkpoints** — see `README_her2.md` §7b |
| Shuffled-image control (`--shuffle_images`) | Implemented and unit-tested. It was prescribed in §6 below but had no implementation until now |
| Colab notebook running the IHC track end to end | `notebooks/her2_colab_arm_a.ipynb`, **never executed** |
| Test suite | 98 tests passing; 89 need no torch / GPU / downloaded data, 9 cover LoRA merging and need torch |
| Fine-tuning run | **Not run** — see §2 |
| Evaluation | **Not run** — see §2 |

### Finding worth reporting upstream, independent of any training run

Boxes are serialised into the decoder target as literal text (`det.py:44` and its
siblings), so the sequence cost grows linearly with box count: **+20 tokens per
box**, measured on the gemma-3-270m-it tokenizer. The intercept depends on the
caption text (185 tokens at one box for a HER2 template, 194 for a PadChest-length
sentence, 175 for a one-word SegTHOR label); the slope does not.

`README.md`'s downstream recipe uses `--decoder_max_length 150` with
`--multimodal_tokens 128`, and `main.py:317` adds the two, giving a budget of 278.
Measured directly, that fits **5 boxes** (6 for a one-word label). Beyond it,
`pad_or_truncate` (`lofi_utils/gemma.py:520`) cuts from the right — removing the
closing code fence and `<end_of_turn>` — with **no exception and no warning**, so
the decoder is trained on a target that never terminates. The pretraining recipe
uses `--decoder_max_length 200 --multimodal_tokens 256` (budget 456) and is not
affected.

Reproduce in under a minute against a fresh clone:
`python tools/repro_decoder_truncation.py --tokenizer_path ./models/gemma-3-270m-it/tokenizer.json`.
It imports the repo's own `pad_or_truncate` (falling back to extracting it
verbatim from source if torch is absent) and prints the truncated output directly.

**How this should be framed when reported.** No box-count cap, assertion, or
filter exists anywhere in their pipeline, so nothing prevents the case — but
whether their published downstream numbers were affected is **unverified here**,
because it depends on the box-count distribution of PadChest-GR / TN5000 / SegTHOR
and none of those are downloaded in this environment. SegTHOR and TN5000 plausibly
stay under the limit (few organs/nodules per slice, and `seg_slice_to_boxes` drops
components under 4% of area, merges overlaps, and removes nested boxes). PadChest-GR
takes its boxes uncapped from the grounded-report JSON and has the longest captions,
so it is the most likely to cross — but that is an inference, not a measurement.
This should therefore be reported as **a real, reproducible latent defect of
unknown impact on their results**, with the request that they check the
distribution themselves, not as a claim that their numbers are wrong. Neither of
the repository's two open issues mentions it.

The HER2 recipes use `--decoder_max_length 200` (budget 328, fits 8); the pairing
is enforced by `her2/boxes.py` helpers and pinned by tests.

## 2. Why no numbers yet

Phases 3 and 4 need three things this environment does not have:

1. **The source data.** BCI (~10 GB) and CAMELYON16 (~700 GB of WSIs) both
   require download from their hosts; CAMELYON also needs OpenSlide binaries.
2. **The checkpoints.** `myeongkyunkang/lofi-medg` plus
   `google/gemma-3-270m-it`, and either the base or merged SigLIP2 encoder.
3. **A GPU.** The development machine has an Intel UHD 620 integrated adapter and
   7.8 GB of system RAM, and no CUDA device.

   Note the constraint is wall-clock, not memory. With `--fix_enc` the encoder
   carries no gradients, no optimiser state and no stored activations, so the
   recipe fits comfortably in 16 GB; what does not fit is running a
   400M-parameter encoder over 1024 patch tokens per image on a CPU. §7b of
   `README_her2.md` removes ~97% of that cost by caching the frozen encoder's
   output once instead of recomputing it for all 30 epochs, which brings the run
   into free-Colab-T4 range. It does not make a CPU-only run practical.

Note also that `google/gemma-3-270m-it` is a gated HuggingFace repository and
needs licence acceptance before download; `myeongkyunkang/lofi-medg` (a single
`last.pt`) and the SigLIP2 encoders are ungated.

The pipeline is complete and tested up to that boundary. Everything in §3–§6 is
one `preprocess_bci.py` run plus two `main.py` invocations away once the data and
a GPU are available.

## 3. Dataset composition *(to fill in)*

| Track | Train | Val | Test | Images | Samples |
|---|---|---|---|---|---|
| BCI IHC (`her2-ihc-templated`) | | | | | |
| CAMELYON (`lesion-polygon-real`) | | | | | |

Per-grade counts, DAB-fraction distribution and the number of patches dropped for
having no localisable staining come from `her2_512p/her2_meta.csv`.

## 4. Training *(to fill in)*

Arm A (frozen encoder, paper recipe) and Arm B (true continual training, resumed
checkpoint), curves via `tools/plot_her2_training.py`. Record for each: final and
best loss, best epoch, wall-clock, peak GPU memory from `gpu_log.txt`.

## 5. Evaluation *(to fill in)*

Mono-class P/R/F1 at IoU 0.5, **reported per track, never pooled**:

| Model | CAMELYON track F1 | BCI IHC track F1 |
|---|---|---|
| `lofi-medg` zero-shot (baseline) | | |
| Arm A | | |
| Arm B | | |
| Shuffled-image control | | |

No pooled column is provided, deliberately. The two tracks do not measure the
same thing — one scores against pathologist polygons, the other against our own
DAB threshold — so a combined figure has no interpretation, and a number that
exists in a table will eventually be quoted as if it did. Per-track breakdown
comes from `tools/her2_eval_breakdown.py`, which splits on the `content_type`
recorded per sample.

Plus qualitative `grounding_demo.py` output on held-out HER2 examples.

## 6. Honest interpretation — the standard fixed in advance

These constraints apply to whatever the numbers turn out to be:

- **The BCI track score is not a HER2 detection result.** Its targets are boxes
  produced by our own DAB threshold. A high score means the model reproduced that
  heuristic. It will be described that way, without exception.
- **The CAMELYON track score is a real grounding result**, but for metastasis
  localisation on H&E — not for HER2.
- **If continual training does not beat the zero-shot baseline, that is the
  finding** and it will be reported as the headline, not buried. A plausible and
  publishable outcome here is: *the weak-supervision bridge was not strong enough
  to teach HER2 semantics, and templated captions over heuristic boxes mostly
  taught the decoder a seven-string text prior.*
- **If the shuffled-image control scores close to the real evaluation**, the model
  is not using the image, and every other number in this file is void. That check
  gates all of the rest.
- No claim of clinical HER2 scoring capability will be made from this dataset
  under any result.

## 7. What would actually move this forward

Region-level HER2 annotation on IHC slides — a pathologist marking membrane
staining regions with per-region intensity — does not exist publicly. That, or
the authors' forthcoming pathology-pretrained release, is the real unlock. The
format and tooling here transfer to either unchanged.
