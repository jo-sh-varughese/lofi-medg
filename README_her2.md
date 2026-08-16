# HER2 Pathology Extension

An extension of LoFi-MedG to breast HER2 pathology, contributed as a community
follow-up to the authors' note that histopathology is not in the MedG training
set. It adds one new dataset, `her2`, built **only from public data**, and reuses
the existing training, evaluation and grounding-demo paths unchanged.

> **Read this section before reading any number this pipeline produces.**
> Part of the supervision here is templated weak supervision, not pathologist
> annotation. Section 2 states exactly which part, and Section 8 states what that
> means for how results may be interpreted.

---

## 1. Why this dataset had to be built the way it is

LoFi trains on `(image, text, box)` triplets. A HER2 grounding fine-tune would
ideally need triplets of the form *(IHC image, membrane region, HER2 staining
intensity text)*.

**No public dataset contains those triplets.** This is a real annotation gap, not
a search failure:

| Source | Images | Native annotation | Usable as HER2 grounding? |
|---|---|---|---|
| HEROHE | H&E WSIs | Slide-level binary HER2 (0/1+ vs 2+/3+) | No — no regions, and no IHC pixels |
| TCGA-BRCA | H&E WSIs | Clinical `her2_status_by_ihc`, subset with 0/1+/2+/3+ | No — the score was read on IHC glass that is not in the image archive |
| CAMELYON16/17 | H&E WSIs | **Real pathologist polygons** (ASAP XML) | Boxes yes, HER2 no — the label is "metastasis" |
| BCI | **IHC patches** | **HER2 grade per patch** (0/1+/2+/3+) | Grade yes, regions no |

HEROHE and TCGA-BRCA are H&E cohorts whose HER2 status comes from a *separate
assay on different glass*. Running DAB stain deconvolution on them would measure
haematoxylin and eosin, not HER2, so a caption asserting membrane staining over
an H&E region would describe something the pixels do not contain. **We therefore
never use HEROHE or TCGA-BRCA as grounding captions.** They are exported to a
flat CSV for a slide-level classification probe instead (Section 6).

The dataset is built from the two sources where the claim and the pixels line up:

- **BCI** — real IHC pixels with a real HER2 grade for exactly those pixels.
  Regions are localised by DAB colour deconvolution, which measures the very
  chromogen that visualises HER2. The box is heuristic; the grade is not.
- **CAMELYON16/17** — real pathologist polygons, with generic tumour captions
  that make no HER2 claim at all. This track supplies trustworthy region geometry
  in histopathology, which the templated track cannot.

## 2. Supervision honesty table

| Track | content_type | Image | Box | Caption |
|---|---|---|---|---|
| BCI IHC | `her2-ihc-templated` | Real IHC patch | **Heuristic** — DAB connected components; whole-tissue for HER2 0 | **Templated**, instantiated from the real per-patch BCI grade |
| CAMELYON | `lesion-polygon-real` | Real H&E tile | **Real** — pathologist polygon bounding box | **Templated**, instantiated from the real annotation group; states no HER2 |

Nothing in this pipeline is model-generated. Every caption is one of seven fixed
strings defined in `her2/captions.py` and printed by `tools/visualize_her2.py`:

```
 HER2 0  : no membranous staining, HER2 0 negative
 HER2 1+ : faint incomplete membranous staining, HER2 1+ negative
 HER2 2+ : weak to moderate complete membranous staining, HER2 2+ equivocal
 HER2 3+ : strong complete membranous staining, HER2 3+ positive
 lesion  : metastatic carcinoma in lymph node tissue
 lesion  : benign lymph node tissue
```

Wording follows the ASCO/CAP HER2 IHC scoring criteria (Wolff et al. 2018;
0 vs 1+ per the 2023 update). Each caption names the visible staining pattern
first and the score second, so it reads correctly in both directions of the LoFi
objective.

Every emitted sample is logged to `her2_512p/her2_meta.csv` with its source, its
DAB-positive fraction and which threshold produced its box, so any sample can be
audited without re-running the pipeline.

## 3. Data sources and licensing

Download under `./data/`. Verify the current terms at each source before
redistributing anything — several require registration and are research-use only.

| Source | Where | Terms (verify at source) |
|---|---|---|
| BCI | https://bupt-ai-cz.github.io/BCI/ | Research use; non-commercial |
| CAMELYON16/17 | https://camelyon17.grand-challenge.org/Data/ | CC0 1.0 public domain dedication |
| HEROHE | https://ecdp2020.grand-challenge.org/ | Registration; research use |
| TCGA-BRCA | https://portal.gdc.cancer.gov/ | NIH GDC open-access data policy |
| Warwick HER2 contest (optional) | https://warwick.ac.uk/fac/cross_fac/tia/data/her2contest/ | Registration; research use |

**Only public datasets belong in this repository.** Every source above is
public and listed with its terms. Institutional or clinical data carries
different governance and sharing constraints and belongs in a separate private
repository; nothing in this pipeline reads from one, and `.gitignore` excludes
`data/` wholesale so slides cannot be committed by accident.

## 4. Dataset format

Identical to the `tn5000` / `segthor` convention, so no loader change was needed
beyond registering the dataset:

```
data/her2_512p/
├── train/  her2_bci_train_00042_3p.jpg, her2_camelyon_tumor_001_16384_8192.jpg, ...
├── val/
├── test/
├── train.json, val.json, test.json      # merged manifests read by the loader
├── train_bci.json, train_camelyon.json  # per-track manifests, so tracks rebuild independently
├── her2_meta.csv                        # provenance for every sample
└── her2_summary.json
```

```json
{
  "modality": "Histopathology",
  "data": {
    "train/her2_bci_train_00042_3p.jpg": [
      {"box": [[51, 102, 256, 307]], "label": "strong complete membranous staining, HER2 3+ positive"}
    ]
  }
}
```

- Boxes are **integer pixels on the letterboxed 512×512 canvas**, sorted by `x0`,
  exactly as `tools/preprocess_segthor.py` writes them. The loader divides by the
  `512` parsed out of the directory name and rescales to the 0–1000 bins the
  decoder emits as text — so **the directory must stay named `her2_512p`**.
- Each entry in the list under an image is one independent training sample.
- Images are longest-side-512 LANCZOS resizes centred on a black canvas.

`her2` deliberately falls into the bidirectional branch of
`lofi_utils/dataset/base.py` (the `else` case, as MedG itself does) rather than
the grounding-only branch used by the downstream datasets. Each sample is
therefore seen ~50% as *"Detect all instances of `<staining description>`"* and
~50% as *"Describe the regions: `<boxes>`"*. For HER2 the captioning direction is
the one that carries the clinical semantics, so both are wanted.

## 5. Building the dataset

```bash
cd tools/

# IHC track (primary)
python preprocess_bci.py \
    --bci_dir ../data/BCI_dataset/ \
    --output_dir ../data/her2_512p/

# Lesion-polygon track (needs OpenSlide)
python preprocess_camelyon.py \
    --slide_dir ../data/CAMELYON16/images/ \
    --annotation_dir ../data/CAMELYON16/annotations/ \
    --output_dir ../data/her2_512p/

# Merge tracks into the split files the loader reads, re-validating on the way
python merge_her2_manifests.py --output_dir ../data/her2_512p/

# Look at what was produced before committing compute to it
python visualize_her2.py --her2_dir ../data/her2_512p/ --split train --num 12
```

Splits are assigned by hashing a **slide or patient identifier**, never a patch
identifier, so tiles from one slide cannot straddle two splits. BCI's own test
split is honoured as our test split.

> **Known limitation.** BCI publishes patch ids, not slide ids, so within the BCI
> *train* pool we cannot prove the train/val carve-out is slide-disjoint. The
> held-out test set is BCI's own and is unaffected. Any val-based model selection
> should be read with that caveat.

Requirements beyond `requirements.txt`: `openslide-python` plus the OpenSlide
binaries, and `matplotlib` for the training curves — see `requirements_her2.txt`.
The IHC track, the tests and the validator need none of that.

## 6. Slide-level HER2 labels (a separate task, not grounding)

```bash
python preprocess_her2_slide_labels.py \
    --herohe_csv ../data/HEROHE/HEROHE_TrainGroundTruth.csv \
    --tcga_clinical ../data/TCGA-BRCA/clinical.tsv \
    --output_path ../data/her2/her2_slide_labels.csv
```

This emits slide-level labels for a classification probe on the encoder's
features. It is **not** wired into `--dataset her2` and never becomes a caption,
for the reason given in Section 1.

## 7. Fine-tuning from the released checkpoint

**Checkpoints.** `myeongkyunkang/lofi-medg` contains a single file, `last.pt`,
which is what `--resume` expects. Note that `google/gemma-3-270m-it` is a **gated**
HuggingFace repository: the `git clone` in the main README fails until the licence
is accepted on the model page and credentials are configured. The SigLIP2 encoders
and the LoFi checkpoint itself are ungated.


Two arms, because the README's downstream recipe and true continual training are
not the same thing:

**Arm A — the paper's downstream recipe (frozen encoder).**
Note that `--fix_enc` sets `lora_r = lora_alpha = 0` in `main.py` and freezes
every encoder parameter, so only the projection and the decoder LoRA train. The
merged encoder carries the LoFi weights; the trained projection and decoder LoRA
from the release are *not* reused.

```bash
python main.py \
    --dataset her2 \
    --her2_dir ./data/her2_512p/ \
    --model_dir ./models/ \
    --model_name siglip2-so400m-patch16-512-lofi-medg \
    --epochs 20 \
    --batch_size 16 \
    --multimodal_tokens 128 \
    --decoder_max_length 200 \
    --lr 1e-4 \
    --cos_eta_min 0.1 \
    --finetune_decoder \
    --pool2x2 \
    --fix_enc \
    --seed 42 \
    --result_dir ./results/
```

**Arm B — true continual training (encoder adapts, checkpoint fully resumed).**
Resuming requires the *unmerged* base encoder plus the checkpoint's own LoRA
geometry (`lora_r=16`), and is therefore incompatible with `--fix_enc`.

```bash
python main.py \
    --dataset her2 \
    --her2_dir ./data/her2_512p/ \
    --model_dir ./models/ \
    --model_name siglip2-so400m-patch16-512 \
    --resume ./models/lofi-medg/last.pt \
    --epochs 20 \
    --batch_size 16 \
    --multimodal_tokens 128 \
    --decoder_max_length 200 \
    --lr 1e-4 \
    --cos_eta_min 0.1 \
    --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 \
    --lora_r 16 --lora_alpha 16 \
    --finetune_decoder \
    --pool2x2 \
    --seed 42 \
    --result_dir ./results/
```

**Hyperparameters, and why they differ from the paper's.** MedG is 4.48M
triplets; this dataset is on the order of 10⁴. The README downstream recipe
(30 epochs, lr 3e-4) is tuned for datasets far larger than ours.

| Setting | Paper downstream | Here | Reason |
|---|---|---|---|
| `--lr` | 3e-4 | 1e-4 | Fewer, more repeated samples; 3e-4 on ~10⁴ samples drives the decoder LoRA to memorise the seven templates within a couple of epochs |
| `--epochs` | 30 | 20 | ~10⁴ samples × 20 epochs is already ~12k steps at bs16; select the checkpoint on `val`, do not assume the last epoch is best |
| `--lora_r` | 16 | 16 (Arm B) | Must match the released checkpoint's geometry to resume at all |
| `--pool2x2` | on | on, but worth an ablation | It halves spatial resolution (1024→256 tokens) before pooling. HER2 membrane staining is fine-grained, so `--no_pool2x2` is a sensible ablation at ~4× projection cost |

**Token budget — why these recipes use `--decoder_max_length 200`, not 150.**
Boxes are emitted as literal text, so each one costs decoder tokens. Measured
against the gemma-3-270m-it tokenizer over every template in `her2/captions.py`,
in both directions of the objective, with worst-case four-digit coordinates:

```
tokens(n boxes) = 168 + 20n        (main.py then adds multimodal_tokens)

--decoder_max_length 150  -> budget 278 -> 5 boxes fit
--decoder_max_length 200  -> budget 328 -> 8 boxes fit   <- MAX_BOXES = 8
```

**MAX_BOXES = 8 sits exactly on the boundary, with zero headroom.** Measured:
8 boxes cost 328 tokens against a budget of exactly 328; 9 cost 348 and truncate.
So *any* lengthening of a template in `her2/captions.py` — even by one token —
silently pushes the longest samples over. If you edit the caption vocabulary,
re-run `tools/check_her2_token_budget.py` and either keep the intercept at 168 or
raise `--decoder_max_length` accordingly. `tests/test_her2_token_budget.py` pins
the pairing, and the `pad_or_truncate` warning added in `lofi_utils/gemma.py`
catches it at training time if both are missed.

The `+20n` slope is a property of the coordinate serialisation and holds for any
dataset. The `168` intercept is **specific to the HER2 templates** — it is the
maximum over `her2/captions.py` and both objective directions, which is exactly
what `MAX_BOXES` must be sized against. Other datasets have different intercepts
(a one-word SegTHOR label measures 175 tokens at one box, a PadChest-length
sentence 194, against 185 here), so do not quote `168 + 20n` as a general fact
about the repository.

This matters because going over budget **does not raise**. `pad_or_truncate` cuts
the *end* of the sequence, removing the closing code fence and `<end_of_turn>`, so
the decoder trains on targets that never terminate. With `MAX_BOXES = 8` and the
paper's default `--decoder_max_length 150`, roughly the top of the box-count
distribution would be silently corrupted.

`her2/boxes.py` exposes `max_boxes_for()` and `required_decoder_max_length()`;
the preprocessing scripts print a warning when the two are mismatched, and the
test suite pins the pairing. Re-measure against any other tokenizer with:

```bash
python tools/check_her2_token_budget.py \
    --tokenizer_path ./models/gemma-3-270m-it/tokenizer.json
```

(The same check also verifies that `[multimodal]` is a single token in the
vocabulary — it is in gemma-3-270m-it, so the 128-token context placeholder
occupies exactly 128 positions.)

**Watch for template memorisation.** With only seven caption strings, a decoder
can reach a low loss by learning the text prior and ignoring the image. This is
the main failure mode of the weak-supervision bridge, and the reason Section 8
prescribes a shuffled-image control.

```bash
# Training curves, CSV + JSON summary, optional multi-run comparison
python tools/plot_her2_training.py \
    --result_dir ./results/her2_.../ --compare ./results/her2_other_.../
```

### 7b. Running this without a dedicated GPU

Two additions make the recipe reachable on a free Colab-class T4, or a laptop
for a wiring check. Neither changes the objective, the optimiser, or the
numbers the run produces.

**Precomputed encoder features.** The recipe passes `--fix_enc`, which sets
`requires_grad=False` on every encoder parameter (`main.py:185-187`), and the
pipeline applies no image augmentation — `BaseDataset.__getitem__` opens the
image and passes it straight to the processor, with no random crop, flip or
jitter anywhere in `lofi_utils/`. So for a fixed checkpoint the encoder output
for a given image is **identical on every epoch**, and the 30-epoch recipe
recomputes it 30 times.

`tools/precompute_features.py` computes it once:

```bash
python tools/precompute_features.py \
    --dataset her2 --her2_dir ./data/her2_512p/ --splits train val test \
    --model_dir ./models/ --model_name siglip2-so400m-patch16-512-lofi-medg \
    --resume ./models/lofi-medg/last.pt \
    --pool2x2 --feature_cache_dir ./cache/her2_features/ --batch_size 8

# then add one flag to the training call
python main.py ... --feature_cache_dir ./cache/her2_features/
```

This is exact, not an approximation. The cached tensor is precisely what
`train_eval.encode_vision` returns. What it does **not** change: the projection
head still trains (the cache sits upstream of it), and the 50/50
grounding/captioning direction is still redrawn per epoch (`base.py:62`) because
only the image side is cached.

Storage is post-`pool2x2` — a parameter-free average pool, so caching after it
is lossless for this recipe — giving 256×1152 per image, 590 KB in `float16`.
`--cache_dtype float32` doubles that and stores exactly what a CPU run computes.

Guards, because a stale cache is a silent correctness failure rather than a
crash:

- `--feature_cache_dir` without `--fix_enc` is a hard error. A training encoder
  invalidates the cache after the first optimiser step.
- A `manifest.json` records the encoder name, resumed-checkpoint fingerprint,
  image size, pooling, dtype and an `encoder_build` tag. A mismatch refuses to
  load. The tag exists because how the encoder is assembled can change without
  any of the other fields moving, which would leave a stale cache looking valid.
- A missing entry raises rather than falling back to running the encoder, so a
  half-built cache cannot look like it worked while costing full price.
- The encoder is kept on CPU when the cache is in use (it never executes),
  freeing ~1.6 GB of accelerator memory.

**Resuming the released checkpoint under `--fix_enc`.** These interact in a way
upstream never had to handle, because `--fix_enc` and resuming a LoRA checkpoint
were not combined before. `main.py:460` sets `lora_r = 0` when the encoder is
frozen — correct, since a frozen encoder has no LoRA to train — so the encoder is
built from plain `nn.Linear` layers. But the released checkpoint stores its
encoder as base weights *plus* LoRA deltas on `q_proj k_proj v_proj out_proj fc1
fc2`, and those tensors then have nowhere to go: `load_state_dict` fails with
every LoRA key unexpected.

`load_encoder_state_dict` folds them in first, using the same
`W' = W + (α/r)·B·A` as `tools/merge_lora.py`, with α and r read from the
checkpoint's own `args` because the CLI values have already been zeroed. The
alternative — merging to a new model directory with `tools/merge_lora.py` — would
discard the checkpoint's projection head and decoder, which Arm A needs.

`tools/precompute_features.py` builds its encoder the same way rather than with
live LoRA modules. The two are equal on paper but not bit-for-bit once rounded,
and the cached features must be what training would have computed.

**Why this matters for the memory budget.** With `--fix_enc` the encoder holds
no gradients, no optimiser state and no stored activations; trainable is decoder
LoRA (`r=4`, on `W_query`/`W_value`) plus the projection head. Memory is not the
binding constraint — wall-clock through the 400M-parameter encoder is, and that
is what the cache removes.

**Smoke test first.** `tools/smoke_test_her2.py` runs a handful of samples for
one epoch on real files with the real checkpoints, to prove the wiring before
spending scarce GPU time:

```bash
python tools/smoke_test_her2.py --her2_dir ./data/her2_512p/ --model_dir ./models/ --with_cache
```

It produces **no result**. `--limit_samples` exists for this and must never be
used for a number that reaches `RESULTS_her2.md`.

## 8. Evaluation, and how it must be read

```bash
# Continually-trained model
python main.py --evaluate test val --dataset her2 \
    --her2_dir ./data/her2_512p/ --model_dir ./models/ \
    --seed 42 --resume <RESULT_DIR>/ep<EPOCH>.pt --result_dir <RESULT_DIR>

# Baseline: the released checkpoint, zero-shot on the same split
python main.py --evaluate test --dataset her2 \
    --her2_dir ./data/her2_512p/ --model_dir ./models/ \
    --seed 42 --resume ./models/lofi-medg/last.pt --result_dir ./results_baseline/

# THE GATE: same checkpoint, same split, every target paired with a DIFFERENT
# image. If this scores close to the real evaluation the model is not using the
# image and every other number here is void. Writes to *_shuffled.csv, so it
# cannot overwrite the real evaluation it is meant to be compared against.
python main.py --evaluate test --shuffle_images --dataset her2 \
    --her2_dir ./data/her2_512p/ --model_dir ./models/ \
    --seed 42 --resume <RESULT_DIR>/ep<EPOCH>.pt --result_dir <RESULT_DIR>

# Split the pooled metric by supervision strength -- the pooled number alone
# mixes pathologist polygons with threshold-derived boxes
python tools/her2_eval_breakdown.py --pkl_path <RESULT_DIR>/eval_test_her2_ground_ep<EPOCH>.pkl

# Qualitative grounding on individual images
python grounding_demo.py --model_dir ./models/ --resume <RESULT_DIR>/ep<EPOCH>.pt \
    --image ./data/her2_512p/test/<some>.jpg \
    --label "strong complete membranous staining, HER2 3+ positive"
```

The repository's only detection metric is mono-class precision/recall/F1 at
IoU 0.5. What that means for each track:

- **CAMELYON track** — a fair grounding score. The targets are real polygons.
- **BCI IHC track** — measures agreement with a *threshold*, not with a
  pathologist. A high score here means the model reproduced our DAB heuristic. It
  is evidence the pipeline trains, not evidence of clinical HER2 localisation.
  Report it separately, never pooled, and never as a HER2 detection result.

Three controls belong in any honest write-up:

1. **Zero-shot baseline** — the released checkpoint on the same test split. If
   continual training does not beat it, say so.
2. **Shuffled-image control** — re-evaluate with images paired to the wrong
   captions. A model exploiting the seven-template text prior scores similarly;
   a model actually using the image degrades sharply.
3. **Per-track breakdown** — via `her2_eval_breakdown.py`.

## 9. Tests

```bash
python -m pytest tests -q      # 98 tests; 89 need no torch, GPU or downloaded data
```

Coverage: colour deconvolution separates DAB from haematoxylin and tracks grade;
ASAP annotation parsing, tile-window clipping at arbitrary pyramid downsamples,
and patient-level slide keying (the CAMELYON track is covered without OpenSlide);
box construction, letterbox padding arithmetic, sliver filtering and box caps;
manifest validation rejecting float coordinates, unsorted boxes, degenerate
boxes, out-of-canvas boxes, wrong split prefixes and trailing periods; split
determinism and slide-keyed assignment; and an end-to-end conversion of synthetic
IHC patches asserting 512×512 JPEG output, template-only captions, HER2 0
whole-tissue behaviour, and the exact box string the decoder is trained to emit.

the decoder token-budget cost model and its box-count pairing; and the chat
template copied into the budget tool staying identical to `gemma.py`'s.

The feature cache (§7b) is covered for the parts that decide correctness:
path-keying agreeing between relative and absolute forms, every manifest field
detecting staleness, a differing resumed checkpoint being rejected, save/load
round-tripping, no temporary files surviving an interrupted write, and a missing
entry raising instead of silently recomputing. The encoder pass itself is not
covered — it needs the real checkpoints.

The shuffled-image control is covered for the property that makes it a valid
gate: that it is a true derangement (no sample keeps its own image, at every
size), that it moves images without touching questions, boxes or `content_type`,
that it is deterministic per seed, and that it writes to a distinct filename so
it cannot overwrite the evaluation it is compared against.

Two tests read `det.py` and `main.py` as source text and assert the HER2 loader
serialises boxes **identically** to SegTHOR and MedG. If anyone edits that
convention, the suite fails rather than silently training against a mismatched
coordinate space.

## 10. Changes to existing repository files

Kept minimal so this can be submitted as a PR:

| File | Change |
|---|---|
| `main.py` | `'her2': HER2Dataset` in `DATASET_MAP`; `--her2_dir` argument and default; `--feature_cache_dir` / `--limit_samples` / `--shuffle_images` and the helpers that apply them; a `_shuffled` suffix on control result files |
| `lofi_utils/dataset/det.py` | `HER2Dataset`, mirroring `SegTHORDataset` and adding per-sample `content_type` metas |
| `lofi_utils/dataset/base.py` | image loading factored into `_load_image`, which serves cached features when one is attached. Default path is byte-for-byte the previous behaviour |
| `lofi_utils/model.py` | `ProjectionWrapper.forward` takes `already_pooled=False`, so cached post-pool features are not pooled twice. Default is unchanged |
| `lofi_utils/train_eval.py` | `encode_vision(..., cached=False)` passes precomputed features through. Default is unchanged |
| `lofi_utils/misc.py` | `import torch` moved inside `set_seed` / `seed_worker` so data preparation and the tests run without torch. Behaviour is unchanged |

Everything else is additive: `her2/`, `tools/preprocess_bci.py`,
`tools/preprocess_camelyon.py`, `tools/preprocess_her2_slide_labels.py`,
`tools/merge_her2_manifests.py`, `tools/visualize_her2.py`,
`tools/her2_eval_breakdown.py`, `tools/plot_her2_training.py`,
`tools/check_her2_token_budget.py`, `tools/precompute_features.py`,
`tools/smoke_test_her2.py`, `lofi_utils/feature_cache.py`, `tests/`.

## 11. Future work

Real region-level HER2 grounding needs annotation that does not currently exist
in public form: a pathologist marking membrane-staining regions on IHC slides
with per-region intensity. Until then this dataset is a weak-supervision bridge,
and the honest ceiling on what it can demonstrate is set by that fact. The
authors' forthcoming pathology-pretrained release may be the better foundation
for the semantics; the format work here transfers to it unchanged.
