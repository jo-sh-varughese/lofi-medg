# Silent target truncation in the downstream fine-tuning recipe

Hi — thanks for releasing LoFi-MedG; the MedG format made it straightforward to
add a new modality. While preparing a pathology extension I hit something in the
data pipeline that I think is worth your attention. I have tried to be precise
about what I actually measured and what I did not.

## Summary

Under the downstream fine-tuning recipe in `README.md`, a training sample whose
box list is long enough has the end of its target silently cut off, removing the
closing ``` fence and `<end_of_turn>`. No exception, no warning. The decoder is
then trained on a target that never terminates.

I could not determine whether this affects your published numbers — that depends
on the box-count distribution of PadChest-GR / TN5000 / SegTHOR, which I do not
have locally. **So this is a reproducible latent defect of unknown impact, not a
claim that any result is wrong.** The check on your side is cheap and I describe
it at the end.

## Mechanism

1. `README.md` downstream recipe: `--decoder_max_length 150 --multimodal_tokens 128`.
2. `main.py:317` adds them: `args.decoder_max_length = 150 + 128 = 278`.
3. `lofi_utils/dataset/base.py:71-73` builds the chat-templated prompt and calls
   `pad_or_truncate(tokens, 278)`.
4. `lofi_utils/gemma.py:520` truncates from the right:

```python
def pad_or_truncate(input_ids, max_length, pad_token_id=0):
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]  # truncate
    else:
        input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
    return input_ids
```

Boxes are serialised into the target as literal text (`det.py:44` and siblings),
so sequence length grows with box count and the right-hand end — the part that
gets cut — is exactly the terminator.

## Measured

Against the real `google/gemma-3-270m-it` tokenizer, worst-case four-digit
coordinates, grounding direction:

```
boxes  tokens   per box
    1     185
    3     225       +20
    5     265       +20
    6     285       +20
    8     325       +20
   10     365       +20
```

The `+20` per box is universal. The intercept depends on caption length —
175 tokens at one box for a one-word SegTHOR-style label, 185 for a short
pathology phrase, 194 for a PadChest-report-length sentence.

**Largest box count surviving intact at budget 278: 5** (6 for a one-word label).

Same 8-box sample under both of your recipes:

```
--decoder_max_length 150  (budget 278)   ->  325 tokens, TRUNCATED
    input tail : ...[1000,1000,1000,1000]]\n```\n<end_of_turn>\n
    output tail: ...[1000,1000,1000,1000],[1000,1000,1000,100
    model turn ends with closing fence : False
    model turn ends with <end_of_turn> : False
    exception raised                   : no

--decoder_max_length 200  (budget 328)   ->  325 tokens, fits
    output tail: ...]]\n```\n<end_of_turn>\n<pad><pad><pad>
    model turn ends with <end_of_turn> : True
```

The pretraining recipe (`--decoder_max_length 200 --multimodal_tokens 256`,
budget 456) has ample headroom and is not affected. The mismatch is only between
the two published recipes.

## Reproduction

`tools/repro_decoder_truncation.py` — single file, depends only on `tokenizers`,
runs in seconds against a fresh clone:

```bash
python tools/repro_decoder_truncation.py \
    --tokenizer_path ./models/gemma-3-270m-it/tokenizer.json
```

It imports your `apply_chat_template` and `pad_or_truncate` directly; if torch is
not installed it extracts those two functions verbatim from `lofi_utils/gemma.py`
via `ast` rather than reimplementing them, and prints which path it took. It dumps
the truncated output directly instead of describing it.

## Suggested fix

Behaviour-preserving: make truncation visible rather than changing any default,
so nobody's reproduction of the paper shifts underneath them. Deduped per
`max_length` so it prints once per run, not once per sample.

```diff
-def pad_or_truncate(input_ids, max_length, pad_token_id=0):
+_truncation_warned = set()
+
+
+def pad_or_truncate(input_ids, max_length, pad_token_id=0, warn=True):
     if len(input_ids) > max_length:
+        # Truncation drops the END of the target, where the closing ``` fence and
+        # <end_of_turn> live, so warn instead of doing it silently.
+        if warn and max_length not in _truncation_warned:
+            _truncation_warned.add(max_length)
+            print(f'WARNING: target of {len(input_ids)} tokens truncated to '
+                  f'decoder_max_length {max_length}; the closing fence and '
+                  f'<end_of_turn> are lost and the decoder will be trained on a '
+                  f'non-terminating target. Raise --decoder_max_length (note '
+                  f'main.py adds --multimodal_tokens to it) or reduce boxes per '
+                  f'sample. Further truncations at this length are not reported.')
         input_ids = input_ids[:max_length]  # truncate
     else:
         input_ids = input_ids + [pad_token_id] * (max_length - len(input_ids))
     return input_ids
```

Raising the downstream `--decoder_max_length` to 200 would also work, but I would
not suggest it as the primary fix: it silently changes a published recipe, and it
only patches the one configuration rather than surfacing the problem generally.

## What I could not check, and what would settle it

I found no box-count cap, assertion, or filter anywhere in the pipeline, so
nothing prevents the case from arising. But whether it *occurs* in your training
runs I genuinely do not know:

- **SegTHOR / TN5000** — plausibly always under the limit. `seg_slice_to_boxes`
  drops components below 4% of area, fuses overlapping boxes and removes nested
  ones, so few boxes survive per slice.
- **PadChest-GR** — the most likely to cross. Boxes come straight from the
  grounded-report JSON with no cap (`data.py:172-183`), and its captions are the
  longest, which lowers the box threshold further.

Both of those are inferences from reading the code, not measurements.

With the data on hand this is a one-liner to settle — the fraction of samples
above the threshold, per dataset:

```python
over = sum(1 for _, boxes in dataset.qas if len(ast.literal_eval(boxes)) > 5)
print(over, len(dataset.qas))
```

If that is zero everywhere, the defect never fired and your results are entirely
unaffected — worth fixing only to protect future datasets. If it is non-zero for
PadChest-GR, it may be worth a look.

Happy to open this as a PR with the reproduction script and the diff if useful.
