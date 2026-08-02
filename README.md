# LoFi-MedG

Location-Aware Fine-Grained Representation Learning for Medical Vision Foundation Models

# Data

See [README_medg.md](README_medg.md) and [README_downstream.md](README_downstream.md) for
instructions on dataset download and preprocessing.

# Location-Aware Fine-Grained Representation Learning

    # Training recipe
    python main.py \
        --dataset medg --concat_dataset mimic --increase_concat_dataset 2 --epochs 10 \
        --medg_dir ./data/MedG_512p/ \
        --mimic_image_dir ./data/mimic_512p_good/ \
        --mimic_json_dir ./data/llava-rad-mimic-cxr-annotations-1.0.0/ \
        --model_dir ./models/ \
        --model_name siglip2-so400m-patch16-512 \
        --batch_size 64 \
        --multimodal_tokens 256 \
        --lr 3e-4 \
        --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2 \
        --lora_r 16 --lora_alpha 16 \
        --finetune_decoder \
        --decoder_max_length 200 \
        --pool2x2 \
        --result_dir ./results/ \
        --max_steps_per_epoch 2_000_000

### Trained Model

<table><tbody>
<tr><td>LoFi-MedG</td>
<td><a href="https://huggingface.co/myeongkyunkang/lofi-medg">download</a></td></tr>
</tbody></table>

### LoRA Merging

To merge the LoRA weights, run:

```bash
python tools/merge_lora.py
```

We provide a model compatible with [
`google/siglip2-so400m-patch16-512`](https://huggingface.co/google/siglip2-so400m-patch16-512).

<table><tbody>
<tr><td>siglip2-so400m-patch16-512-lofi-medg</td>
<td><a href="https://huggingface.co/myeongkyunkang/siglip2-so400m-patch16-512-lofi-medg">download</a></td></tr>
</tbody></table>

# External Validation on Grounding

    # Evaluation on SegTHOR
    python main.py \
        --evaluate test \
        --dataset segthor \
        --segthor_dir ./data/segthor_512p/ \
        --model_dir ./models/ \
        --model_name siglip2-so400m-patch16-512 \
        --seed 42 \
        --resume ./models/lofi-medg/last.pt \
        --result_dir ./results_ext/

# Downstream Fine-Tuning

    # DATASET: padchest, tn5000, segthor, slake, vqarad, omnimedvqa

    # Fine-tuning recipe
    python main.py \
        --dataset <DATASET> \
        --epochs 30 \
        --decoder_max_length 150 \
        --padchest_image_dir ./data/padchest_512p/ \
        --tn5000_dir ./data/tn5000_512p/ \
        --segthor_dir ./data/segthor_512p/ \
        --slake_dir ./data/SLAKE/ \
        --vqarad_dir ./data/VQA_RAD/ \
        --omnimedvqa_dir ./data/omnimedvqa_512p/ \
        --model_dir ./models/ \
        --model_name siglip2-so400m-patch16-512-lofi-medg \
        --seed 42 \
        --batch_size 16 \
        --multimodal_tokens 128 \
        --lr 3e-4 \
        --cos_eta_min 0.1 \
        --finetune_decoder \
        --pool2x2 \
        --fix_enc \
        --result_dir ./results/

    # Evaluation
    main.py \
        --evaluate test val \
        --dataset <DATASET> \
        --padchest_image_dir ./data/padchest_512p/ \
        --tn5000_dir ./data/tn5000_512p/ \
        --segthor_dir ./data/segthor_512p/ \
        --slake_dir ./data/SLAKE/ \
        --vqarad_dir ./data/VQA_RAD/ \
        --omnimedvqa_dir ./data/omnimedvqa_512p/ \
        --model_dir ./models/ \
        --seed 42 \
        --resume <RESULT_DIR>/ep<EPOCH>.pt \
        --result_dir <RESULT_DIR>

# Models

Download model checkpoints under `./models`.

    cd ./models
    git clone https://huggingface.co/google/siglip2-so400m-patch16-512
    git clone https://huggingface.co/google/gemma-3-270m-it

# Requirements

    python -m venv lofi
    source lofi/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
