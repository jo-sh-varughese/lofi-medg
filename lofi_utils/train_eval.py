import ast
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from lofi_utils.gemma import apply_chat_template
from lofi_utils.metrics import calc_detection_metrics, calc_exactmatch_accuracy


def encode_vision(model, images, cached=False):
    # cached: `images` is not pixel values but the encoder output itself,
    # precomputed by tools/precompute_features.py. Valid only with --fix_enc,
    # where the encoder is deterministic across epochs.
    if cached:
        return images
    v = model.vision_model(pixel_values=images)
    return v.last_hidden_state


def train(model, loader, optimizer, decoder, projection, device, dtype, args):
    model.train()
    decoder.train()
    projection.train()

    total_loss_list = []

    steps = 0
    for images, tokens, attention_mask in tqdm(loader, desc='Training', leave=False):
        images = images.to(device=device, dtype=dtype)

        optimizer.zero_grad()

        # encode
        cached = bool(getattr(args, 'feature_cache_dir', ''))
        last = encode_vision(model, images, cached=cached)

        tokens = tokens.to(device=device)
        attention_mask = attention_mask.to(device=device)

        # projection (cached features are stored post-pool2x2, so do not pool again)
        multimodal_tokens = projection(last, already_pooled=cached)
        decoder_logits = decoder(tokens, multimodal=multimodal_tokens)

        # shift logits and labels for auto-regressive loss
        logits_shifted = decoder_logits[:, :-1, :].contiguous()  # (B, T-1, V)
        labels_shifted = tokens[:, 1:].contiguous()  # (B, T-1)

        # shift attention_mask
        mask_shifted = attention_mask[:, 1:].contiguous()  # (B, T-1)
        labels_shifted = labels_shifted.masked_fill(mask_shifted == 0, -100)  # 1 for real tokens, 0 for padding

        # calculate loss
        caption_loss = F.cross_entropy(logits_shifted.view(-1, logits_shifted.size(-1)), labels_shifted.view(-1), ignore_index=-100)

        total_loss = caption_loss * args.lambda_caption

        # backward
        total_loss.backward()

        # clip gradient norm
        if args.clip_grad_norm > 0:
            params = list(filter(lambda p: p.requires_grad, model.parameters()))
            if args.finetune_decoder:
                params.extend(list(filter(lambda p: p.requires_grad, decoder.parameters())))
            params.extend(projection.parameters())
            torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad_norm))
        optimizer.step()

        # append loss
        total_loss_list.append(total_loss.item() / images.shape[0])

        # save gpu usage
        if steps < (3 * args.batch_size):
            log_gpu_usage(args.result_dir)

        # increase steps
        steps += args.batch_size

        if steps > args.max_steps_per_epoch:
            print('Exceeded max steps per epoch; breaking.')
            break

    return np.mean(total_loss_list)


def evaluate_text_generation(model, data_loader, decoder, projection, device, dtype, args):
    model.eval()
    decoder.eval()
    projection.eval()

    # get tokenizer and etc
    dataset = data_loader.dataset
    dataset_name = dataset.name
    decoder_tokenizer = dataset.decoder_tokenizer
    decoder_max_length = dataset.decoder_max_length
    context = dataset.context
    end_token = dataset.end_token

    gt_list, pred_list = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing', leave=False):
            images, questions, answers = batch[0], batch[1], batch[2]

            # run each batch
            for image, question, answer in zip(images, questions, answers):
                image = image.to(device=device, dtype=dtype).unsqueeze(0)

                if dataset_name in ['slake', 'vqarad', 'omnimedvqa']:
                    instruction = question
                else:
                    question = question[:-1] if question.endswith('.') else question
                    instruction = f'Detect all instances of "{question}".'

                # build prompt and encode
                prompt = apply_chat_template(f'{context}\n{instruction}')
                input_token = decoder_tokenizer.encode(prompt)
                begin_index = len(input_token)

                # encode
                cached = bool(getattr(args, 'feature_cache_dir', ''))
                last = encode_vision(model, image, cached=cached)

                # projection (cached features are stored post-pool2x2)
                multimodal_tokens = projection(last, already_pooled=cached)

                output_token = torch.tensor(input_token, device=device).unsqueeze(0)  # batchify
                for _ in range(decoder_max_length):
                    out = decoder(output_token, multimodal=multimodal_tokens)[:, -1]
                    next_token = torch.argmax(out, dim=-1, keepdim=True)
                    if torch.all(next_token == end_token):
                        break
                    output_token = torch.cat([output_token, next_token], dim=1)

                # decode prediction
                pred_response = decoder_tokenizer.decode(output_token.squeeze(0).tolist()[begin_index:]).strip()

                if dataset_name in ['slake', 'vqarad', 'omnimedvqa']:
                    pred_list.append(pred_response.strip())
                    gt_list.append(answer.strip())
                else:
                    try:
                        pred_response = pred_response.replace('`', '').strip()
                        pred_boxes = ast.literal_eval(pred_response)
                        if len(pred_boxes) == 0:
                            raise ValueError()  # raise an error for empty and a single value
                        pred_boxes = np.array(pred_boxes)
                        if not all_valid_boxes(pred_boxes):
                            raise ValueError()

                        pred_classes = np.array([1] * len(pred_boxes), dtype=int)
                        pred_confidences = np.array([0.99] * len(pred_boxes))
                    except:
                        pred_boxes, pred_classes, pred_confidences = np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))

                    # decode gt
                    tgt_boxes = np.array(ast.literal_eval(answer))
                    tgt_class = np.array([1] * len(tgt_boxes), dtype=int)

                    # append boxes
                    pred_list.append((pred_boxes, pred_classes, pred_confidences))
                    gt_list.append((tgt_boxes, tgt_class))

    return gt_list, pred_list


def save_vqa_eval(csv_path, test_dataset, targets, predictions):
    answer_types = [m['answer_type'] for m in test_dataset.metas]

    # save csv
    pd.DataFrame({
        'predictions': predictions, 'targets': targets, 'answer_types': answer_types,
    }).to_csv(csv_path.replace('.csv', '_data.csv'), index=False)
    open_targets, open_predictions, closed_targets, closed_predictions = [], [], [], []
    for t, p, at in zip(targets, predictions, answer_types):
        if at == 'OPEN':
            open_targets.append(t)
            open_predictions.append(p)
        elif at == 'CLOSED':
            closed_targets.append(t)
            closed_predictions.append(p)
        else:
            raise ValueError()

    # calculate metrics
    all_acc = calc_exactmatch_accuracy(targets, predictions)
    open_acc = calc_exactmatch_accuracy(open_targets, open_predictions)
    closed_acc = calc_exactmatch_accuracy(closed_targets, closed_predictions)

    # save metrics
    rows = {}
    rows['All Acc'] = [round(all_acc, 6)]
    rows['Open Acc'] = [round(open_acc, 6)]
    rows['Closed Acc'] = [round(closed_acc, 6)]
    rows['All N'] = [len(targets)]
    rows['Open N'] = [len(open_targets)]
    rows['Closed N'] = [len(closed_targets)]
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_detection_eval(csv_path, test_dataset, targets, predictions):
    pkl_path = csv_path.replace('.csv', '.pkl')
    metas = getattr(test_dataset, 'metas', None)
    content_types = [m.get('content_type', '') for m in metas] if metas is not None else [''] * len(targets)

    # save pickle
    with open(pkl_path, 'wb') as wf:
        pickle.dump({'predictions': predictions, 'targets': targets, 'content_types': content_types}, wf)

    pd.DataFrame({
        'predictions': predictions,
        'targets': targets,
        'content_types': content_types,
    }).to_csv(csv_path.replace('.csv', '_data.csv'), index=False)

    # save metrics
    calc_detection_metrics(pkl_path, pkl_path.replace('.pkl', '_mono.csv'), mono_class=True)


def all_valid_boxes(boxes):
    for box in boxes:
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return False
    return True


def log_gpu_usage(output_dir):
    used_gb = reserved_gb = total_gb = 0
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        used_gb = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    with open(os.path.join(output_dir, "gpu_log.txt"), "a", encoding="utf-8") as wf:
        wf.write(f"GPU Memory: allocated={used_gb:.2f} GB, reserved={reserved_gb:.2f} GB, total={total_gb:.2f} GB ({used_gb / max(total_gb, 1e-6) * 100:.1f}%)\n")
