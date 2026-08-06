import argparse
import ast
import io
import os
import urllib.request
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from lofi_utils.gemma import Gemma3Model, GemmaTokenizer, apply_chat_template
from lofi_utils.misc import convert_ori_space
from lofi_utils.model import ProjectionWrapper, apply_lora, build_model
from lofi_utils.train_eval import encode_vision
from tools.preprocess_utils import resize_with_padding
from tools.seg_label_info import NAME_TO_LABEL_DICT

DEFAULT_IMAGE = 'https://prod-images-static.radiopaedia.org/images/62559660/89612e877be0703fc6f7e480d9e1d1cdc189404430caef927b37283b0fc6942b.png'
MEDG_LABEL_SET = set(NAME_TO_LABEL_DICT.values())


def load_image(path_or_url):
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        request = urllib.request.Request(path_or_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(request) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert('RGB')
    return Image.open(path_or_url).convert('RGB')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if not os.path.isfile(args.resume):
        raise ValueError('Invalid resume', args.resume)

    print('Loading checkpoint...')
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    ckpt_args = ckpt['args']

    print('Create model...')
    model, processor = build_model(ckpt_args['model_name'], args.model_dir)

    if processor.image_processor.size['height'] != processor.image_processor.size['width']:
        raise ValueError('Image sizes do not match:', processor.image_processor.size)
    image_size = processor.image_processor.size['height']
    print('Image size:', image_size)

    if ckpt_args['lora_r'] > 0:
        model = apply_lora(
            model,
            r=ckpt_args['lora_r'],
            lora_alpha=ckpt_args['lora_alpha'],
            target_modules=ckpt_args['lora_target_modules'],
            head_name=ckpt_args['head_name'],
        )

    decoder_dir = os.path.join(args.model_dir, ckpt_args['decoder_name'])
    decoder_tokenizer = GemmaTokenizer(tokenizer_file_path=os.path.join(decoder_dir, 'tokenizer.json'))
    decoder = Gemma3Model(weights_file=os.path.join(decoder_dir, 'model.safetensors'))

    finetune_decoder = ckpt_args['finetune_decoder']
    if finetune_decoder:
        decoder = apply_lora(
            decoder,
            r=ckpt_args['decoder_lora_r'],
            lora_alpha=ckpt_args['decoder_lora_alpha'],
            target_modules=ckpt_args['decoder_lora_target_modules'],
            head_name='NO_HEAD_FOR_DECODER',
        )

    multimodal_tokens = ckpt_args['multimodal_tokens']
    pool2x2 = ckpt_args.get('pool2x2', False)
    decoder_max_length = ckpt_args['decoder_max_length']

    vision_hidden_size = 1152
    vision_model_config = SimpleNamespace(
        hidden_size=vision_hidden_size, intermediate_size=(vision_hidden_size // 4),
        num_attention_heads=16, hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-06
    )
    projection = ProjectionWrapper(vision_model_config, multimodal_tokens, decoder.cfg['emb_dim'], use_pool2x2=pool2x2)

    print('Resuming from checkpoint...')
    model.load_state_dict(ckpt['state_dict'])
    projection.load_state_dict(ckpt['projection'])
    if finetune_decoder:
        decoder.load_state_dict(ckpt['decoder'])

    model = model.to(device=device, dtype=dtype).eval()
    decoder = decoder.to(device=device, dtype=dtype).eval()
    projection = projection.to(device=device, dtype=dtype).eval()

    print(f'Loading image: {args.image}')
    ori_image = load_image(args.image)
    ori_w, ori_h = ori_image.size

    padded_image = resize_with_padding(ori_image, image_size)
    pixel_values = processor(images=[padded_image], return_tensors='pt')['pixel_values'][0]
    pixel_values = pixel_values.to(device=device, dtype=dtype).unsqueeze(0)

    label = args.label[:-1] if args.label.endswith('.') else args.label
    if label not in MEDG_LABEL_SET:
        print(f'Warning: "{label}" is not a label the model was trained on.')
    instruction = f'Detect all instances of "{label}".'
    context = '[multimodal]' * multimodal_tokens
    prompt = apply_chat_template(f'{context}\n{instruction}')
    input_token = decoder_tokenizer.encode(prompt)
    begin_index = len(input_token)
    end_token = decoder_tokenizer.encode('<end_of_turn>')[-1]

    print('Running inference...')
    with torch.no_grad():
        last = encode_vision(model, pixel_values)
        multimodal = projection(last)

        output_token = torch.tensor(input_token, device=device).unsqueeze(0)
        for _ in range(decoder_max_length):
            out = decoder(output_token, multimodal=multimodal)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)
            if torch.all(next_token == end_token):
                break
            output_token = torch.cat([output_token, next_token], dim=1)

    pred_response = decoder_tokenizer.decode(output_token.squeeze(0).tolist()[begin_index:]).strip()

    pred_response_clean = pred_response.replace('`', '').strip()
    try:
        pred_boxes = ast.literal_eval(pred_response_clean)
        if len(pred_boxes) == 0:
            raise ValueError()
    except Exception:
        pred_boxes = []

    boxes_pixel = []
    if len(pred_boxes) > 0:
        norm_boxes = [[v / 1000 for v in box] for box in pred_boxes]  # pad-space -> [0,1]
        ori_boxes = convert_ori_space(norm_boxes, (ori_w, ori_h), image_size)  # pad-space -> original-space [0,1]
        for x0, y0, x1, y1 in ori_boxes:
            boxes_pixel.append([x0 * ori_w, y0 * ori_h, x1 * ori_w, y1 * ori_h])

    print('Prediction:', boxes_pixel, '| Raw prediction:', pred_response)

    result_image = ori_image.copy()
    draw = ImageDraw.Draw(result_image)
    line_width = max(2, min(ori_w, ori_h) // 200)
    for x0, y0, x1, y1 in boxes_pixel:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=line_width)
        draw.text((x0 + 2, max(0, y0 - 14)), label, fill=(255, 0, 0))

    result_image.save(args.output, 'JPEG', quality=100)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE)
    parser.add_argument('--label', type=str, default='lung')
    parser.add_argument('--model_dir', type=str, default='./models/')
    parser.add_argument('--resume', type=str, default='./models/lofi-medg/last.pt')
    parser.add_argument('--output', type=str, default='./pred.jpg')
    args = parser.parse_args()
    main(args)
