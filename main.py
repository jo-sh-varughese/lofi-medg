import argparse
import json
import os
import re
import shutil
from types import SimpleNamespace

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from lofi_utils.dataset.det import MedGDataset, MIMICDataset, PadChestDataset, TN5000Dataset, SegTHORDataset, HER2Dataset
from lofi_utils.dataset.vqa import SLAKEDataset, VQARADDataset, OmniMedVQADataset
from lofi_utils.feature_cache import build_manifest, FeatureCache, read_manifest
from lofi_utils.gemma import Gemma3Model, GemmaTokenizer
from lofi_utils.misc import set_seed, seed_worker
from lofi_utils.model import apply_lora, build_model, ProjectionWrapper
from lofi_utils.train_eval import train, evaluate_text_generation, save_vqa_eval, save_detection_eval

DATASET_MAP = {
    'medg': MedGDataset,
    'mimic': MIMICDataset,
    'padchest': PadChestDataset,
    'tn5000': TN5000Dataset,
    'segthor': SegTHORDataset,
    'her2': HER2Dataset,
    'slake': SLAKEDataset,
    'vqarad': VQARADDataset,
    'omnimedvqa': OmniMedVQADataset,
}


def attach_feature_cache(dataset, args, image_size):
    '''
    Point a dataset at precomputed frozen-encoder features instead of images.

    Only valid when the encoder is frozen (--fix_enc): if any encoder parameter
    trains, its output changes between epochs and cached features are stale
    after the first step. That is a silent correctness failure, so it is a hard
    error rather than a warning.
    '''
    if not args.feature_cache_dir:
        return dataset

    if not args.fix_enc:
        raise ValueError(
            '--feature_cache_dir requires --fix_enc. Without a frozen encoder the cached '
            'features go stale after the first optimizer step and training would silently '
            'be wrong.'
        )

    found = read_manifest(args.feature_cache_dir)
    if found is None:
        raise FileNotFoundError(
            f'no feature cache in {args.feature_cache_dir}; build it first:\n'
            f'  python tools/precompute_features.py --dataset {args.dataset} '
            f'--feature_cache_dir {args.feature_cache_dir} ...'
        )

    expected = build_manifest(args.model_name, args.resume, image_size, args.pool2x2,
                              found.get('dtype'), found.get('feature_shape', [0, 0]))
    cache = FeatureCache(args.feature_cache_dir, expected_manifest=expected)

    missing = cache.missing(dataset.imgs)
    if missing:
        raise KeyError(
            f'{len(missing)} of this split\'s images are not in the cache '
            f'(e.g. {missing[0]}). Rerun tools/precompute_features.py with --splits covering '
            f'every split this run touches.'
        )

    dataset.feature_cache = cache
    print(f'Using precomputed features from {args.feature_cache_dir} '
          f'(shape {found.get("feature_shape")}, dtype {found.get("dtype")}) -- encoder will not run.')
    return dataset


def limit_samples(dataset, n):
    '''
    Truncate a dataset in place, for the laptop smoke test. Keeps imgs/qas/metas
    aligned so per-sample provenance still matches.
    '''
    if n <= 0 or n >= len(dataset.imgs):
        return dataset
    dataset.imgs = dataset.imgs[:n]
    dataset.qas = dataset.qas[:n]
    if getattr(dataset, 'metas', None) is not None:
        dataset.metas = dataset.metas[:n]
    print(f'--limit_samples: using {len(dataset.imgs)} samples (SMOKE TEST -- not a result)')
    return dataset


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    #################
    # create model
    #################
    print('Create model...')
    model, processor = build_model(args.model_name, args.model_dir)

    # set image_size
    if processor.image_processor.size['height'] != processor.image_processor.size['width']:
        raise ValueError('Image sizes do not match:', processor.image_processor.size)
    args.image_size = processor.image_processor.size['height']

    print('Image size:', args.image_size)

    # apply lora
    if args.lora_r > 0:
        print(f'Apply LoRA to the model containing: {args.lora_target_modules}')
        model = apply_lora(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            head_name=args.head_name
        )

    #################
    # create decoder
    #################
    decoder_dir = os.path.join(args.model_dir, args.decoder_name)
    tokenizer_file_path = os.path.join(decoder_dir, 'tokenizer.json')
    weights_file = os.path.join(decoder_dir, 'model.safetensors')

    # create decoder tokenizer
    decoder_tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)

    # create decoder model
    decoder = Gemma3Model(weights_file=weights_file)
    if args.finetune_decoder:  # synchronized with the resumed ckpt
        print(f'Apply LoRA to the decoder containing: {args.decoder_lora_target_modules}')
        decoder = apply_lora(
            decoder,
            r=args.decoder_lora_r,
            lora_alpha=args.decoder_lora_alpha,
            target_modules=args.decoder_lora_target_modules,
            head_name='NO_HEAD_FOR_DECODER'
        )
    else:
        for param in decoder.parameters():
            param.requires_grad = False

    # create projection
    vision_hidden_size = 1152
    vision_model_config = SimpleNamespace(
        hidden_size=vision_hidden_size, intermediate_size=(vision_hidden_size // 4),  # follow resset bottleneck design # default intermediate_size in model.vision_model.config is 4304
        num_attention_heads=16, hidden_act='gelu_pytorch_tanh', layer_norm_eps=1e-06
    )
    projection = ProjectionWrapper(vision_model_config, args.multimodal_tokens, decoder.cfg['emb_dim'], use_pool2x2=args.pool2x2)

    # resume
    if os.path.isfile(args.resume):
        print('Resuming from checkpoint...')
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        projection.load_state_dict(ckpt['projection'])
        if args.finetune_decoder:
            print('Resuming decoder from checkpoint...')
            decoder.load_state_dict(ckpt['decoder'])

    # set device
    if args.feature_cache_dir:
        # The encoder never runs in this mode, so keep its ~1.6 GB off the
        # accelerator. It is still constructed and loaded, so checkpoint saving
        # and --resume round-trip unchanged.
        print('Feature cache in use: keeping the encoder on CPU (it will not be executed).')
        model = model.to(device='cpu', dtype=torch.float32)
    else:
        model = model.to(device=device, dtype=dtype)
    decoder = decoder.to(device=device, dtype=dtype)
    projection = projection.to(device=device, dtype=dtype)

    # compile models
    uncompiled_model = model  # for state_dict
    uncompiled_decoder = decoder
    uncompiled_projection = projection
    if args.compile_model:
        model = torch.compile(model)
        decoder = torch.compile(decoder)
        projection = torch.compile(projection)

    Dataset = DATASET_MAP.get(args.dataset)
    if Dataset is None:
        raise ValueError('Unknown dataset', args.dataset)

    if len(args.evaluate) > 0:
        name = os.path.splitext(os.path.basename(args.resume))[0]

        # evaluate grounding
        for evaluate in args.evaluate:
            test_dataset = Dataset(
                split=evaluate,
                processor=processor,
                decoder_tokenizer=decoder_tokenizer,
                multimodal_tokens=args.multimodal_tokens,
                decoder_max_length=args.decoder_max_length,
                args=args,
                type='qa',
            )
            test_dataset = limit_samples(test_dataset, args.limit_samples)
            test_dataset = attach_feature_cache(test_dataset, args, args.image_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            targets, predictions = evaluate_text_generation(model, test_loader, decoder, projection, device, dtype, args)

            csv_path = os.path.join(args.result_dir, f'eval_{evaluate}_{args.dataset}_ground_{name}.csv')

            if args.dataset in ['slake', 'vqarad', 'omnimedvqa']:
                save_vqa_eval(csv_path, test_dataset, targets, predictions)
            else:
                save_detection_eval(csv_path, test_dataset, targets, predictions)

        exit()

    print('Build datasets...')
    train_dataset = Dataset(
        split='train',
        processor=processor,
        decoder_tokenizer=decoder_tokenizer,
        multimodal_tokens=args.multimodal_tokens,
        decoder_max_length=args.decoder_max_length,
        args=args,
    )
    train_dataset = limit_samples(train_dataset, args.limit_samples)
    train_dataset = attach_feature_cache(train_dataset, args, args.image_size)

    if args.concat_dataset != '':
        if args.feature_cache_dir:
            raise ValueError('--feature_cache_dir does not cover --concat_dataset splits; '
                             'cache those datasets too and extend this branch before using both.')
        concat_dataset_list = [train_dataset]
        if 'mimic' in args.concat_dataset:
            concat_dataset_list.extend([
                MIMICDataset(
                    split='train',
                    processor=processor,
                    decoder_tokenizer=decoder_tokenizer,
                    multimodal_tokens=args.multimodal_tokens,
                    decoder_max_length=args.decoder_max_length,
                    args=args,
                )
                for _ in range(args.increase_concat_dataset)
            ])
        train_dataset = ConcatDataset(concat_dataset_list)

    print('Build data loaders...')
    train_generator = worker_init_fn = None
    if args.seed is not None:
        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)
        worker_init_fn = seed_worker
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn, generator=train_generator)

    # freeze encoder
    if args.fix_enc:
        for param in model.parameters():
            param.requires_grad = False

    # create optimizer
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.finetune_decoder:
        params.extend(list(filter(lambda p: p.requires_grad, decoder.parameters())))
    params.extend(projection.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # create lr scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * args.cos_eta_min)

    for epoch in range(args.epochs):
        total_loss = train(model, train_loader, optimizer, decoder, projection, device=device, dtype=dtype, args=args)
        train_log = f'Train: [{epoch}/{args.epochs}] | Total Loss: {total_loss:.4f}'
        print(train_log)
        with open(os.path.join(args.result_dir, 'train_log.txt'), 'a') as wf:
            wf.write(train_log + '\n')

        # update learning rate
        scheduler.step()

        # save last checkpoint
        last_ckpt_path = os.path.join(args.result_dir, 'last.pt')
        projection_state_dict = uncompiled_projection.state_dict()
        decoder_state_dict = uncompiled_decoder.state_dict() if args.finetune_decoder else {}
        torch.save({'state_dict': uncompiled_model.state_dict(), 'projection': projection_state_dict, 'decoder': decoder_state_dict, 'args': vars(args), 'epoch': epoch}, last_ckpt_path)
        save_ckpt_path = os.path.join(args.result_dir, f'ep{epoch}.pt')
        shutil.copy(last_ckpt_path, save_ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=str, nargs='+', default=[])
    parser.add_argument('--result_dir', type=str)

    # dataset
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--concat_dataset', type=str, default='')
    parser.add_argument('--increase_concat_dataset', type=int, default=1)

    # det dataset
    parser.add_argument('--medg_dir', type=str)
    parser.add_argument('--mimic_image_dir', type=str)
    parser.add_argument('--mimic_json_dir', type=str)
    parser.add_argument('--chexmask_mimic_path', type=str)
    parser.add_argument('--mimic_ext_path', type=str)
    parser.add_argument('--padchest_image_dir', type=str)
    parser.add_argument('--padchest_grounded_reports_path', type=str)
    parser.add_argument('--padchest_master_table_path', type=str)
    parser.add_argument('--padchest_ori_size_path', type=str)
    parser.add_argument('--tn5000_dir', type=str)
    parser.add_argument('--segthor_dir', type=str)
    parser.add_argument('--her2_dir', type=str)

    # vqa dataset
    parser.add_argument('--slake_dir', type=str)
    parser.add_argument('--vqarad_dir', type=str)
    parser.add_argument('--vqarad_split_tsv_path', type=str)
    parser.add_argument('--omnimedvqa_dir', type=str)

    # training
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cos_eta_min', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)

    # frozen-encoder feature cache (see tools/precompute_features.py)
    parser.add_argument('--feature_cache_dir', type=str, default='',
                        help='load precomputed frozen-encoder features instead of images; requires --fix_enc')
    parser.add_argument('--limit_samples', type=int, default=0,
                        help='truncate each split to N samples; for smoke tests only, never for a reported result')

    # model
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name', type=str, default='siglip2-so400m-patch16-512')
    parser.add_argument('--decoder_name', type=str, default='gemma-3-270m-it')
    parser.add_argument('--decoder_max_length', type=int, default=400)
    parser.add_argument('--resume', type=str)

    # lora
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'out_proj'])
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--head_name', type=str, default='head')

    # decoder lora
    parser.add_argument('--finetune_decoder', dest='finetune_decoder', action='store_true')
    parser.add_argument('--no_finetune_decoder', dest='finetune_decoder', action='store_false')
    parser.add_argument('--decoder_lora_target_modules', type=str, nargs='+', default=['W_query', 'W_value'])
    parser.add_argument('--decoder_lora_r', type=int, default=4)
    parser.add_argument('--decoder_lora_alpha', type=int, default=4)

    # etc
    parser.add_argument('--lambda_caption', type=float, default=1.0)
    parser.add_argument('--multimodal_tokens', type=int, default=128)
    parser.add_argument('--pool2x2', dest='pool2x2', action='store_true')
    parser.add_argument('--no_pool2x2', dest='pool2x2', action='store_false')
    parser.add_argument('--fix_enc', dest='fix_enc', action='store_true')
    parser.add_argument('--compile_model', dest='compile_model', action='store_true')
    parser.add_argument('--no_compile_model', dest='compile_model', action='store_false')
    parser.add_argument('--max_steps_per_epoch', type=int, default=2_000_000)
    parser.set_defaults(
        result_dir='./results/',
        # det dataset
        medg_dir='./data/MedG_512p/',
        mimic_image_dir='./data/mimic_512p_good/',
        mimic_json_dir='./data/llava-rad-mimic-cxr-annotations-1.0.0/',
        chexmask_mimic_path='data/mimic/chexmask_mimic_cxr.csv',
        mimic_ext_path='data/mimic/mimic_ext.csv',
        padchest_image_dir='./data/padchest_512p/',
        padchest_grounded_reports_path='data/padchest/grounded_reports_20240819.json',
        padchest_master_table_path='data/padchest/master_table.csv',
        padchest_ori_size_path='data/padchest/ori_size.csv',
        tn5000_dir='./data/tn5000_512p/',
        segthor_dir='./data/segthor_512p/',
        her2_dir='./data/her2_512p/',
        # vqa dataset
        slake_dir='./data/SLAKE/',
        vqarad_dir='./data/VQA_RAD/',
        vqarad_split_tsv_path='data/vqarad/vqa_rad_balanced_split_and_human_eval_inclusions.tsv',
        omnimedvqa_dir='./data/omnimedvqa_512p/',
        model_dir='./models/',
        resume='',
        finetune_decoder=True,
        pool2x2=True,
    )
    args = parser.parse_args()

    args.resume = str(args.resume)

    # increase decoder_max_length based on multimodal_tokens
    args.decoder_max_length = args.decoder_max_length + args.multimodal_tokens

    if len(args.evaluate) == 0:
        if args.fix_enc:
            args.lora_r = args.lora_alpha = 0

        # for full tuning without LoRA
        if args.lora_r <= 0:
            args.lora_r = args.lora_alpha = 0

        # set postfix
        pre_postfix = ''

        # caption postfix
        pre_postfix += f'_cp{args.multimodal_tokens}-{args.lambda_caption}'
        if args.finetune_decoder:
            pre_postfix += f'_de{args.decoder_lora_r}-{args.decoder_lora_alpha}'
        if args.pool2x2:
            pre_postfix += f'_p2'

        if args.fix_enc:
            pre_postfix += '_fix'

        # settings postfix
        seed_postfix = f'_seed{args.seed}' if args.seed is not None else ''
        pre_postfix = f'_ep{args.epochs}_bs{args.batch_size}_lora{args.lora_r}-{args.lora_alpha}{pre_postfix}{seed_postfix}'
        _model_name = '_' + re.sub(r'[aeiou]', '', args.model_name)
        _concat_dataset = '_' + re.sub(r'[aeiouhtds]', '', args.concat_dataset) + str(args.increase_concat_dataset) if args.concat_dataset != '' else ''

        # set result_dir
        args.result_dir = os.path.join(args.result_dir, f'{args.dataset}{_concat_dataset}{_model_name}{pre_postfix}')

        # create directory
        os.makedirs(args.result_dir, exist_ok=True)

        # save args
        with open(os.path.join(args.result_dir, 'args.json'), 'w') as wf:
            json.dump(vars(args), wf, indent=4)
    else:
        # for evaluation
        if not os.path.isfile(args.resume):
            raise ValueError('Invalid resume', args.resume)

        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        args_dict = ckpt['args']

        args.model_name = args_dict['model_name']
        args.decoder_name = args_dict['decoder_name']

        args.lora_r = args_dict['lora_r']
        args.lora_alpha = args_dict['lora_alpha']
        args.lora_target_modules = args_dict['lora_target_modules']
        args.head_name = args_dict['head_name']

        args.finetune_decoder = args_dict['finetune_decoder']
        args.decoder_lora_r = args_dict['decoder_lora_r']
        args.decoder_lora_alpha = args_dict['decoder_lora_alpha']
        args.decoder_lora_target_modules = args_dict['decoder_lora_target_modules']

        args.multimodal_tokens = args_dict['multimodal_tokens']
        args.pool2x2 = args_dict.get('pool2x2', False)

        if len(args.dataset) == 0:
            args.dataset = args_dict['dataset']

        # create directory
        os.makedirs(args.result_dir, exist_ok=True)

    # start main
    main(args)
