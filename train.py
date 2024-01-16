
import argparse
from transformers import AutoTokenizer, get_scheduler, SchedulerType
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random
import datetime
from model.modeling import Model
from model.dataset import generate_dataset
from model.configuration import ModelConfig

def train(
    model,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator,
    lr_scheduler
) -> float:
    model.train()
    log = {
        'loss': 0
    }
    with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                log['loss'] += loss.item()
                if accelerator.is_main_process:
                    pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                    pbar.set_postfix(OrderedDict(
                        loss=loss.item(),
                        lr=optimizer.optimizer.param_groups[0]['lr']
                    ))
    return {k: v/len(loader) for k, v in log.items()}

def valid(model,
    loader: DataLoader,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.eval()
    log = {
        'loss': 0
    }
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    log['loss'] += loss.item()
                    if accelerator.is_main_process:
                        pbar.set_description(f'[Epoch {epoch}] [VALID]')
                        pbar.set_postfix(OrderedDict(
                            loss=loss.item()
                        ))
    return {k: v/len(loader) for k, v in log.items()}

def main(args):
    config = json.load(open(os.path.join(args.restore_dir, 'training_state.json'))) if args.restore_dir else {'argparse': dict()}
    current_epoch = config.get('current_epoch', -1) + 1
    min_valid_loss = config.get('min_valid_loss', float('inf'))
    seed = config['argparse'].get('seed', args.seed)
    max_len = config['argparse'].get('max_len', args.max_len)
    log_dict = json.load(open(os.path.join(args.restore_dir, '../log.json'))) if args.restore_dir else dict()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    if args.restore_dir is not None:
        model = Model.from_pretrained(args.restore_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        config = ModelConfig(
            model_id=args.model_id
        )
        model = Model(config)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = generate_dataset(
        input_file=args.train_input,
        tokenizer=tokenizer,
        max_len=max_len
    )
    valid_dataset = generate_dataset(
        input_file=args.valid_input,
        tokenizer=tokenizer,
        max_len=max_len
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * args.epochs,
    )
    best_path = os.path.join(args.outdir, 'best')
    last_path = os.path.join(args.outdir, 'last')
    os.makedirs(best_path, exist_ok=True)
    os.makedirs(last_path, exist_ok=True)
    tokenizer.save_pretrained(best_path)
    tokenizer.save_pretrained(last_path)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    accelerator.wait_for_everyone()
    for epoch in range(current_epoch, args.epochs):
        train_log = train(model, train_loader, optimizer, epoch, accelerator, lr_scheduler)
        valid_log = valid(model, valid_loader, epoch, accelerator)
        log_dict[f'Epoch {epoch}'] = {
            'train_log': train_log,
            'valid_log': valid_log,
            'time': str(datetime.datetime.now())
        }
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if min_valid_loss > valid_log['loss']:
                # Save the best chckpoint
                accelerator.unwrap_model(model).save_pretrained(best_path)
                min_valid_loss = valid_log['loss']
                training_state = {
                    'current_epoch': epoch,
                    'min_valid_loss': min_valid_loss,
                    'argparse': args.__dict__
                }
                with open(os.path.join(best_path, 'training_state.json'), 'w') as fp:
                    json.dump(training_state, fp, indent=4)
            # Save checkpoint as the last checkpoint for each epoch
            accelerator.unwrap_model(model).save_pretrained(last_path)
            training_state = {
                'current_epoch': epoch,
                'min_valid_loss': min_valid_loss,
                'argparse': args.__dict__
            }
            with open(os.path.join(last_path, 'training_state.json'), 'w') as fp:
                    json.dump(training_state, fp, indent=4)
            with open(os.path.join(args.outdir, 'log.json'), 'w') as fp:
                json.dump(log_dict, fp, indent=4)
    print('Finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', required=True)
    parser.add_argument('--valid_input', required=True)
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--outdir', default='models/sample/')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--restore_dir', default=None)
    parser.add_argument('--num_warmup_steps', type=int, default=500)
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
