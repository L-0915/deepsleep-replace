"""DeepSleep Supervised Fine-Tuning with logging and training curves.

Features:
  - JSONL logging (train_log.jsonl, eval_log.jsonl)
  - Training curves PNG (loss, lr) with resume merge support
  - TensorBoard logging
  - Periodic sample generation during training
  - Resume from checkpoint with full state restoration
"""

import os
import sys
import json
import time
import math
import logging

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_deepsleep import DeepSleepConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings('ignore')

# Structured logging (same as pretrain script)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-SFT")

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Suppress torch.compile graph tracing (set TORCH_LOGS="" in run script)


# =============================================================================
# Training Logger (JSONL + curves, with resume merge)
# =============================================================================


def _load_jsonl(path):
    records = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def _merge_logs(old, new, key="step"):
    existing = {r[key] for r in new if key in r}
    merged = [r for r in old if key not in r or r[key] not in existing]
    merged.extend(new)
    merged.sort(key=lambda r: r.get(key, 0))
    return merged


def _save_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_training_curves(save_dir, train_log, eval_log):
    try:
        from trainer.plot_utils import plot_sft_curves
        path = plot_sft_curves(save_dir, train_log, eval_log, title="DeepSleep SFT Training Curves")
        logger.info("Training curves saved to %s", path)
    except ImportError:
        logger.info("matplotlib not installed, skipping training_curves.png")


# =============================================================================
# Training
# =============================================================================


def generate_sample(model, tokenizer, device, prompts, max_new_tokens=100):
    """Generate sample text for visual inspection."""
    model.eval()
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(output[0], skip_special_tokens=False)
            logger.info("  [%s] -> %s", prompt, text)
        except Exception as e:
            logger.warning("  Generation failed for '%s': %s", prompt, e)
    model.train()


def evaluate(model, dataset, args, autocast_ctx, eval_log, step):
    """Compute eval loss on a subset of SFT data."""
    eval_size = min(500, len(dataset))
    indices = list(range(len(dataset) - eval_size, len(dataset)))
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in indices:
            input_ids, labels = dataset[idx]
            input_ids = input_ids.unsqueeze(0).to(args.device)
            labels = labels.unsqueeze(0).to(args.device)
            with autocast_ctx:
                res = model(input_ids, labels=labels)
                total_loss += (res.loss + res.aux_loss).item()
            count += 1
            if count >= 100:
                break
    avg_loss = total_loss / max(count, 1)
    model.train()
    eval_log.append({"step": step, "eval_loss": avg_loss})
    logger.info('Eval @ step %d: eval_loss=%.4f', step, avg_loss)
    if is_main_process() and args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = getattr(train_epoch, '_writer', None)
        if writer:
            writer.add_scalar("eval/loss", avg_loss, step)
    return avg_loss


def train_epoch(epoch, loader, steps_per_epoch, args, model, optimizer, scaler, autocast_ctx,
                train_log, global_step_offset, tokenizer):
    start_time = time.time()
    last_step = global_step_offset
    total_steps = args.epochs * steps_per_epoch

    for step_in_epoch, (input_ids, labels) in enumerate(loader, start=1):
        global_step = global_step_offset + step_in_epoch
        last_step = global_step

        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        lr = get_lr(global_step - 1, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step_in_epoch % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        if global_step % args.log_interval == 0 or step_in_epoch == steps_per_epoch:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            cur_aux = res.aux_loss.item() if res.aux_loss is not None else 0.0
            eta = spend / max(step_in_epoch, 1) * (steps_per_epoch - step_in_epoch) // 60
            logger.info(
                'Epoch:[%d/%d](%d/%d), loss: %.4f, aux: %.4f, lr: %.2e, eta: %.1fmin',
                epoch + 1, args.epochs, global_step, total_steps,
                cur_loss, cur_aux, lr, eta,
            )
            train_log.append({"step": global_step, "loss": cur_loss, "aux_loss": cur_aux, "lr": lr})

            # TensorBoard
            writer = getattr(train_epoch, '_writer', None)
            if writer is not None:
                writer.add_scalar("train/loss", cur_loss, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/aux_loss", cur_aux, global_step)

        # Checkpoint + sample generation
        if (global_step % args.save_interval == 0 or step_in_epoch == steps_per_epoch) and is_main_process():
            model.eval()
            ckpt_dir = os.path.join(args.save_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            lm_checkpoint(
                args._lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=global_step,
                save_dir=ckpt_dir,
            )
            # Keep only 2 latest checkpoints
            ckpts = sorted(
                [d for d in os.listdir(args.save_dir)
                 if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.save_dir, d))],
                key=lambda d: int(d.split("-")[1])
            )
            for old in ckpts[:-2]:
                import shutil
                shutil.rmtree(os.path.join(args.save_dir, old))
                logger.info("Removed old checkpoint: %s", old)
            if tokenizer:
                logger.info("--- Sample Generation (step %d) ---", global_step)
                generate_sample(model, tokenizer, args.device,
                                ["睡眠不好怎么办", "失眠的原因", "Sleep is important"])
            model.train()

        del input_ids, labels, res, loss

    # Flush remaining gradients
    if steps_per_epoch % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSleep SFT")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", default="sft", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)
    # Model
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=1, type=int, choices=[0, 1])
    parser.add_argument("--num_experts", default=8, type=int)
    parser.add_argument("--num_shared_experts", default=0, type=int)
    parser.add_argument("--num_experts_per_tok", default=2, type=int)
    parser.add_argument("--vocab_size", default=7200, type=int)
    parser.add_argument("--max_seq_len", default=2048, type=int)
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--from_weight", type=str, required=True, help="Pretrain checkpoint path (.pth)")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="DeepSleep-SFT")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    args = parser.parse_args()
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = DeepSleepConfig(
        d_model=args.hidden_size, n_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe), num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts, top_k=args.num_experts_per_tok,
        vocab_size=args.vocab_size, max_position_embeddings=args.max_seq_len,
    )
    args._lm_config = lm_config

    # Resume: find latest checkpoint subdirectory automatically
    ckp_data = None
    if args.from_resume:
        ckpts = sorted(
            [d for d in os.listdir(args.save_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.save_dir, d))],
            key=lambda d: int(d.split("-")[1])
        )
        if ckpts:
            latest = os.path.join(args.save_dir, ckpts[-1])
            ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=latest)
            if ckp_data:
                logger.info("Resumed from %s (step %d)", latest, ckp_data.get('step', 0))
        if not ckp_data:
            logger.warning("No checkpoint found in %s, starting from scratch", args.save_dir)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wb.init(project=args.wandb_project, name=f"sft-d{args.hidden_size}", id=wandb_id, resume='must' if wandb_id else None)
        wandb = wb

    model, tokenizer = init_model(lm_config, args.from_weight, args.tokenizer_path, args.device)

    # torch.compile for memory efficiency + speed (same as pretrain/CPT)
    model = torch.compile(model, backend="inductor")
    logger.info("torch.compile enabled (inductor backend)")

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    logger.info("Dataset: %d samples, max_length=%d", len(train_ds), args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # --- TensorBoard writer (create eagerly, same as pretrain) ---
    tb_writer = None
    if is_main_process() and args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))
        train_epoch._writer = tb_writer
        logger.info("TensorBoard logging to %s", os.path.join(args.save_dir, "runs"))

    # --- Log training config ---
    steps_per_epoch_est = len(train_ds) // args.batch_size
    total_steps_est = args.epochs * steps_per_epoch_est
    logger.info("Training config: %d epochs × %d steps/epoch = %d total steps (batch=%d, accum=%d, eff_batch=%d)",
                args.epochs, steps_per_epoch_est, total_steps_est,
                args.batch_size, args.accumulation_steps, args.batch_size * args.accumulation_steps)

    # --- Load existing logs (resume support) ---
    train_log = []
    eval_log = []

    # --- Training ---
    total_start = time.time()
    cumulative_steps = 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        steps_per_epoch = len(loader)
        if skip:
            logger.info('Epoch [%d/%d]: skip %d steps', epoch + 1, args.epochs, skip)
        else:
            logger.info('Epoch [%d/%d]: %d steps', epoch + 1, args.epochs, steps_per_epoch)
        cumulative_steps = train_epoch(epoch, loader, steps_per_epoch, args, model, optimizer, scaler,
                                autocast_ctx, train_log, cumulative_steps, tokenizer)

        # Eval after each epoch
        if is_main_process():
            evaluate(model, train_ds, args, autocast_ctx, eval_log, cumulative_steps)

    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()

    # --- Save final model ---
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(raw_model.state_dict(), final_path)
    logger.info("Final model saved to %s", final_path)

    # --- Cleanup intermediate checkpoints, keep only final ---
    import shutil
    for entry in os.listdir(args.save_dir):
        entry_path = os.path.join(args.save_dir, entry)
        if entry.startswith("checkpoint-") and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            logger.info("Removed intermediate checkpoint: %s", entry_path)
    logger.info("Cleanup done. Only final model retained at %s/final_model.pth", args.save_dir)

    # --- Save logs and curves ---
    old_train = _load_jsonl(os.path.join(args.save_dir, "train_log.jsonl"))
    old_eval = _load_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"))
    merged_train = _merge_logs(old_train, train_log)
    merged_eval = _merge_logs(old_eval, eval_log)

    _save_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), merged_train)
    _save_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"), merged_eval)
    _save_training_curves(args.save_dir, merged_train, merged_eval)

    # --- Save report ---
    total_time = time.time() - total_start
    final_loss = train_log[-1]["loss"] if train_log else None
    report = {
        "total_steps": cumulative_steps,
        "epochs": args.epochs,
        "total_time_hours": round(total_time / 3600, 2),
        "final_loss": round(final_loss, 6) if final_loss else None,
        "config": {
            "batch_size": args.batch_size,
            "effective_batch": args.batch_size * args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "from_weight": args.from_weight,
        },
    }
    with open(os.path.join(args.save_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if dist.is_initialized():
        dist.destroy_process_group()

    logger.info("SFT done. %d steps, %.1f minutes", cumulative_steps, total_time / 60)
