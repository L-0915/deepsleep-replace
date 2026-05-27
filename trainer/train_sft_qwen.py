"""Qwen2.5-0.5B SFT — identical hyperparameters to DeepSleep SFT for fair comparison.

Features:
  - Resume from checkpoint with full state restoration
  - TensorBoard logging (eager initialization)
  - JSONL logging (train_log.jsonl, eval_log.jsonl)
  - Periodic sample generation during training
"""

import os
import sys
import json
import time
import math
import logging

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, SkipBatchSampler

# Structured logging (same as DeepSleep SFT)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-SFT-Qwen")

# Performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# =============================================================================
# Checkpoint save/load (Qwen-specific, saves HF state_dict)
# =============================================================================


def save_checkpoint(model, optimizer, scaler, epoch, step, save_dir):
    """Save full training state for resume."""
    import gc
    os.makedirs(save_dir, exist_ok=True)
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    resume_data = {
        'model': state_dict,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'step': step,
    }
    if scaler:
        resume_data['scaler'] = scaler.state_dict()
    path = os.path.join(save_dir, "resume.pth")
    tmp = path + '.tmp'
    torch.save(resume_data, tmp)
    os.replace(tmp, path)
    del state_dict, resume_data
    gc.collect()
    torch.cuda.empty_cache()


def load_checkpoint(save_dir):
    """Load training state from checkpoint directory. Returns None if not found."""
    path = os.path.join(save_dir, "resume.pth")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location='cpu', weights_only=True)
    logger.info("Resumed from checkpoint at %s", path)
    return data


# =============================================================================
# Logging helpers
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
        path = plot_sft_curves(save_dir, train_log, eval_log, title="Qwen2.5-0.5B SFT Training Curves")
        logger.info("Training curves saved to %s", path)
    except ImportError:
        logger.info("matplotlib not installed, skipping training_curves.png")


# =============================================================================
# Eval & Generation
# =============================================================================


def evaluate(model, dataset, device, autocast_ctx, eval_log, step):
    eval_size = min(500, len(dataset))
    indices = list(range(len(dataset) - eval_size, len(dataset)))
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in indices:
            input_ids, labels = dataset[idx]
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with autocast_ctx:
                res = model(input_ids, labels=labels)
                total_loss += res.loss.item()
            count += 1
            if count >= 100:
                break
    avg_loss = total_loss / max(count, 1)
    model.train()
    eval_log.append({"step": step, "eval_loss": avg_loss})
    logger.info('Eval @ step %d: eval_loss=%.4f', step, avg_loss)
    return avg_loss


def generate_sample(model, tokenizer, device, prompts, max_new_tokens=100):
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


# =============================================================================
# Training
# =============================================================================


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
            loss = res.loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step_in_epoch % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if global_step % args.log_interval == 0 or step_in_epoch == steps_per_epoch:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            eta = spend / max(step_in_epoch, 1) * (steps_per_epoch - step_in_epoch) // 60
            logger.info(
                'Epoch:[%d/%d](%d/%d), loss: %.4f, lr: %.2e, eta: %.1fmin',
                epoch + 1, args.epochs, global_step, total_steps,
                cur_loss, lr, eta,
            )
            train_log.append({"step": global_step, "loss": cur_loss, "lr": lr})

            # TensorBoard
            tb_writer = getattr(train_epoch, '_writer', None)
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", cur_loss, global_step)
                tb_writer.add_scalar("train/lr", lr, global_step)

        if (global_step % args.save_interval == 0 or step_in_epoch == steps_per_epoch):
            model.eval()
            ckpt_dir = os.path.join(args.save_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_checkpoint(model, optimizer, scaler, epoch, global_step, ckpt_dir)
            logger.info("Checkpoint saved to %s", ckpt_dir)
            # Keep only 1 latest checkpoint
            ckpts = sorted(
                [d for d in os.listdir(args.save_dir)
                 if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.save_dir, d))],
                key=lambda d: int(d.split("-")[1])
            )
            for old in ckpts[:-1]:
                import shutil
                shutil.rmtree(os.path.join(args.save_dir, old))
                logger.info("Removed old checkpoint: %s", old)
            if tokenizer:
                logger.info("--- Sample Generation (step %d) ---", global_step)
                generate_sample(model, tokenizer, args.device,
                                ["睡眠不好怎么办", "失眠的原因", "Sleep is important"])
            model.train()

        del input_ids, labels, res, loss

    if steps_per_epoch % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return last_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B SFT")
    parser.add_argument("--model_path", type=str, default="/root/eb-public/huggingface-models/Qwen/Qwen2.5-0.5B")
    parser.add_argument("--save_dir", type=str, default="out/sft_qwen/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_tensorboard", action="store_true", default=True)
    parser.add_argument("--config", type=str, default=None, help="YAML config file (for shared hyperparams)")
    args = parser.parse_args()
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)
        logger.info("Loaded config from %s", args.config)

    os.makedirs(args.save_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # --- Load Qwen model & tokenizer ---
    logger.info("Loading model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(args.device)

    # Gradient checkpointing for memory efficiency (torch.compile causes CUDA errors with Qwen)
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Model: %.2fM total, %.2fM trainable", total_params, trainable_params)

    # === Identical optimizer to DeepSleep SFT ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    # --- Dataset (same data, Qwen tokenizer applies its own chat template) ---
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # Fix: Qwen's bos_token=None, so SFTDataset can't find the assistant pattern.
    # Override to use <|im_start|> / <|im_end|> directly (same ChatML format).
    train_ds.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
    train_ds.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids
    logger.info("Dataset: %d samples, max_length=%d", len(train_ds), args.max_seq_len)

    # --- Resume from checkpoint ---
    start_epoch, start_step = 0, 0
    if args.from_resume:
        # Find latest checkpoint subdirectory
        ckpt_dirs = sorted(
            [d for d in os.listdir(args.save_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.save_dir, d))],
            key=lambda d: int(d.split("-")[1])
        )
        ckpt_path = os.path.join(args.save_dir, ckpt_dirs[-1]) if ckpt_dirs else args.save_dir
        ckp_data = load_checkpoint(ckpt_path)
    else:
        ckp_data = None
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)
        logger.info("Resumed: epoch=%d, step=%d", start_epoch, start_step)

    # --- TensorBoard writer (create eagerly) ---
    tb_writer = None
    if args.use_tensorboard:
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

    # --- Training ---
    train_log = []
    eval_log = []
    total_start = time.time()
    cumulative_steps = 0

    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        steps_per_epoch = len(loader)
        if skip:
            logger.info('Epoch [%d/%d]: skip %d steps', epoch + 1, args.epochs, skip)
        else:
            logger.info('Epoch [%d/%d]: %d steps', epoch + 1, args.epochs, steps_per_epoch)
        cumulative_steps = train_epoch(epoch, loader, steps_per_epoch, args, model, optimizer, scaler,
                                autocast_ctx, train_log, cumulative_steps, tokenizer)

        # Eval after each epoch
        evaluate(model, train_ds, args.device, autocast_ctx, eval_log, cumulative_steps)

    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()

    # --- Save final model ---
    final_path = os.path.join(args.save_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Final model saved to %s", final_path)

    # --- Cleanup intermediate checkpoints, keep only final ---
    import shutil
    for entry in os.listdir(args.save_dir):
        entry_path = os.path.join(args.save_dir, entry)
        if entry.startswith("checkpoint-") and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            logger.info("Removed intermediate checkpoint: %s", entry_path)
    logger.info("Cleanup done. Only final model retained at %s", final_path)

    # --- Save logs and curves ---
    old_train = _load_jsonl(os.path.join(args.save_dir, "train_log.jsonl"))
    old_eval = _load_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"))
    _save_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), _merge_logs(old_train, train_log))
    _save_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"), _merge_logs(old_eval, eval_log))
    _save_training_curves(args.save_dir, train_log, eval_log)

    # --- Report ---
    total_time = time.time() - total_start
    final_loss = train_log[-1]["loss"] if train_log else None
    report = {
        "model": "Qwen2.5-0.5B",
        "total_steps": cumulative_steps,
        "epochs": args.epochs,
        "total_time_hours": round(total_time / 3600, 2),
        "final_loss": round(final_loss, 6) if final_loss else None,
        "config": {
            "batch_size": args.batch_size,
            "effective_batch": args.batch_size * args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "model_path": args.model_path,
        },
    }
    with open(os.path.join(args.save_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("SFT done. %d steps, %.1f minutes", cumulative_steps, total_time / 60)
