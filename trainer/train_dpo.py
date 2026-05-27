"""DeepSleep DPO (Direct Preference Optimization) Training.

Features:
  - JSONL logging (train_log.jsonl, eval_log.jsonl)
  - Accuracy tracking (chosen vs rejected logps)
  - Final report.json
  - Checkpoint/resume support
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
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from model.model_deepsleep import DeepSleepConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler,
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-DPO")


def get_batch_logps(logits, labels, mask=None):
    """Compute log probabilities for the response tokens."""
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        per_token_logps = per_token_logps * mask
    return per_token_logps.sum(dim=-1)


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                      ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss (sigmoid variant). Returns loss and accuracy."""
    logits_diff = (
        policy_chosen_logps - ref_chosen_logps
        - policy_rejected_logps + ref_rejected_logps
    )
    loss = -F.logsigmoid(beta * logits_diff)
    accuracy = (logits_diff > 0).float().mean()
    return loss.mean(), accuracy


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


# =============================================================================
# Eval
# =============================================================================


def evaluate(model, ref_model, dataset, device, autocast_ctx, beta, eval_log, step):
    eval_size = min(200, len(dataset))
    indices = list(range(eval_size))
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            x_chosen = batch['x_chosen'].unsqueeze(0).to(device)
            y_chosen = batch['y_chosen'].unsqueeze(0).to(device)
            mask_chosen = batch['mask_chosen'].unsqueeze(0).to(device)
            x_rejected = batch['x_rejected'].unsqueeze(0).to(device)
            y_rejected = batch['y_rejected'].unsqueeze(0).to(device)
            mask_rejected = batch['mask_rejected'].unsqueeze(0).to(device)
            with autocast_ctx:
                policy_chosen_logits = model(x_chosen).logits
                policy_chosen_logps = get_batch_logps(policy_chosen_logits, y_chosen, mask_chosen)
                policy_rejected_logits = model(x_rejected).logits
                policy_rejected_logps = get_batch_logps(policy_rejected_logits, y_rejected, mask_rejected)
                ref_chosen_logits = ref_model(x_chosen).logits
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, y_chosen, mask_chosen)
                ref_rejected_logits = ref_model(x_rejected).logits
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, y_rejected, mask_rejected)
                loss, acc = compute_dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=beta,
                )
                total_loss += loss.item()
                total_acc += acc.item()
            count += 1
            if count >= 50:
                break
    avg_loss = total_loss / max(count, 1)
    avg_acc = total_acc / max(count, 1)
    model.train()
    eval_log.append({"step": step, "eval_loss": avg_loss, "eval_accuracy": avg_acc})
    logger.info('Eval @ step %d: eval_loss=%.4f, eval_accuracy=%.4f', step, avg_loss, avg_acc)
    return avg_loss, avg_acc


def train_epoch(epoch, loader, iters, start_step=0, wandb=None,
                train_log=None, global_step_offset=0):
    start_time = time.time()
    last_step = start_step
    total_steps = args.epochs * iters
    zero_loss_count = 0
    for step, batch in enumerate(loader, start=start_step + 1):
        x_chosen = batch['x_chosen'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        last_step = step
        global_step = global_step_offset + (step - start_step)
        lr = get_lr(global_step, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            # Policy forward
            policy_chosen_logits = model(x_chosen).logits
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, y_chosen, mask_chosen)
            policy_rejected_logits = model(x_rejected).logits
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, y_rejected, mask_rejected)

            # Reference forward (no gradients)
            with torch.no_grad():
                ref_chosen_logits = ref_model(x_chosen).logits
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, y_chosen, mask_chosen)
                ref_rejected_logits = ref_model(x_rejected).logits
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, y_rejected, mask_rejected)

            loss, accuracy = compute_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.dpo_beta,
            )
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend = time.time() - start_time
            cur_loss = loss.item() * args.accumulation_steps
            eta = spend / max(step - start_step, 1) * (iters - step) // 60
            logger.info(
                'Epoch:[%d/%d](%d/%d), loss: %.4f, acc: %.4f, lr: %.2e, eta: %.1fmin',
                epoch + 1, args.epochs, global_step, total_steps,
                cur_loss, accuracy.item(), lr, eta,
            )
            if train_log is not None:
                train_log.append({
                    "step": global_step, "loss": cur_loss,
                    "accuracy": accuracy.item(), "lr": lr,
                })

            # Early stopping: loss converges to 0
            if cur_loss < 1e-4:
                zero_loss_count += 1
            else:
                zero_loss_count = 0
            if zero_loss_count >= 3:
                logger.info('Early stopping: loss < 1e-4 for %d consecutive logs, stopping at step %d',
                            zero_loss_count, global_step)
                break

            if wandb:
                wandb.log({"dpo_loss": cur_loss, "accuracy": accuracy.item(), "lr": lr})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model,
                optimizer=optimizer, scaler=scaler, epoch=epoch, step=step,
                wandb=wandb, save_dir=args.save_dir,
            )
            model.train()

        del batch, loss

    # Flush remaining gradients
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return global_step_offset + (last_step - start_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSleep DPO")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--save_weight", default="dpo", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=200)
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
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="SFT checkpoint path (used for both policy and reference)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeepSleep-DPO")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    args = parser.parse_args()
    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)
        logger.info("Loaded config from %s", args.config)

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = DeepSleepConfig(
        d_model=args.hidden_size, n_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe), num_experts=args.num_experts,
        num_shared_experts=args.num_shared_experts, top_k=args.num_experts_per_tok,
        vocab_size=args.vocab_size, max_position_embeddings=args.max_seq_len,
    )

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)

    # Policy model (trainable)
    logger.info("Loading policy model from %s", args.sft_checkpoint)
    model, tokenizer = init_model(lm_config, args.sft_checkpoint, args.tokenizer_path, args.device)
    # Reference model (frozen copy)
    logger.info("Loading reference model from %s", args.sft_checkpoint)
    ref_model, _ = init_model(lm_config, args.sft_checkpoint, args.tokenizer_path, args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir) if args.from_resume else None
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        if ckp_data.get('optimizer'):
            optimizer.load_state_dict(ckp_data['optimizer'])
        start_epoch = ckp_data.get('epoch', 0)
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb as wb
        wb.init(project=args.wandb_project, name=f"dpo-d{args.hidden_size}")
        wandb = wb

    # --- Log training config ---
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps_est = args.epochs * steps_per_epoch
    logger.info("Dataset: %d samples, max_length=%d", len(train_ds), args.max_seq_len)
    logger.info("Training config: %d epochs × %d steps = %d total (batch=%d, accum=%d, eff=%d, beta=%.2f)",
                args.epochs, steps_per_epoch, total_steps_est,
                args.batch_size, args.accumulation_steps, args.batch_size * args.accumulation_steps, args.dpo_beta)

    train_log = []
    eval_log = []
    total_start = time.time()
    cumulative_steps = 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        cumulative_steps = train_epoch(
            epoch, loader, len(loader) + skip, start_step if epoch == start_epoch else 0,
            wandb, train_log, cumulative_steps,
        )

        # Eval after each epoch
        evaluate(model, ref_model, train_ds, args.device, autocast_ctx, args.dpo_beta, eval_log, cumulative_steps)

    if dist.is_initialized():
        dist.destroy_process_group()

    # --- Save final model ---
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    final_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save(raw_model.state_dict(), final_path)
    logger.info("Final DPO model saved to %s", final_path)

    # --- Save logs ---
    old_train = _load_jsonl(os.path.join(args.save_dir, "train_log.jsonl"))
    old_eval = _load_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"))
    _save_jsonl(os.path.join(args.save_dir, "train_log.jsonl"), _merge_logs(old_train, train_log))
    _save_jsonl(os.path.join(args.save_dir, "eval_log.jsonl"), _merge_logs(old_eval, eval_log))

    # --- Save report ---
    total_time = time.time() - total_start
    final_loss = train_log[-1]["loss"] if train_log else None
    final_acc = train_log[-1]["accuracy"] if train_log else None
    report = {
        "model": "DeepSleep-MoE",
        "total_steps": cumulative_steps,
        "epochs": args.epochs,
        "dpo_beta": args.dpo_beta,
        "seed": args.seed,
        "total_time_hours": round(total_time / 3600, 2),
        "final_loss": round(final_loss, 6) if final_loss else None,
        "final_accuracy": round(final_acc, 4) if final_acc else None,
        "config": {
            "batch_size": args.batch_size,
            "effective_batch": args.batch_size * args.accumulation_steps,
            "learning_rate": args.learning_rate,
            "dpo_beta": args.dpo_beta,
            "sft_checkpoint": args.sft_checkpoint,
        },
    }
    with open(os.path.join(args.save_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("DPO done. %d steps, %.1f minutes, final_loss=%.4f, final_acc=%.4f",
                cumulative_steps, total_time / 60,
                final_loss or 0, final_acc or 0)
