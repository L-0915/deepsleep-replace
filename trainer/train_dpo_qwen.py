"""Qwen2.5-0.5B DPO (Direct Preference Optimization) Training.

Mirrors train_dpo.py logic but uses HuggingFace Qwen model.
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
import logging

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, SkipBatchSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-DPO-Qwen")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


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
# Checkpoint save/load
# =============================================================================


def save_checkpoint(model, optimizer, scaler, epoch, step, save_dir):
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
                logits = model(x_chosen).logits
                policy_chosen_logps = get_batch_logps(logits, y_chosen, mask_chosen)
                del logits

                logits = model(x_rejected).logits
                policy_rejected_logps = get_batch_logps(logits, y_rejected, mask_rejected)
                del logits

                logits = ref_model(x_chosen).logits
                ref_chosen_logps = get_batch_logps(logits, y_chosen, mask_chosen)
                del logits

                logits = ref_model(x_rejected).logits
                ref_rejected_logps = get_batch_logps(logits, y_rejected, mask_rejected)
                del logits
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


# =============================================================================
# Training
# =============================================================================


def train_epoch(epoch, loader, steps_per_epoch, args, model, ref_model, optimizer, scaler,
                autocast_ctx, train_log, global_step_offset):
    start_time = time.time()
    total_steps = args.epochs * steps_per_epoch
    zero_loss_count = 0

    for step_in_epoch, batch in enumerate(loader, start=1):
        global_step = global_step_offset + step_in_epoch

        x_chosen = batch['x_chosen'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)

        lr = get_lr(global_step - 1, total_steps, args.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        with autocast_ctx:
            # Policy forward — release logits immediately, keep only scalar logps
            logits = model(x_chosen).logits
            policy_chosen_logps = get_batch_logps(logits, y_chosen, mask_chosen)
            del logits

            logits = model(x_rejected).logits
            policy_rejected_logps = get_batch_logps(logits, y_rejected, mask_rejected)
            del logits

            # Reference forward (no gradients)
            with torch.no_grad():
                logits = ref_model(x_chosen).logits
                ref_chosen_logps = get_batch_logps(logits, y_chosen, mask_chosen)
                del logits

                logits = ref_model(x_rejected).logits
                ref_rejected_logps = get_batch_logps(logits, y_rejected, mask_rejected)
                del logits

            loss, accuracy = compute_dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.dpo_beta,
            )
            loss = loss / args.accumulation_steps

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
                'Epoch:[%d/%d](%d/%d), loss: %.4f, acc: %.4f, lr: %.2e, eta: %.1fmin',
                epoch + 1, args.epochs, global_step, total_steps,
                cur_loss, accuracy.item(), lr, eta,
            )
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
                return global_step_offset + step_in_epoch

        del batch, loss

    # Flush remaining gradients
    if steps_per_epoch % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return global_step_offset + steps_per_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B DPO")
    parser.add_argument("--sft_model_path", type=str, default="out/sft_qwen/final_model",
                        help="Path to Qwen SFT model (HF format)")
    parser.add_argument("--save_dir", type=str, default="out/dpo_qwen")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
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
    logger.info("Loading policy model from %s", args.sft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, trust_remote_code=True)
    model = model.to(args.device)

    # Reference model (frozen copy)
    logger.info("Loading reference model from %s", args.sft_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, trust_remote_code=True)
    ref_model = ref_model.to(args.device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model: %.2fM params", total_params)

    # Gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

    # --- Dataset ---
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # Fix bos/eos for Qwen tokenizer (same as train_sft_qwen.py)
    train_ds.bos_id = tokenizer('<|im_start|>assistant\n', add_special_tokens=False).input_ids
    train_ds.eos_id = tokenizer('<|im_end|>\n', add_special_tokens=False).input_ids

    # --- Resume ---
    start_epoch, start_step = 0, 0
    if args.from_resume:
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

    # --- Log training config ---
    torch.manual_seed(args.seed)
    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps_est = args.epochs * steps_per_epoch
    logger.info("Dataset: %d samples, max_length=%d", len(train_ds), args.max_seq_len)
    logger.info("Training config: %d epochs × %d steps = %d total (batch=%d, accum=%d, eff=%d, beta=%.2f)",
                args.epochs, steps_per_epoch, total_steps_est,
                args.batch_size, args.accumulation_steps, args.batch_size * args.accumulation_steps, args.dpo_beta)

    # --- Training ---
    train_log = []
    eval_log = []
    total_start = time.time()
    cumulative_steps = 0

    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if epoch == start_epoch and start_step > 0 else 0
        batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=True)
        steps_per_epoch = len(loader)
        logger.info('Epoch [%d/%d]: %d steps', epoch + 1, args.epochs, steps_per_epoch)
        cumulative_steps = train_epoch(
            epoch, loader, steps_per_epoch, args, model, ref_model, optimizer, scaler,
            autocast_ctx, train_log, cumulative_steps,
        )

        # Eval after each epoch
        evaluate(model, ref_model, train_ds, args.device, autocast_ctx, args.dpo_beta, eval_log, cumulative_steps)

    # --- Save final model ---
    final_path = os.path.join(args.save_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
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
        "model": "Qwen2.5-0.5B",
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
            "sft_model_path": args.sft_model_path,
        },
    }
    with open(os.path.join(args.save_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("DPO done. %d steps, %.1f minutes, final_loss=%.4f, final_acc=%.4f",
                cumulative_steps, total_time / 60,
                final_loss or 0, final_acc or 0)
