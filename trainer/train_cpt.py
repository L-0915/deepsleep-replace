"""DeepSleep Continued Pretraining (CPT) on Medical QA Data.

Industrial-grade continued pretraining that loads the pretrained model
and trains on the Malikeh1375/medical-question-answering-datasets.

=== Key Features ===

Resume-Safe Training Curves:
  - On resume, loads existing train_log.jsonl + eval_log.jsonl
  - Appends new metrics from the resumed step
  - Deduplicates by step number (no overlap at resume boundary)
  - Generates complete, continuous training_curves.png

Checkpoint Management:
  - save_total_limit=2 (only keeps 2 most recent checkpoints)
  - Full resume support: model, optimizer, scheduler, RNG, global step

Data:
  - Local parquet files (no network needed)
  - Sequence packing for maximum throughput
  - Deterministic shuffle with seed
"""

import os
import sys
import math
import json
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
)

from model.model_deepsleep import DeepSleepConfig, DeepSleepForCausalLM
from dataset.medical_qa_dataset import MedicalQADataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DeepSleep-CPT")


# =============================================================================
# MoE-aware Trainer
# =============================================================================


class MoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)

        loss = outputs.loss
        if outputs.aux_loss is not None and outputs.aux_loss.requires_grad:
            loss = loss + outputs.aux_loss

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Callbacks
# =============================================================================

MEDICAL_PROMPTS = [
    "The patient presents with",
    "Common symptoms of insomnia include",
    "Treatment for sleep apnea typically involves",
    "The recommended dosage for",
]


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts=None, max_new_tokens=100):
        self.tokenizer = tokenizer
        self.prompts = prompts or MEDICAL_PROMPTS
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return

        device = next(model.parameters()).device
        model.eval()
        logger.info("=" * 60)
        logger.info("Sample Generation (step %d)", state.global_step)
        logger.info("=" * 60)

        for prompt in self.prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                logger.info("  [%s] -> %s", prompt, text)
            except Exception as e:
                logger.warning("  Generation failed for '%s': %s", prompt, e)

        model.train()


class PerplexityCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
                metrics["eval_perplexity"] = ppl
                logger.info("Eval perplexity: %.2f", ppl)
            except OverflowError:
                metrics["eval_perplexity"] = float("inf")
                logger.info("Eval perplexity: inf")


class ResumeSafeReportCallback(TrainerCallback):
    """Resume-safe training report callback.

    On train_begin: loads existing JSONL logs from output_dir (if resuming).
    During training: collects new metrics.
    On train_end: merges old+new, deduplicates by step, saves curves + report.

    This ensures training curves are complete across resume boundaries with
    no overlapping data points.
    """

    def __init__(self):
        self.train_log: list[dict] = []
        self.eval_log: list[dict] = []
        self.prior_train_log: list[dict] = []
        self.prior_eval_log: list[dict] = []
        self.start_time = None
        self.best_eval_loss = float("inf")
        self.best_eval_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

        # Load existing logs for resume-safe curves
        train_log_path = os.path.join(args.output_dir, "train_log.jsonl")
        eval_log_path = os.path.join(args.output_dir, "eval_log.jsonl")

        if os.path.exists(train_log_path):
            self.prior_train_log = self._load_jsonl(train_log_path)
            logger.info("Loaded %d prior train log entries from %s",
                        len(self.prior_train_log), train_log_path)

        if os.path.exists(eval_log_path):
            self.prior_eval_log = self._load_jsonl(eval_log_path)
            logger.info("Loaded %d prior eval log entries from %s",
                        len(self.prior_eval_log), eval_log_path)

        # Track best from prior logs
        for entry in self.prior_eval_log:
            if "eval_loss" in entry and entry["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = entry["eval_loss"]
                self.best_eval_step = entry["step"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        entry = {"step": step}
        if "loss" in logs:
            entry["loss"] = logs["loss"]
        if "learning_rate" in logs:
            entry["lr"] = logs["learning_rate"]
        if "total_flos" in logs:
            entry["total_flos"] = logs["total_flos"]
        if entry.keys() != {"step"}:
            self.train_log.append(entry)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = state.global_step
        entry = {"step": step}
        if "eval_loss" in metrics:
            entry["eval_loss"] = metrics["eval_loss"]
            if metrics["eval_loss"] < self.best_eval_loss:
                self.best_eval_loss = metrics["eval_loss"]
                self.best_eval_step = step
        if "eval_perplexity" in metrics:
            entry["eval_perplexity"] = metrics["eval_perplexity"]
        if "eval_runtime" in metrics:
            entry["eval_runtime_s"] = metrics["eval_runtime"]
        self.eval_log.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        output_dir = args.output_dir
        total_time = time.time() - self.start_time if self.start_time else 0
        final_step = state.global_step

        # Merge old + new, deduplicate by step (new takes precedence)
        merged_train = self._merge_logs(self.prior_train_log, self.train_log)
        merged_eval = self._merge_logs(self.prior_eval_log, self.eval_log)

        # Save merged logs
        self._save_jsonl(os.path.join(output_dir, "train_log.jsonl"), merged_train)
        self._save_jsonl(os.path.join(output_dir, "eval_log.jsonl"), merged_eval)

        # Save curves and report using merged data
        self._save_training_curves(output_dir, merged_train, merged_eval)
        self._save_report(output_dir, final_step, total_time, args, merged_train, merged_eval)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> list[dict]:
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _save_jsonl(path: str, records: list[dict]):
        # Sort by step before saving
        records.sort(key=lambda r: r.get("step", 0))
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @staticmethod
    def _merge_logs(prior: list[dict], new: list[dict]) -> list[dict]:
        """Merge prior and new logs, deduplicating by step.

        New entries take precedence over prior entries at the same step.
        """
        if not prior:
            return list(new)
        if not new:
            return list(prior)

        step_map: dict[int, dict] = {}
        for entry in prior:
            step = entry.get("step", 0)
            step_map[step] = entry
        for entry in new:
            step = entry.get("step", 0)
            step_map[step] = entry  # new overwrites prior at same step

        return sorted(step_map.values(), key=lambda r: r.get("step", 0))

    def _save_training_curves(self, output_dir, train_log, eval_log):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping training_curves.png")
            return

        if not train_log and not eval_log:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("DeepSleep CPT Training Curves", fontsize=14, fontweight="bold")

        # (0,0) Train Loss
        ax = axes[0, 0]
        loss_entries = [e for e in train_log if "loss" in e]
        if loss_entries:
            steps = [e["step"] for e in loss_entries]
            losses = [e["loss"] for e in loss_entries]
            ax.plot(steps, losses, alpha=0.6, linewidth=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel("Train Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)

        # (0,1) Eval Loss
        ax = axes[0, 1]
        eval_entries = [e for e in eval_log if "eval_loss" in e]
        if eval_entries:
            steps = [e["step"] for e in eval_entries]
            losses = [e["eval_loss"] for e in eval_entries]
            ax.plot(steps, losses, "o-", color="orange", linewidth=1.5, markersize=4)
            ax.set_xlabel("Step")
            ax.set_ylabel("Eval Loss")
            ax.set_title("Eval Loss")
            ax.grid(True, alpha=0.3)

        # (1,0) Learning Rate
        ax = axes[1, 0]
        lr_entries = [e for e in train_log if "lr" in e]
        if lr_entries:
            steps = [e["step"] for e in lr_entries]
            lrs = [e["lr"] for e in lr_entries]
            ax.plot(steps, lrs, color="green", linewidth=1.2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)

        # (1,1) Eval Perplexity
        ax = axes[1, 1]
        ppl_entries = [e for e in eval_log if "eval_perplexity" in e]
        if ppl_entries:
            steps = [e["step"] for e in ppl_entries]
            ppls = [e["eval_perplexity"] for e in ppl_entries]
            ax.plot(steps, ppls, "s-", color="red", linewidth=1.5, markersize=4)
            ax.set_xlabel("Step")
            ax.set_ylabel("Perplexity")
            ax.set_title("Eval Perplexity")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved to %s", path)

    def _save_report(self, output_dir, final_step, total_time, args, train_log, eval_log):
        final_train_loss = None
        for entry in reversed(train_log):
            if "loss" in entry:
                final_train_loss = entry["loss"]
                break

        final_eval_loss = None
        final_eval_ppl = None
        if eval_log:
            last = eval_log[-1]
            final_eval_loss = last.get("eval_loss")
            final_eval_ppl = last.get("eval_perplexity")

        report = {
            "stage": "continued_pretraining",
            "final_step": final_step,
            "total_time_hours": round(total_time / 3600, 2),
            "best_eval_loss": round(self.best_eval_loss, 6) if self.best_eval_loss < float("inf") else None,
            "best_eval_step": self.best_eval_step,
            "final_train_loss": round(final_train_loss, 6) if final_train_loss else None,
            "final_eval_loss": round(final_eval_loss, 6) if final_eval_loss else None,
            "final_eval_perplexity": round(final_eval_ppl, 2) if final_eval_ppl else None,
            "config": {
                "output_dir": args.output_dir,
                "pretrained_model": getattr(args, "pretrained_model_path", "N/A"),
                "max_steps": args.max_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "lr_scheduler_type": args.lr_scheduler_type,
                "bf16": args.bf16,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
                "save_total_limit": args.save_total_limit,
            },
        }

        path = os.path.join(output_dir, "report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Training report saved to %s", path)


# =============================================================================
# Data Collator
# =============================================================================


def pretrain_data_collator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    batch = {"input_ids": input_ids, "labels": labels}
    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    return batch


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSleep CPT (Medical QA)")

    # Model
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=1, type=int, choices=[0, 1])
    parser.add_argument("--num_experts", default=8, type=int)
    parser.add_argument("--num_shared_experts", default=0, type=int)
    parser.add_argument("--num_experts_per_tok", default=2, type=int)
    parser.add_argument("--vocab_size", default=7200, type=int)
    parser.add_argument("--max_seq_len", default=2048, type=int)

    # Pretrained model path
    parser.add_argument("--pretrained_model_path", type=str,
                        default="out/pretrain/final",
                        help="Path to pretrained model checkpoint")

    # Data
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default="/public/huggingface-datasets/Malikeh1375/medical-question-answering-datasets")
    parser.add_argument("--max_eval_samples", type=int, default=10_000)
    parser.add_argument("--min_text_length", type=int, default=30)
    parser.add_argument("--pack_sequences", default=1, type=int, choices=[0, 1])

    # Training
    parser.add_argument("--output_dir", type=str, default="out/cpt")
    parser.add_argument("--max_steps", type=int, default=2_000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=96)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=96)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")

    # Eval & Save
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="DeepSleep-CPT")
    parser.add_argument("--run_name", type=str, default=None)

    # Generation
    parser.add_argument("--generate_prompts", type=str, default=None)

    # Config
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        from configs.config_utils import load_yaml_config
        args = load_yaml_config(args)
        logger.info("Loaded config from %s", args.config)

    return args


def main():
    args = parse_args()
    pack = bool(args.pack_sequences)

    # --- Performance optimizations ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # --- Tokenizer ---
    logger.info("Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    logger.info("  Vocab size: %d", tokenizer.vocab_size)

    # --- Model Config (from pretrained checkpoint) ---
    pretrained_path = args.pretrained_model_path
    config_path = os.path.join(pretrained_path, "config.json")

    if os.path.exists(config_path):
        logger.info("Loading model config from %s", config_path)
        config = DeepSleepConfig.from_pretrained(pretrained_path)
    else:
        logger.info("No config.json found, using CLI args for model config")
        config = DeepSleepConfig(
            d_model=args.hidden_size,
            n_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            num_experts=args.num_experts,
            num_shared_experts=args.num_shared_experts,
            top_k=args.num_experts_per_tok,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.max_seq_len,
        )

    # --- Model (load pretrained weights) ---
    model = DeepSleepForCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info("Model: %.2fM total params, %.2fM trainable", total_params, trainable_params)

    # Load pretrained weights
    safetensors_path = os.path.join(pretrained_path, "model.safetensors")
    pth_path = os.path.join(pretrained_path, "model.pth")

    state_dict = None
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights from %s (safetensors)", safetensors_path)
    elif os.path.exists(pth_path):
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights from %s (pth)", pth_path)
    else:
        logger.warning("No pretrained weights found at %s, training from scratch!", pretrained_path)

    if state_dict is not None:
        del state_dict

    # --- Datasets ---
    logger.info("Creating medical QA datasets (pack=%s)", pack)
    train_dataset = MedicalQADataset(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        data_dir=args.data_dir,
        seed=42,
        pack_sequences=pack,
        min_length=args.min_text_length,
    )
    eval_dataset = MedicalQADataset(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        data_dir=args.data_dir,
        seed=123,
        num_samples=args.max_eval_samples,
        pack_sequences=pack,
        min_length=args.min_text_length,
    )
    logger.info("  Train: %s, Eval: %d samples", "streaming" if train_dataset.num_samples is None else train_dataset.num_samples, args.max_eval_samples)

    # --- Callbacks ---
    gen_prompts = (
        args.generate_prompts.split(",") if args.generate_prompts
        else MEDICAL_PROMPTS
    )
    report_callback = ResumeSafeReportCallback()
    callbacks = [
        SampleGenerationCallback(
            tokenizer=tokenizer,
            prompts=gen_prompts,
            max_new_tokens=100,
        ),
        PerplexityCallback(),
        report_callback,
    ]

    # --- Training Arguments ---
    run_name = args.run_name or f"cpt-medical-d{config.d_model}-moe"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Training
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        # Optimizer
        optim=args.optim,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        # Precision
        bf16=args.bf16,
        bf16_full_eval=True,
        # Memory
        gradient_checkpointing=args.gradient_checkpointing,
        # Speed
        torch_compile=True,
        torch_compile_backend="inductor",
        # Eval & Save
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Data loading
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        # Logging
        report_to=args.report_to.split(",") if args.report_to else [],
        run_name=run_name,
        # Misc
        remove_unused_columns=False,
        include_num_input_tokens_seen=True,
    )
    logger.info("TrainingArguments:")
    for key, val in sorted(training_args.to_dict().items()):
        if val is not None and val is not False:
            logger.info("  %s: %s", key, val)

    # --- Tokens budget ---
    eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_tokens = eff_batch * args.max_seq_len * args.max_steps
    logger.info("Tokens budget: %d (effective_batch=%d, seq_len=%d, steps=%d)",
                total_tokens, eff_batch, args.max_seq_len, args.max_steps)
    logger.info("  = %.1fM tokens | %.1f tokens/param (model %dM)",
                total_tokens / 1e6,
                total_tokens / (total_params * 1e6),
                int(total_params))

    # --- Trainer ---
    trainer = MoETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=pretrain_data_collator,
        callbacks=callbacks,
    )

    # --- Train ---
    logger.info("Starting continued pretraining on medical QA data...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save Final Model (HF format) ---
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Final model saved to %s (HF format)", final_dir)

    # --- Save .pth for SFT/DPO compatibility ---
    pth_path = os.path.join(final_dir, "model.pth")
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), pth_path)
    logger.info("Final model saved to %s (.pth)", pth_path)

    # --- Final Eval ---
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", metrics)

    # --- Cleanup intermediate checkpoints, keep only final ---
    import shutil
    for entry in os.listdir(args.output_dir):
        entry_path = os.path.join(args.output_dir, entry)
        if entry.startswith("checkpoint-") and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
            logger.info("Removed intermediate checkpoint: %s", entry_path)
    logger.info("Cleanup done. Only final model retained at %s/final/", args.output_dir)


if __name__ == "__main__":
    main()
