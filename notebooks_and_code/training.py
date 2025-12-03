import copy
import logging
import os
from typing import Dict, List
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, 
    get_linear_schedule_with_warmup, set_seed
)
from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import math
from dataclasses import dataclass
import torch.nn.functional as F
from inference.model import Transformer, ModelArgs
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configuration
dataset_name: str = "HuggingFaceFW/fineweb"
dataset_config: str = "sample-10BT"  # Using a subset for demonstration
tokenizer_name: str = "EleutherAI/gpt-neox-20b"  # Compatible tokenizer for base LLM

# Training hyperparameters
TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "max_steps": 50000,
    "logging_steps": 10,
    "eval_steps": 500,
    "save_steps": 1000,
    "fp16": True,  # Use mixed precision training
    "bf16": False,  # Use if you have Ampere+ GPU
    "optim": "adamw_torch",
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "seed": 42,
}

def prepare_dataset(tokenizer, args, max_samples=10000):
    """Prepare dataset for training"""
    logger.info(f"Loading dataset {dataset_name}")
    
    # Load dataset - using a manageable subset for training
    dataset = load_dataset(dataset_name, name=dataset_config, split=f"train[:{max_samples}]")
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.1, seed=TRAINING_CONFIG["seed"])
    
    # Tokenization function
    def tokenize_function(examples):
        # Concatenate text if needed
        texts = examples["text"]
        
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=args.original_seq_len,
            return_tensors="pt"
        )
        
        # Create labels for language modeling (shifted input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Tokenize dataset
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=args.max_batch_size,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )
    
    # Set format for PyTorch
    tokenized_datasets.set_format("torch")
    
    return tokenized_datasets

def compute_metrics(eval_pred, tokenizer):
    """Compute perplexity and other metrics"""
    logits, labels = eval_pred
    # Shift for language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    perplexity = torch.exp(loss)
    
    return {
        "perplexity": perplexity.item(),
        "loss": loss.item()
    }

class CustomTrainer(Trainer):
    """Custom trainer with additional logging and metrics"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute custom loss.
        inputs: dict containing 'input_ids', 'attention_mask', 'labels'
        """
        # Forward pass
        outputs = model(input_ids=inputs["input_ids"])
        
        # Get logits from model output
        logits = outputs
        
        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Perform prediction step.
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            
            if prediction_loss_only:
                return (loss, None, None)
            
            # Get logits
            logits = outputs
            
            labels = inputs["labels"]
            # Shift labels for evaluation
            labels = labels[..., 1:].contiguous()
            
        return (loss, logits, labels)

def manual_training_loop():
    """Manual training loop for more control (Recommended for custom models)"""
    # Set seed
    set_seed(TRAINING_CONFIG["seed"])
    
    # Initialize model args
    args = ModelArgs()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    args.vocab_size = tokenizer.vocab_size
    
    # Prepare dataset - using a smaller subset for faster iteration
    tokenized_datasets = prepare_dataset(tokenizer, args, max_samples=5000)
    
    # Create data loaders
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        shuffle=True,
        num_workers=2,  # Reduced for stability
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model = Transformer(args)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(0.9, 0.95)  # Common LLM optimizer settings
    )
    
    # Scheduler
    total_steps = TRAINING_CONFIG["max_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
        num_training_steps=total_steps
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(enabled=TRAINING_CONFIG["fp16"])
    
    # Create output directory
    output_dir = "./llm-training-output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    train_losses = []
    eval_losses = []
    
    logger.info("Starting training...")
    
    # Training epochs
    for epoch in range(TRAINING_CONFIG["max_steps"] // len(train_dataloader) + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=TRAINING_CONFIG["fp16"]):
                outputs = model(input_ids=batch["input_ids"])
                
                # Compute loss
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Gradient accumulation
                loss = loss / TRAINING_CONFIG["gradient_accumulation_steps"]
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if ((batch_idx + 1) % TRAINING_CONFIG["gradient_accumulation_steps"] == 0) or \
               (batch_idx + 1 == len(train_dataloader)):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    TRAINING_CONFIG["max_grad_norm"]
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
            
            epoch_loss += loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"]
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"],
                "lr": scheduler.get_last_lr()[0]
            })
            
            # Logging
            if global_step % TRAINING_CONFIG["logging_steps"] == 0:
                current_loss = loss.item() * TRAINING_CONFIG["gradient_accumulation_steps"]
                logger.info(f"Step {global_step}: Loss = {current_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
                train_losses.append((global_step, current_loss))
            
            # Evaluation
            if global_step % TRAINING_CONFIG["eval_steps"] == 0 and global_step > 0:
                model.eval()
                eval_loss = evaluate_model(model, eval_dataloader, device, tokenizer)
                eval_losses.append((global_step, eval_loss))
                logger.info(f"Step {global_step}: Eval Loss = {eval_loss:.4f}, Perplexity = {math.exp(eval_loss):.2f}")
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    checkpoint_path = os.path.join(output_dir, f"best_model_step_{global_step}.pt")
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': eval_loss,
                        'config': args,
                    }, checkpoint_path)
                    logger.info(f"Saved best model to {checkpoint_path}")
                
                model.train()
            
            # Save checkpoint
            if global_step % TRAINING_CONFIG["save_steps"] == 0 and global_step > 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'eval_losses': eval_losses,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            if global_step >= TRAINING_CONFIG["max_steps"]:
                break
        
        # Log epoch statistics
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        if global_step >= TRAINING_CONFIG["max_steps"]:
            break
    
    # Save final model
    final_checkpoint_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'config': args,
    }, final_checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training completed! Final model saved to {output_dir}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        if train_losses:
            steps, losses = zip(*train_losses)
            plt.figure(figsize=(10, 5))
            plt.plot(steps, losses, label='Training Loss')
            if eval_losses:
                eval_steps, eval_losses_vals = zip(*eval_losses)
                plt.plot(eval_steps, eval_losses_vals, label='Validation Loss', marker='o')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'training_curve.png'))
            plt.close()
    except ImportError:
        pass
    
    return model, tokenizer

def evaluate_model(model, eval_dataloader, device, tokenizer):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(input_ids=batch["input_ids"])
            
            # Compute loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
    
    model.train()
    return total_loss / total_samples

if __name__ == "__main__":
    # Use manual training loop for custom model
    model, tokenizer = manual_training_loop()
    
    logger.info("Training completed successfully!")