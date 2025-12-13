import logging
import os
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup, set_seed
)
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from inference.model import ExpTransformer
from inference.core.arguments import ModelArgs
from torch.amp import GradScaler, autocast

dataset_name: str = "wikimedia/wikipedia"
dataset_config: str = "20231101.en"  # Using a subset for demonstration
tokenizer_name: str = "deepseek-ai/DeepSeek-V3.2"  # Compatible tokenizer for base LLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = ModelArgs()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    args.vocab_size = tokenizer.vocab_size

    tokenizer.vocab_size

    logger.info(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, name=dataset_config, split=f"train[:{int(10_000)}]")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

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

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=2,
        shuffle=True,
        num_workers=2,  # Reduced for stability
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["test"],
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Initialize model
    logger.info("Initializing model...")
    model = ExpTransformer(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

        # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.001,
        betas=(0.9, 0.95)  # Common LLM optimizer settings
    )
    total_steps = 5000
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2000,
        num_training_steps=total_steps
    )

    scaler = GradScaler(enabled=True)

    output_dir = "./llm-training-output"
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    global_step = 0
    best_eval_loss = float('inf')
    train_losses = []
    eval_losses = []

    logger.info("Starting training...")

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training epochs
    for epoch in range(int(1_000_000)):
        model.train()
        epoch_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=100)

        for batch_idx, batch in enumerate(progress_bar):

            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=True, device_type="cpu"):
                outputs = model(batch["input_ids"])
                target = batch["labels"][:, -1]
                loss = loss_fct(outputs, target)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            batch_count += 1

            progress_bar.set_postfix({
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })

            current_loss = loss.item()
            train_losses.append((global_step, current_loss))

        if epoch % 1000 == 0 or epoch == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        tokenizer.save_pretrained(output_dir)
            
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
