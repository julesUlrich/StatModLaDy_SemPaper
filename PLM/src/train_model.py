# src/train_model.py

import os
import math
import json
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset, Dataset
from phoneme_tokenizer import PhonemeSplitter
from tokenizers import Tokenizer
import torch
import unicodedata
import numpy as np

def preprocess_line(text: str) -> str:
    """Remove leading ID and special tokens, normalize NFC."""
    text = text.strip()
    if "\t" in text:
        parts = text.split("\t", 1)
        if len(parts) == 2 and parts[0]:
            text = parts[1]
    # Remove special tokens
    for token in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]:
        text = text.replace(token, "")
    return unicodedata.normalize("NFC", text).strip()

class CustomTokenizerWrapper:
    """Wrapper to make the custom tokenizer compatible with transformers."""

    def __init__(self, tokenizer_path, splitter):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.splitter = splitter

        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.bos_token_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")

        # Set special tokens
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"

    def __call__(self, text, truncation=True, max_length=1024, **kwargs):
        # Make sure we preprocess the input line before splitting
        text = preprocess_line(text)
        phonemes = self.splitter.split(text)
        joined = " ".join(phonemes)
        encoded = self.tokenizer.encode(joined)
        token_ids = encoded.ids[:max_length]
        return {
            'input_ids': token_ids,
            'attention_mask': [1] * len(token_ids)
        }

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def convert_tokens_to_ids(self, token):
        return self.tokenizer.token_to_id(token)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        """Add padding method required by DataCollatorForLanguageModeling"""
        import torch

        if isinstance(encoded_inputs, dict):
            encoded_inputs = [encoded_inputs]

        batch_input_ids = []
        batch_attention_mask = []

        # Find max length in batch
        if max_length is None:
            max_length = max(len(example['input_ids']) for example in encoded_inputs)

        for example in encoded_inputs:
            input_ids = example['input_ids']
            attention_mask = example.get('attention_mask', [1] * len(input_ids))

            # Pad sequences
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        result = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask
        }

        if return_tensors == "pt":
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])

        return result

    def save_pretrained(self, save_directory):
        """Save the tokenizer to a directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_path = os.path.join(save_directory, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)

        # Also save a simple config file with special tokens info
        config = {
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id
        }

        import json
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def encode(example, tokenizer):
    enc = tokenizer(example["text"], truncation=True, max_length=1024)
    enc["labels"] = enc["input_ids"].copy()
    return enc


class CustomDataCollator:
    """Custom data collator for language modeling with our custom tokenizer."""

    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features):
        import torch

        # Extract input_ids from features
        batch_input_ids = []
        batch_attention_mask = []

        for feature in features:
            batch_input_ids.append(feature['input_ids'])
            batch_attention_mask.append(feature.get('attention_mask', [1] * len(feature['input_ids'])))

        # Find max length in batch
        max_length = max(len(input_ids) for input_ids in batch_input_ids)

        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []

        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * padding_length)
                padded_attention_mask.append(attention_mask + [0] * padding_length)
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)

        # Convert to tensors
        batch = {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long)
        }

        # For language modeling, labels are the same as input_ids
        if not self.mlm:
            batch['labels'] = batch['input_ids'].clone()

        return batch


class EvaluationMetricsCallback(TrainerCallback):
    """Custom callback to compute additional evaluation metrics."""

    def __init__(self, eval_dataset, tokenizer, output_dir, target_perplexity=None):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.target_perplexity = target_perplexity
        self.metrics_history = []
        self.best_perplexity = float('inf')
        self.best_step = 0

    def on_evaluate(self, args, state, control, model, eval_dataloader=None, **kwargs):
        """Compute additional metrics after evaluation."""
        if eval_dataloader is None:
            return

        print("\nComputing additional evaluation metrics...")

        # Identify special token IDs to exclude
        special_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id
        }

        def exclude_special(tensor):
            return ~torch.isin(tensor, torch.tensor(list(special_ids), device=tensor.device))

        total_loss = 0.0
        total_tokens = 0
        all_predictions = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                # Compute loss
                total_loss += loss.item() * batch['input_ids'].numel()
                total_tokens += batch['input_ids'].numel()

                # Accuracy
                predictions = torch.argmax(logits[:, :-1], dim=-1)
                labels = batch['labels'][:, 1:]
                mask = exclude_special(labels)

                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        # Top-k accuracy
        top5_correct = 0
        top10_correct = 0
        total_eval_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits[:, :-1]
                labels = batch['labels'][:, 1:]

                mask = exclude_special(labels)
                total_eval_tokens += mask.sum().item()

                top5_preds = torch.topk(logits, 5, dim=-1).indices
                top10_preds = torch.topk(logits, 10, dim=-1).indices

                labels_exp = labels.unsqueeze(-1).expand_as(top5_preds)
                top5_hits = (top5_preds == labels_exp).any(dim=-1)
                top5_correct += top5_hits[mask].sum().item()

                labels_exp = labels.unsqueeze(-1).expand_as(top10_preds)
                top10_hits = (top10_preds == labels_exp).any(dim=-1)
                top10_correct += top10_hits[mask].sum().item()

        top5_accuracy = top5_correct / total_eval_tokens
        top10_accuracy = top10_correct / total_eval_tokens

        metrics = {
            'step': state.global_step,
            'epoch': state.epoch,
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'learning_rate': state.learning_rate if hasattr(state, 'learning_rate') else args.learning_rate
        }

        self.metrics_history.append(metrics)

        # Print metrics
        print(f"Step {state.global_step} Evaluation Metrics:")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"  Top-10 Accuracy: {top10_accuracy:.4f}")
        print(f"Loss: {loss.item()}")
        print(f"Batch size: {batch['input_ids'].shape}")
        print(f"Labels max ID: {batch['labels'].max().item()}, Vocab size: {model.config.vocab_size}")

        if perplexity < self.best_perplexity:
            self.best_perplexity = perplexity
            self.best_step = state.global_step
            best_model_dir = os.path.join(self.output_dir, "best_model")
            model.save_pretrained(best_model_dir)
            print(f"  New best model saved! (Perplexity: {perplexity:.4f})")

        metrics_file = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.target_perplexity and perplexity <= self.target_perplexity:
            print(f"Target perplexity {self.target_perplexity} reached! Current: {perplexity:.4f}")
            control.should_training_stop = True

        model.train()


def create_eval_dataset(eval_path, tokenizer):
    """Create evaluation dataset."""
    if eval_path and os.path.exists(eval_path):
        eval_dataset = load_dataset("text", data_files={"eval": eval_path})["eval"]
        eval_dataset = eval_dataset.map(lambda x: encode(x, tokenizer), batched=False)
        return eval_dataset
    return None


def train_model(
        data_path: str,
        tokenizer_path: str,
        output_dir: str,
        eval_path: str = None,
        use_csv_units: bool = False,
        csv_path: str = None,
        csv_column: str = None,
        batch_size: int = 2,
        eval_batch_size: int = 4,
        epochs: int = 3,
        max_steps: int = None,
        learning_rate: float = 5e-5,
        layers: int = 6,
        heads: int = 8,
        eval_steps: int = 500,
        save_steps: int = 500,
        target_perplexity: float = None,
        patience: int = None
):
    # Load phoneme tokenizer
    splitter = PhonemeSplitter(
        use_csv_units=use_csv_units,
        csv_path=csv_path,
        csv_column=csv_column
    )

    tokenizer = CustomTokenizerWrapper(tokenizer_path, splitter)

    # Get actual vocabulary size from tokenizer
    actual_vocab_size = len(tokenizer)

    config = GPT2Config(
        vocab_size=actual_vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=512,
        n_layer=layers,
        n_head=heads,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        resid_pdrop=0.1
    )

    model = GPT2LMHeadModel(config)

    # Load datasets
    dataset = load_dataset("text", data_files={"train": data_path})["train"]
    dataset = dataset.map(lambda x: encode(x, tokenizer), batched=False)

    eval_dataset = create_eval_dataset(eval_path, tokenizer)

    print("Eval dataset size:",len(eval_dataset))

    # Determine evaluation strategy
    eval_strategy = "epoch" if eval_dataset else "no"

    max_steps = max_steps if max_steps is not None else -1

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        eval_strategy=eval_strategy, #TODO: fix these
        eval_steps=1,
        save_strategy=eval_strategy, #TODO: fix these
        save_steps=1,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to=None,  # Disable wandb/tensorboard
    )

    # Early stopping callback
    callbacks = []
    if eval_dataset:
        metrics_callback = EvaluationMetricsCallback(
            eval_dataset, tokenizer, output_dir, target_perplexity
        )
        callbacks.append(metrics_callback)

        if patience:
            from transformers import EarlyStoppingCallback
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=0.001
            )
            callbacks.append(early_stopping)

    import warnings
    warnings.filterwarnings("ignore", message=".*tokenizer.*is deprecated.*", category=FutureWarning)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(tokenizer, mlm=False),
        callbacks=callbacks
    )

    # Print training info
    print(f"Training Configuration:")
    print(f"  Model: GPT-2 ({layers} layers, {heads} heads)")
    print(f"  Vocabulary size: {actual_vocab_size}")
    print(f"  Training samples: {len(dataset)}")
    if eval_dataset:
        print(f"  Evaluation samples: {len(eval_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max epochs: {epochs}")

    if max_steps:
        print(f"  Max steps: {max_steps}")
    if target_perplexity:
        print(f"  Target perplexity: {target_perplexity}")
    if patience:
        print(f"  Early stopping patience: {patience}")
    print()

    # Train the model
    trainer.train()

    # Save final model
    final_model_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_dir)

    # Save tokenizer
    os.makedirs(os.path.join(final_model_dir, "tokenizer"), exist_ok=True)
    tokenizer.tokenizer.save(os.path.join(final_model_dir, "tokenizer", "tokenizer.json"))

    # Print final results
    if eval_dataset and callbacks:
        metrics_callback = callbacks[0]
        print(f"\nTraining completed!")
        print(f"Best model at step {metrics_callback.best_step} with perplexity {metrics_callback.best_perplexity:.4f}")
        print(f"Metrics saved to: {os.path.join(output_dir, 'training_metrics.json')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a GPT-2 model with phoneme tokenization and enhanced evaluation")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training text file")
    parser.add_argument("--eval_path", type=str, help="Path to evaluation text file")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to trained tokenizer JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument("--phoneme_csv", type=str, help="CSV file with phoneme definitions")
    parser.add_argument("--column", type=str, help="Column in CSV to use as phoneme unit")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--target_perplexity", type=float, help="Stop training when this perplexity is reached")
    parser.add_argument("--patience", type=int,
                        help="Early stopping patience (number of evaluations without improvement)")

    args = parser.parse_args()

    train_model(
        data_path=args.train_path,
        eval_path=args.eval_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        use_csv_units=bool(args.phoneme_csv),
        csv_path=args.phoneme_csv,
        csv_column=args.column,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        layers=args.n_layers,
        heads=args.heads,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        target_perplexity=args.target_perplexity,
        patience=args.patience
    )