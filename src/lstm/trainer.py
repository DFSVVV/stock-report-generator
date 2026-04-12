"""Training pipeline for LSTM stock prediction model."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data_processor import StockDataProcessor
from .model import StockLSTM


@dataclass
class TrainingConfig:
    """Configuration for training."""

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    sequence_length: int = 60
    forecast_horizon: int = 1


@dataclass
class TrainingHistory:
    """Training history record."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    epochs_completed: int = 0


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor):
        """Initialize dataset.

        Args:
            sequences: Input sequences (num_samples, seq_len, features)
            labels: Target labels (num_samples,)
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


class StockTrainer:
    """Trainer for StockLSTM model."""

    def __init__(
        self,
        model: StockLSTM,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: StockLSTM model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            config: Training configuration
            device: Device to train on
        """
        if config is None:
            config = TrainingConfig()

        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        self.criterion = nn.MSELoss()

        self.history = TrainingHistory()
        self.best_val_loss = float("inf")

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Tuple of (sequences, labels)

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        sequences, labels = batch
        sequences = sequences.to(self.device)
        labels = labels.to(self.device).unsqueeze(1)

        self.optimizer.zero_grad()
        outputs = self.model(sequences)
        loss = self.criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics
        """
        if not self.val_loader:
            return {"loss": 0.0}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    def train(self) -> TrainingHistory:
        """Run full training loop with early stopping.

        Returns:
            Training history
        """
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_train_loss = 0.0
            num_train_batches = 0

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_train_loss += metrics["loss"]
                num_train_batches += 1

            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            val_metrics = self.validate()
            avg_val_loss = val_metrics["loss"]

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history.train_losses.append(avg_train_loss)
            self.history.val_losses.append(avg_val_loss)
            self.history.learning_rates.append(current_lr)
            self.history.epochs_completed = epoch + 1

            self.scheduler.step(avg_val_loss)

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": {
                "train_losses": self.history.train_losses,
                "val_losses": self.history.val_losses,
                "learning_rates": self.history.learning_rates,
                "epochs_completed": self.history.epochs_completed,
            },
            "config": {
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "dropout": self.config.dropout,
                "bidirectional": self.config.bidirectional,
                "sequence_length": self.config.sequence_length,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        device: Optional[str] = None,
    ) -> tuple["StockTrainer", TrainingHistory]:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            device: Device to load on

        Returns:
            Tuple of (StockTrainer, TrainingHistory)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = TrainingConfig(
            hidden_size=checkpoint["config"]["hidden_size"],
            num_layers=checkpoint["config"]["num_layers"],
            dropout=checkpoint["config"]["dropout"],
            bidirectional=checkpoint["config"]["bidirectional"],
            sequence_length=checkpoint["config"]["sequence_length"],
        )

        model = StockLSTM(
            input_size=10,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )

        trainer = cls(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
        )

        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.best_val_loss = checkpoint["best_val_loss"]

        history_data = checkpoint["history"]
        trainer.history = TrainingHistory(
            train_losses=history_data["train_losses"],
            val_losses=history_data["val_losses"],
            learning_rates=history_data["learning_rates"],
            epochs_completed=history_data["epochs_completed"],
        )

        return trainer, trainer.history
