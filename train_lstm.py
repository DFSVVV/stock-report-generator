"""Train LSTM model for stock prediction.

Usage:
    python train_lstm.py --stock 000001 --epochs 50

This will:
1. Fetch stock data from Sina API
2. Create training sequences
3. Train the LSTM model
4. Save the model to models/stock_lstm.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data_source import SinaDataSource
from src.lstm.data_processor import StockDataProcessor
from src.lstm.model import StockLSTM
from src.lstm.trainer import StockTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train LSTM stock prediction model")
    parser.add_argument("--stock", default="000001", help="Stock code to train on")
    parser.add_argument("--start_date", default="20230101", help="Start date (YYYYMMDD)")
    parser.add_argument("--end_date", default="20260412", help="End date (YYYYMMDD)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length")
    parser.add_argument("--model_path", default="models/stock_lstm.pt", help="Path to save model")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    args = parser.parse_args()

    # Create models directory
    model_dir = Path(args.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for stock {args.stock}...")
    source = SinaDataSource()
    bundle = source.fetch(
        stock_code=args.stock,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Fetched {len(bundle.data)} days of data")

    # Prepare training data
    print("Preparing training sequences...")
    processor = StockDataProcessor(sequence_length=args.seq_len)
    sequences, labels = processor.create_sequences(bundle, forecast_horizon=1)
    print(f"Created {len(sequences)} training sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Label stats: mean={labels.mean():.2f}, std={labels.std():.2f}, min={labels.min():.2f}, max={labels.max():.2f}")

    # Create dataset
    from src.lstm.trainer import StockDataset
    dataset = StockDataset(
        torch.from_numpy(sequences),
        torch.from_numpy(labels),
    )

    # Split train/val
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size: {val_size}")

    # Create model
    print("Creating model...")
    model = StockLSTM(
        input_size=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
    )

    # Training config
    config = TrainingConfig(
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        learning_rate=1e-3,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=10,
        sequence_length=args.seq_len,
    )

    # Train
    print("Starting training...")
    trainer = StockTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

    history = trainer.train()

    # Save final model
    print(f"Saving model to {args.model_path}...")
    trainer.save_checkpoint(args.model_path)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Epochs completed: {history.epochs_completed}")
    print(f"\nTo use the model, run with --model_path {args.model_path}")


if __name__ == "__main__":
    main()
