import logging
import os
from collections import defaultdict
from datetime import datetime
from itertools import islice
from typing import Type

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.models.base_model import BaseModel


class Trainer:
    def __init__(self,
                 model_type: Type[BaseModel],
                 epochs: int,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 early_stopping=True,
                 early_stop_patience=5,
                 checkpoint_dir='../data/checkpoints',
                 device='cpu',):
        self.model_type = model_type
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stop_patience = early_stop_patience if early_stopping else epochs
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # Set up logging with automatic filename
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging with automatic filename generation."""
        # Create log directory if it doesn't exist
        log_dir = '../logs'
        os.makedirs(log_dir, exist_ok=True)

        # Generate filename with current UTC time
        model_name = self.model_type.__name__
        current_time = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
        log_filename = f"{log_dir}/{model_name}-{current_time}.log"

        # Configure logger
        self.logger = logging.getLogger(f"{model_name}_trainer")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Log initialization message
        self.logger.info(f"Training session started at {datetime.utcnow().isoformat()}Z")
        self.logger.info(f"Logging to: {os.path.abspath(log_filename)}")
        self.logger.info(f"Model type: {model_name}")
        self.logger.info(f"Device: {self.device}")

    def train_model(self, dry_run=False):
        train_losses_arr = []
        val_losses_arr = []

        model = self.model_type().to(self.device)
        optimizer = torch.optim.Adam(model.parameters())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_val_loss = np.inf
        epochs_no_improve = 0

        num_epochs = self.epochs if not dry_run else 1

        self.logger.info("Starting training for %d epochs", num_epochs)
        self.logger.info("Early stopping patience: %d epochs", self.early_stop_patience)

        for epoch in range(num_epochs):
            train_losses = self.train_epoch(model, optimizer, dry_run)
            train_losses_arr.append(train_losses)

            val_losses = self.validate_epoch(model, dry_run)
            val_losses_arr.append(val_losses)

            # Log epoch summary
            self.logger.info("Epoch %d/%d:", epoch + 1, self.epochs)
            self.logger.info("Train Loss: %.4f", train_losses['total'])
            self.logger.info("Val Loss: %.4f", val_losses['total'])

            current_loss = val_losses['total']
            if current_loss < best_val_loss:
                self.logger.info("Validation loss improved (%.4f → %.4f)",
                                 best_val_loss, current_loss)
                self.logger.info("Validation metrics: %s", str(val_losses))

                best_val_loss = current_loss
                epochs_no_improve = 0

                # Save best model
                model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': {'train': train_losses, 'val': val_losses}
                }, model_path)
                self.logger.info("Saved best model to %s", model_path)
            else:
                epochs_no_improve += 1
                self.logger.info("No loss improvement for %d/%d epochs",
                                 epochs_no_improve, self.early_stop_patience)

                if epochs_no_improve >= self.early_stop_patience:
                    self.logger.info("Early stopping triggered at epoch %d!", epoch + 1)
                    break

        # Save final metrics
        metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.pth')
        torch.save({
            'train_losses_arr': train_losses_arr,
            'val_losses_arr': val_losses_arr,
        }, metrics_path)
        self.logger.info("Saved training metrics to %s", metrics_path)

        self.logger.info("Training completed. Best validation loss: %.4f", best_val_loss)

        return model

    def train_epoch(self, model, optimizer, dry_run=False):
        model.train()
        epoch_metrics = defaultdict(float)

        if dry_run:
            pbar = tqdm(islice(self.train_loader, 10))
        else:
            pbar = tqdm(self.train_loader)

        for batch in pbar:
            optimizer.zero_grad()
            batch_loss, batch_metrics = model.compute_batch_loss(
                batch, self.device
            )

            batch_loss.backward()
            optimizer.step()

            for key in batch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            pbar.set_description(f"Train Loss: {batch_loss.item():.4f}")

        num_batches = len(self.train_loader) if not dry_run else 10
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def validate_epoch(self, model, dry_run=False):
        model.eval()
        epoch_metrics = defaultdict(float)

        if dry_run:
            pbar = tqdm(islice(self.val_loader, 10))
        else:
            pbar = tqdm(self.val_loader)

        with torch.no_grad():
            for batch in pbar:
                batch_loss, batch_metrics = model.compute_batch_loss(
                    batch, self.device
                )

                for key in batch_metrics:
                    epoch_metrics[key] += batch_metrics[key]

                pbar.set_description(f"Val Loss: {batch_loss.item():.4f}")

        num_batches = len(self.val_loader) if not dry_run else 10
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics
