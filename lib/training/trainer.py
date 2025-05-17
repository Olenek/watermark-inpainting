import logging
import os
from collections import defaultdict
from datetime import datetime
from itertools import islice
from typing import Type, Any, Optional

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
                 early_stop_patience=10,
                 checkpoint_dir='/app/checkpoints/',
                 log_dir='/app/logs/',
                 device='cpu', ):
        self.model_type = model_type
        self.epochs = epochs
        self.warmup_epochs = 10
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.early_stop_patience = early_stop_patience if early_stopping else epochs
        self.experiment_name = f"{self.model_type.__name__}-" \
                               f"{datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')}"
        self.checkpoint_dir = os.path.join(checkpoint_dir, self.experiment_name)
        self.device = device

        # Set up logging with automatic filename
        self._setup_logging(log_dir)

    def _setup_logging(self, log_dir: str):
        """Configure logging with automatic filename generation."""
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate filename with current UTC time
        log_filename = f"{log_dir}/{self.experiment_name}.log"

        # Configure logger
        self.logger = logging.getLogger(f"{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Log initialization message
        self.logger.info(f"Training session started at {datetime.utcnow().isoformat()}Z")
        self.logger.info(f"Logging to: {os.path.abspath(log_filename)}")
        self.logger.info(f"Model type: {self.model_type.__name__}")
        self.logger.info(f"Device: {self.device}")

    def train_model(self, model_hyperparams: Optional[dict[str, Any]]=None, dry_run=False):
        train_losses_arr = []
        val_losses_arr = []

        if model_hyperparams is None:
            model = self.model_type().to(self.device)
        else:
            model = self.model_type(**model_hyperparams).to(self.device)

        stage1_opt = torch.optim.Adam(model.stage1_parameters())
        final_opt = torch.optim.Adam(model.parameters())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        best_val_loss = np.inf
        epochs_no_improve = 0

        hyperparams_path = os.path.join(self.checkpoint_dir, 'hyperparams.pth')
        torch.save(model_hyperparams, hyperparams_path)

        warmup_epochs = self.warmup_epochs if not dry_run else 1
        num_epochs = self.epochs if not dry_run else 1

        self.logger.info("Starting warmup training for %d epochs", warmup_epochs)

        for epoch in range(warmup_epochs):
            train_losses = self.train_stage1_epoch(model, stage1_opt, dry_run)
            train_losses_arr.append(train_losses)

            val_losses = self.validate_stage1_epoch(model, dry_run)
            val_losses_arr.append(val_losses)

            # Log epoch summary
            self.logger.info("W.Epoch %d/%d:", epoch + 1, self.warmup_epochs)
            self.logger.info("Train Loss: %.4f", train_losses['L_total'])
            self.logger.info("Val Loss: %.4f", val_losses['L_total'])
            self.logger.info("Validation metrics: %s", str(val_losses))

        self.logger.info("Starting training for %d epochs", num_epochs)
        self.logger.info("Early stopping patience: %d epochs", self.early_stop_patience)

        for epoch in range(num_epochs):
            train_losses = self.train_epoch(model, final_opt, dry_run)
            train_losses_arr.append(train_losses)

            val_losses = self.validate_epoch(model, dry_run)
            val_losses_arr.append(val_losses)

            # Log epoch summary
            self.logger.info("Epoch %d/%d:", epoch + 1, self.epochs)
            self.logger.info("Train Loss: %.4f", train_losses['L_total'])
            self.logger.info("Val Loss: %.4f", val_losses['L_total'])

            current_loss = val_losses['L_total']
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
                    'optimizer_state_dict': final_opt.state_dict(),
                    'losses': {'train': train_losses, 'val': val_losses},
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

    def train_stage1_epoch(self, model: BaseModel, optimizer, dry_run=False):
        model.train()
        model.stage2.eval()
        epoch_metrics = defaultdict(float)

        if dry_run:
            pbar = tqdm(islice(self.train_loader, 10))
        else:
            pbar = tqdm(self.train_loader)

        for batch in pbar:
            optimizer.zero_grad()
            batch_loss, batch_metrics = model.compute_stage1_loss(
                batch, self.device
            )

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # clip gradients
            optimizer.step()

            for key in batch_metrics:
                epoch_metrics[key] += batch_metrics[key]
            pbar.set_description(f"Train Loss: {batch_loss.item():.4f}")

        num_batches = len(self.train_loader) if not dry_run else 10
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # clip gradients
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


    def validate_stage1_epoch(self, model: BaseModel, dry_run=False):
        model.eval()
        epoch_metrics = defaultdict(float)

        if dry_run:
            pbar = tqdm(islice(self.val_loader, 10))
        else:
            pbar = tqdm(self.val_loader)

        with torch.no_grad():
            for batch in pbar:
                batch_loss, batch_metrics = model.compute_stage1_loss(
                    batch, self.device
                )

                for key in batch_metrics:
                    epoch_metrics[key] += batch_metrics[key]

                pbar.set_description(f"Val Loss: {batch_loss.item():.4f}")

        num_batches = len(self.val_loader) if not dry_run else 10
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics
