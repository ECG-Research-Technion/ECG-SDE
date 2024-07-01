import os
import abc
import sys
import torch
from typing import Any, Callable
from statistics import mean
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_results import FitResult, BatchResult, EpochResult
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table
from torch import nn


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu", scheduler=None, log_dir="logs", enable_tensorboard=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.enable_tensorboard = enable_tensorboard
        self.writer = SummaryWriter(log_dir) if enable_tensorboard else None
        model.to(self.device)

        # Register hooks if TensorBoard is enabled
        # if self.enable_tensorboard:
        #     self._register_hooks()

    def _log_layer_outputs(self, epoch, data_sample):
        def hook_fn(module, input, output, name):
            if self.writer:
                self.writer.add_histogram(f'{name}_output', output, epoch)

        hooks = []

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.ConvTranspose1d):
                hook = layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
                hooks.append(hook)
        
        with torch.no_grad():
            self.model(data_sample.to(self.device))

        for hook in hooks:
            hook.remove()

    def _log_histograms(self, epoch):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.ConvTranspose1d):
                if hasattr(layer, 'weight'):
                    self.writer.add_histogram(f'{name}_weight', layer.weight, epoch)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    self.writer.add_histogram(f'{name}_bias', layer.bias, epoch)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        save_weights_every=10,
        post_epoch_fn=None,
        X_test_data=None,
        model=None,
        wrapper_model=None,
        **kw,
    ) -> FitResult:
        actual_num_epochs = 0
        train_loss, test_loss = [], []

        data_sample = next(iter(dl_test))[0]

        for epoch in range(num_epochs):
            verbose = (epoch + 1) % print_every == 0 or epoch == num_epochs - 1
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train, **kw)
            train_loss += train_result.losses

            test_result = self.test_epoch(dl_test, **kw)
            test_loss += test_result.losses

            train_loss_epoch = mean(train_result.losses)
            test_loss_epoch = mean(test_result.losses)

            if self.scheduler:
                self.scheduler.step(test_loss_epoch)

            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss_epoch, epoch)
                self.writer.add_scalar('Loss/Test', test_loss_epoch, epoch)
                self._log_histograms(epoch)
                self._log_layer_outputs(epoch, data_sample)

            if post_epoch_fn:
                post_epoch_fn(verbose, epoch, X_test_data, model)

            if epoch % save_weights_every == 0:
                checkpoint_filename = f"{checkpoints}_{epoch}_{train_loss_epoch:.4f}_{test_loss_epoch:.4f}.pt"
                self.save_checkpoint(checkpoint_filename)
                print(f"*** Saved checkpoint {checkpoint_filename} at epoch {epoch+1}")

        return FitResult(actual_num_epochs, train_loss, test_loss)

    def save_checkpoint(self, checkpoint_filename: str):
        dirname = os.path.dirname(checkpoint_filename) or "."
        os.makedirs(dirname, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        self.model.eval()
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        losses = []
        num_batches = len(dl) if max_batches is None else min(len(dl), max_batches)
        console = Console()

        custom_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[yellow]{task.completed}/{task.total} batches"),
            TextColumn("[bold green]{task.fields[loss]}"),
            TimeElapsedColumn(),
        ]

        progress = Progress(*custom_columns, expand=True)
        task_name = f"{forward_fn.__name__.replace('_', ' ').capitalize()} Processing"

        with progress:
            task = progress.add_task(task_name, total=num_batches, loss="Avg Loss: N/A")
            try:
                for batch_idx, data in enumerate(dl):
                    if max_batches and batch_idx >= max_batches:
                        break
                    batch_res = forward_fn(data)
                    losses.append(batch_res.loss)
                    avg_loss = sum(losses) / len(losses)
                    progress.update(task, advance=1, loss=f"Avg Loss: {avg_loss:.4f}")
                    
            except Exception as e:
                console.log(f"[red]Exception during training: {e}[/red]")

            avg_loss = sum(losses) / len(losses) if losses else float('inf')
            progress.console.log(f"{task_name} completed. Avg. Loss: {avg_loss:.4f}")

        return EpochResult(losses=losses)


class VQVAETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x = batch 
        x = x.to(self.device)
        outputs = self.model(x)
        latent_loss = outputs['latent_loss']
        loss = self.loss_fn(outputs['reconstructed'], x) + (0.25 * latent_loss.mean())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return BatchResult(loss.item())

    def test_batch(self, batch) -> BatchResult:
        x = batch
        x = x.to(self.device) 

        with torch.no_grad():
            outputs = self.model(x)
            latent_loss = outputs['latent_loss']
            loss = self.loss_fn(outputs['reconstructed'], x) + (0.25 * latent_loss.mean())

        return BatchResult(loss.item())