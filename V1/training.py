import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
from vqvae import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY 
from hyper_params_sde import *
from torch import distributions, nn, optim
from train_results import FitResult, BatchResult, EpochResult
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu", scheduler=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        model.to(self.device)

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
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, test_loss= [], []

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if (epoch+1) % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # ========================
            train_result = self.train_epoch(dl_train, **kw)
            train_loss += train_result.losses
             
            test_result = self.test_epoch(dl_test,**kw)
            test_loss += test_result.losses

            train_loss_epoch = sum(train_result.losses) / len(train_result.losses)
            val_loss_epoch = sum(test_result.losses) / len(test_result.losses)
            # ========================

            if self.scheduler:
                self.scheduler.step(val_loss_epoch)

            if post_epoch_fn:
                post_epoch_fn(verbose, epoch, X_test_data, model)
                
            if epoch%save_weights_every == 0:
                checkpoint_filename = f"{checkpoints}_{epoch}_{train_loss_epoch:.4f}_{val_loss_epoch:.4f}.pt"
                torch.save(self.model.state_dict(), checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

        return FitResult(actual_num_epochs, train_loss, test_loss)
    
    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        dirname = os.path.dirname(checkpoint_filename) or "."
        os.makedirs(dirname, exist_ok=True)
        torch.save({"model_state": self.model.state_dict()}, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way. Enhanced with rich logging for
        more detailed progress information.
        """
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
                # Optionally, handle or re-raise exception

            avg_loss = sum(losses) / len(losses) if losses else float('inf')
            progress.console.log(f"{task_name} completed. Avg. Loss: {avg_loss:.4f}")

        return EpochResult(losses=losses)
    

class VQVAETrainer(Trainer):
    
    def train_batch(self, batch) -> BatchResult:
        x = batch 
        x = x.to(self.device)
        outputs = self.model(x)
        latent_loss = outputs[OTHER_KEY]['latent_loss']
        loss = self.loss_fn(outputs[MODELS_TENSOR_PREDICITONS_KEY], x) + (0.2 * latent_loss.mean())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return BatchResult(loss.item())

    def test_batch(self, batch) -> BatchResult:
        x = batch
        x = x.to(self.device) 

        with torch.no_grad():
            outputs = self.model(x)
            latent_loss = outputs[OTHER_KEY]['latent_loss']
            loss = self.loss_fn(outputs[MODELS_TENSOR_PREDICITONS_KEY], x) + (0.2 * latent_loss.mean())

        return BatchResult(loss.item())
    

class LatentSDETrainer(Trainer):
    
    def __init__(self, model, loss_fn, optimizer, device, vqvae, scheduler, kl_scheduler, logpy_metric, kl_metric, loss_metric):
        super().__init__(model, loss_fn, optimizer, device)
        self.vqvae = vqvae
        self.scheduler = scheduler
        self.kl_scheduler = kl_scheduler
        self.logpy_metric = logpy_metric
        self.kl_metric = kl_metric
        self.loss_metric = loss_metric
        
    def train_batch(self, batch) -> BatchResult:
        hyperparams = sde_hyperparams()
        
        (x,) = batch # [4,2,5*128]
        x = x.to(self.device)
        X_train = x[::4]
        row_indices = torch.arange(0, x.size(0))  # Generate row indices from 0 to batch_size-1
        not_divisible_by_4_indices = row_indices[row_indices % 4 != 0]
        X_original = x[not_divisible_by_4_indices]
        
        #### Encode using vq vae from original to latent
        self.vqvae.requires_grad_(False)
        quant_b, _, _ = self.vqvae.module.encode(X_train) # [1,8,5*128]

        # Reshaping quant_b from [batch, channels, window_size] to [batch, channels*window_size], it will arrange the values in the way you described.
        # It will first take all the values from the first channel of window_size, then concatenate the values from the second channel of window_size, and so on, 
        # until it has gone through all the channels.
        quant_b = quant_b.view(quant_b.size(0), -1) # [1, 5120]
        #### forward into SDE into latent space
        ts_ext = torch.linspace(0, 1, 5) # [5]
        self.optimizer.zero_grad()
        zs, kl = self.model(ts=ts_ext, y=quant_b, batch_size=quant_b.size(0)) # zs being [1,1,5120] is wrong! should be [5,1,5120]
        zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
        zs = zs.reshape(zs.size(0), zs.size(1), 8, 128*5) # change it
        kl = kl.mean()
        # zs should be [3,128]
        
        average_loss = 0
        #### Decode using vq vae from latent to original
        for i in range(zs.size(0)):
            for channel in range(2):
                X_reconstructed = self.vqvae.module.decode(zs[i])

                likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[hyperparams['likelihood']]
                likelihood = likelihood_constructor(loc=X_reconstructed[channel], scale=hyperparams['scale'])
                logpy = likelihood.log_prob(X_original[i][channel]).sum(dim=0).mean(dim=0)

                loss = -logpy + kl * self.kl_scheduler.val
                loss.backward()
                
                # TODO: sum losses and do backward once
                self.optimizer.step()
                self.scheduler.step()
                self.kl_scheduler.step()

                self.logpy_metric.step(logpy)
                self.kl_metric.step(kl)
                self.loss_metric.step(loss)

                average_loss += loss.item()
        
        return BatchResult(average_loss/zs.size(0))

    def test_batch(self, batch) -> BatchResult:
        hyperparams = sde_hyperparams()
        
        (x,) = batch
        x = x.to(self.device)
        X_test = x[::4]
        row_indices = torch.arange(0, x.size(0))  # Generate row indices from 0 to batch_size-1
        not_divisible_by_4_indices = row_indices[row_indices % 4 != 0]
        X_original = x[not_divisible_by_4_indices]
        
        with torch.no_grad():
            #### Encode using vq vae from original to latent
            self.vqvae.requires_grad_(False)
            quant_b, _, _ = self.vqvae.module.encode(X_test)
            quant_b = quant_b.view(quant_b.size(0), -1)

            #### forward into SDE into latent space
            ts_ext = torch.linspace(0, 1, 5)
            zs, kl = self.model(ts=ts_ext, y=quant_b, batch_size=quant_b.size(0)) # y should be [batch_size, 128]
            zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
            # zs should be [3,128]
            kl = kl.mean()
            zs = zs.reshape(zs.size(0), zs.size(1), 8, 128*5)

            average_loss = 0
            #### Decode using vq vae from latent to original
            for i in range(zs.size(0)):
                for channel in range(2):
                    self.vqvae.requires_grad_(False)
                    X_reconstructed = self.vqvae.module.decode(zs[i])

                    likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[hyperparams['likelihood']]
                    likelihood = likelihood_constructor(loc=X_reconstructed[channel], scale=hyperparams['scale'])
                    logpy = likelihood.log_prob(X_original[i][channel]).sum(dim=0).mean(dim=0)
                    loss = -logpy + kl * self.kl_scheduler.val
                    
                    average_loss += loss.item()
            
        return BatchResult(average_loss/zs.size(0), 0)
