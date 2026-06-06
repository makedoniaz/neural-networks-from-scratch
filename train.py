from network.network import Network
from optimizers import Optimizer
from dataloader import DataLoader
import numpy as np
from losses import Loss
from eval.history import History

class Trainer:
    """Orchestrates model training with optional validation and checkpointing.

    Handles training loops, loss computation, gradient updates, and metric tracking.

    Args:
        model: Neural network model to train.
        loss_func: Loss function for computing training loss.
        optimizer: Optimizer for updating model parameters.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data. Defaults to None.
        reg: Regularization strength. Defaults to 0.0.
        scheduler: Optional learning rate scheduler. Defaults to None.
    """

    def __init__(
        self,
        model: Network,
        loss_func: Loss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        reg: float = 0.0,
        scheduler=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.reg = reg

    def train(
        self, epochs: int, checkpoint_monitor: str | None = None, checkpoint_mode: str = "max"
    ) -> History:
        """Train the model for a specified number of epochs.

        Args:
            epochs: Number of epochs to train.
            checkpoint_monitor: Metric name to monitor for checkpointing. Defaults to None.
            checkpoint_mode: 'max' to maximize or 'min' to minimize the monitored metric. Defaults to 'max'.

        Returns:
            History object containing training metrics.

        Raises:
            ValueError: If checkpoint_monitor is set but val_loader is None.
        """
        history = History()

        checkpoint_enabled = checkpoint_monitor is not None

        if checkpoint_enabled and self.val_loader is None:
            raise ValueError(
                "Checkpointing requires a validation loader."
            )

        if checkpoint_enabled:
            monitor = checkpoint_monitor
            mode = checkpoint_mode
            best_score = -np.inf if mode == "max" else np.inf
            best_parameters = None

        for epoch in range(epochs):
            train_stats = self._run_epoch(self.train_loader, training=True)
            history.append_stats("train", train_stats)

            val_stats = None

            if self.val_loader is not None:
                val_stats = self._run_epoch(self.val_loader, training=False)
                history.append_stats("val", val_stats)

                if checkpoint_enabled:
                    current_score = val_stats[monitor]
                    improved = (
                        current_score > best_score
                        if mode == "max"
                        else current_score < best_score
                    )

                    if improved:
                        best_score = current_score
                        best_parameters = self.model.get_parameter_values()

            self._print_epoch(epoch, epochs, train_stats, val_stats)

            if self.scheduler is not None:
                self.scheduler.step()

        if checkpoint_enabled and best_parameters is not None:
            self.model.set_parameters(best_parameters)
        
        return history

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> dict:
        """Run a single epoch of training or validation.

        Args:
            dataloader: DataLoader providing batches.
            training: If True, compute gradients and update parameters.

        Returns:
            Dictionary with 'loss' and 'accuracy' metrics.
        """
        total_loss = 0
        total_samples = 0
        correct = 0

        for batch in dataloader:
            X = batch["data"]
            y = batch["label"]

            y_out = self.model.forward(X)
            loss = self.loss_func.forward(y_out, y)

            if training:
                dout = self.loss_func.backward()
                self.model.backward(dout)
                self.optimizer.step()

            preds = np.argmax(y_out, axis=1)

            batch_size = len(y)
            total_loss += loss * batch_size
            total_samples += batch_size
            correct += np.sum(preds == y)

        return {
            "loss": float(total_loss / total_samples),
            "accuracy": float(correct / total_samples),
        }
    
    def _print_epoch(
        self,
        epoch: int,
        epochs: int,
        train_stats: dict,
        val_stats: dict | None = None,
    ) -> None:
        """Print training progress for an epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            epochs: Total number of epochs.
            train_stats: Training metrics dictionary.
            val_stats: Optional validation metrics dictionary.
        """
        message = (
            f"(Epoch {epoch + 1} / {epochs}) "
            f"train loss: {train_stats['loss']:.6f}; "
            f"train accuracy: {train_stats['accuracy']:.4f}"
        )

        if val_stats is not None:
            message += (
                f"; val loss: {val_stats['loss']:.6f}; "
                f"val accuracy: {val_stats['accuracy']:.4f}"
            )

        print(message)