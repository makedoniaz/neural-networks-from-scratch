from network import Network
from optimizers import Optimizer
from dataloader import DataLoader
import numpy as np
from losses import Loss
from history import History

class Trainer():
    def __init__(self, model: Network, loss_func: Loss, optimizer: Optimizer, train_loader: DataLoader, val_loader: DataLoader = None, reg=0.0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.reg = reg

    def train(self, epochs):
        history = History()

        for epoch in range(epochs):
            train_stats = self._run_epoch(self.train_loader, training=True)
            history.append_stats("train", train_stats)

            val_stats = None

            if self.val_loader is not None:
                val_stats = self._run_epoch(self.val_loader, training=False)
                history.append_stats("val", val_stats)


            self._print_epoch(epoch, epochs, train_stats, val_stats)
        
        return history

    def evaluate(self, dataloader):
        return self._run_epoch(dataloader, training=False)

    def _run_epoch(self, dataloader, training):
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
    
    def _print_epoch(self, epoch, epochs, train_stats, val_stats=None):
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