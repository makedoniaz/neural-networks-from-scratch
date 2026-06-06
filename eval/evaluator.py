import numpy as np
import matplotlib.pyplot as plt

class Evaluator:
    """Evaluator for model predictions and performance metrics.

    Provides methods for generating predictions, computing accuracy, confusion matrices,
    and visualizing model performance.

    Args:
        model: The neural network model to evaluate.
        class_names: Optional list of class names for labeling. Defaults to None.
    """

    def __init__(self, model, class_names: list[str] | None = None) -> None:
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names is not None else None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions for input samples.

        Args:
            X: Input array of shape (n_samples, ...).

        Returns:
            Array of predicted class indices, shape (n_samples,).
        """
        self.model.eval()
        scores = self.model.forward(X)
        return np.argmax(scores, axis=1)

    def evaluate(self, dataloader, store_data: bool = True) -> dict:
        """Evaluate model on a dataset.

        Args:
            dataloader: DataLoader providing batches with 'data' and 'label' keys.
            store_data: Whether to store input data in results. Defaults to True.

        Returns:
            Dictionary with keys 'y_true', 'y_pred', 'scores', 'accuracy',
            and optionally 'X' if store_data is True.
        """
        self.model.eval()
        X_all = []
        y_true_all = []
        y_pred_all = []
        scores_all = []

        for batch in dataloader:
            X = batch["data"]
            y_true = batch["label"]

            scores = self.model.forward(X)
            y_pred = np.argmax(scores, axis=1)

            if store_data:
                X_all.append(X)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            scores_all.append(scores)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        scores_all = np.concatenate(scores_all)

        results = {
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "scores": scores_all,
            "accuracy": float(np.mean(y_true_all == y_pred_all)),
        }

        if store_data:
            results["X"] = np.concatenate(X_all)

        return results

    def confusion_matrix(self, results: dict) -> np.ndarray:
        """Compute confusion matrix from evaluation results.

        Args:
            results: Dictionary from evaluate() containing 'y_true' and 'y_pred'.

        Returns:
            Confusion matrix of shape (num_classes, num_classes).
        """
        y_true = results["y_true"]
        y_pred = results["y_pred"]

        num_classes = self.num_classes
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1

        cm = np.zeros((num_classes, num_classes), dtype=np.int32)

        for truth, pred in zip(y_true, y_pred):
            cm[int(truth), int(pred)] += 1

        return cm

    def plot_confusion_matrix(self, results: dict) -> None:
        """Plot confusion matrix visualization.

        Args:
            results: Dictionary from evaluate() containing prediction results.
        """
        cm = self.confusion_matrix(results)
        num_classes = cm.shape[0]

        labels = self.class_names
        if labels is None:
            labels = list(range(num_classes))

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap="Blues")

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")

        plt.xticks(range(num_classes), labels, rotation=45, ha="right")
        plt.yticks(range(num_classes), labels)

        plt.colorbar()

        threshold = cm.max() / 2

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

        plt.tight_layout()
        plt.show()

    def plot_wrong_predictions(
        self, results: dict, max_images: int = 9, image_shape: tuple[int, int] = (28, 28)
    ) -> None:
        """Visualize misclassified samples from evaluation results.

        Args:
            results: Dictionary from evaluate() containing 'X', 'y_true', and 'y_pred'.
            max_images: Maximum number of wrong predictions to display. Defaults to 9.
            image_shape: Shape to reshape images for display (height, width). Defaults to (28, 28).

        Raises:
            ValueError: If results does not contain 'X' (requires store_data=True in evaluate()).
        """
        if "X" not in results:
            raise ValueError("results must contain X. Use evaluate(..., store_data=True).")

        X = results["X"]
        y_true = results["y_true"]
        y_pred = results["y_pred"]

        wrong_indices = np.where(y_true != y_pred)[0]
        wrong_indices = wrong_indices[:max_images]

        cols = 3
        rows = int(np.ceil(len(wrong_indices) / cols))

        plt.figure(figsize=(8, 8))

        for plot_idx, data_idx in enumerate(wrong_indices):
            x = X[data_idx]
            true = int(y_true[data_idx])
            pred = int(y_pred[data_idx])

            true_name = self.class_names[true] if self.class_names else true
            pred_name = self.class_names[pred] if self.class_names else pred

            plt.subplot(rows, cols, plot_idx + 1)
            plt.imshow(x.reshape(image_shape), cmap="gray")
            plt.title(f"True: {true_name}\nPred: {pred_name}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()