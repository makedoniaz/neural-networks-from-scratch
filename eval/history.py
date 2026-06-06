import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class History:
    """Container for tracking training metrics across epochs.

    Stores and visualizes metrics like loss and accuracy during model training.
    """

    def __init__(self) -> None:
        self.metrics = {}

    def append(self, metric_name: str, value: float) -> None:
        """Append a value to a metric's history.

        Args:
            metric_name: Name of the metric.
            value: Value to append.
        """
        self.metrics.setdefault(metric_name, []).append(value)

    def append_stats(self, prefix: str, stats: dict) -> None:
        """Append multiple metrics with a common prefix.

        Args:
            prefix: Prefix to prepend to each metric name.
            stats: Dictionary of metric names and values.
        """
        for metric_name, value in stats.items():
            self.append(f"{prefix}_{metric_name}", value)

    def __getitem__(self, metric_name: str) -> list:
        """Retrieve history for a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            List of values for the metric.
        """
        return self.metrics[metric_name]

    def keys(self):
        """Return all metric names."""
        return self.metrics.keys()

    def plot_metric(self, metric_name: str) -> None:
        """Plot a single metric over epochs.

        Args:
            metric_name: Name of the metric to plot.
        """
        self._plot(
            {metric_name: self.metrics[metric_name]},
            ylabel=metric_name,
            title=metric_name,
        )

    def plot_group(self, metric_type: str) -> None:
        """Plot all metrics of a given type (e.g., all metrics ending with 'loss').

        Args:
            metric_type: Suffix to filter metrics by (e.g., 'loss', 'accuracy').

        Raises:
            ValueError: If no metrics match the specified type.
        """
        metrics = {
            name: values
            for name, values in self.metrics.items()
            if name.endswith(metric_type)
        }

        if not metrics:
            raise ValueError(f"No metrics found for type: {metric_type}")

        self._plot(
            metrics,
            ylabel=metric_type.capitalize(),
            title=metric_type.capitalize(),
        )

    def plot_loss(self) -> None:
        """Plot all metrics ending with 'loss'."""
        self.plot_group("loss")

    def plot_accuracy(self) -> None:
        """Plot all metrics ending with 'accuracy'."""
        self.plot_group("accuracy")

    def _plot(self, metrics: dict, ylabel: str, title: str) -> None:
        """Plot metrics on a single figure.

        Args:
            metrics: Dictionary of metric names and value lists.
            ylabel: Label for the y-axis.
            title: Title for the plot.
        """
        plt.figure(figsize=(8, 4))

        for metric_name, values in metrics.items():
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=metric_name)

        plt.xlabel("Epoch")
        plt.xlim(left=1)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()