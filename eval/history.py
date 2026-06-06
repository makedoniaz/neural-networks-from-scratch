import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class History:
    def __init__(self):
        self.metrics = {}

    def append(self, metric_name, value):
        self.metrics.setdefault(metric_name, []).append(value)

    def append_stats(self, prefix, stats):
        for metric_name, value in stats.items():
            self.append(f"{prefix}_{metric_name}", value)

    def __getitem__(self, metric_name):
        return self.metrics[metric_name]

    def keys(self):
        return self.metrics.keys()

    def plot_metric(self, metric_name):
        self._plot(
            {metric_name: self.metrics[metric_name]},
            ylabel=metric_name,
            title=metric_name,
        )

    def plot_group(self, metric_type):
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

    def plot_loss(self):
        self.plot_group("loss")

    def plot_accuracy(self):
        self.plot_group("accuracy")

    def _plot(self, metrics, ylabel, title):
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