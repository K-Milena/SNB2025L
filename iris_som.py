# === ZAAWANSOWANA ANALIZA DLA ETAPU 3 ===
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import os

# === PARAMETRY ===
learning_rates = [0.1, 0.5, 0.9]
grid_sizes = [(5, 5), (10, 10), (15, 15)]
num_epochs = 50

# === DANE ===
iris = load_iris()
data = iris.data
labels = iris.target
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === KLASA SOM ===
class SOM:
    def __init__(self, x, y, input_len, learning_rate=0.5, radius=None, num_epochs=100):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.radius = radius if radius else max(x, y) / 2
        self.time_constant = self.num_epochs / np.log(self.radius)
        self.weights = np.random.rand(x, y, input_len)
        self.neuron_locations = np.array(list(self._neuron_locations()))
        self.errors = []

    def _neuron_locations(self):
        for i in range(self.x):
            for j in range(self.y):
                yield np.array([i, j])

    def _find_bmu(self, sample):
        bmu_idx = None
        min_dist = np.inf
        for loc in self.neuron_locations:
            w = self.weights[loc[0], loc[1], :]
            dist = np.linalg.norm(sample - w)
            if dist < min_dist:
                min_dist = dist
                bmu_idx = loc
        return bmu_idx, min_dist

    def _decay_radius(self, epoch):
        return self.radius * np.exp(-epoch / self.time_constant)

    def _decay_learning_rate(self, epoch):
        return self.learning_rate * np.exp(-epoch / self.num_epochs)

    def train(self, data):
        for epoch in range(self.num_epochs):
            total_error = 0
            for sample in data:
                bmu, error = self._find_bmu(sample)
                total_error += error
                radius = self._decay_radius(epoch)
                lr = self._decay_learning_rate(epoch)
                for loc in self.neuron_locations:
                    dist_to_bmu = np.linalg.norm(loc - bmu)
                    if dist_to_bmu <= radius:
                        influence = np.exp(-(dist_to_bmu ** 2) / (2 * (radius ** 2)))
                        delta = lr * influence * (sample - self.weights[loc[0], loc[1], :])
                        self.weights[loc[0], loc[1], :] += delta
            self.errors.append(total_error)

    def map_input(self, data):
        return [self._find_bmu(x)[0] for x in data]

# === WYNIKI ===
metrics_summary = []
os.makedirs("results", exist_ok=True)

for lr in learning_rates:
    for grid_x, grid_y in grid_sizes:
        # Trenowanie SOM
        som = SOM(x=grid_x, y=grid_y, input_len=4, learning_rate=lr, num_epochs=num_epochs)
        som.train(data_scaled)
        mapped = som.map_input(data_scaled)

        # Mapowanie klas
        neuron_label_map = defaultdict(list)
        for pos, label in zip(mapped, labels):
            neuron_label_map[tuple(pos)].append(label)
        neuron_class_map = {}
        for neuron, classes in neuron_label_map.items():
            neuron_class_map[neuron] = np.bincount(classes).argmax()
        predictions = [neuron_class_map[tuple(bmu)] for bmu in mapped]

        # Metryki
        conf_matrix = confusion_matrix(labels, predictions)
        class_report = classification_report(labels, predictions, output_dict=True)

        sensitivity = np.mean([class_report[str(i)]['recall'] for i in range(3)])
        specificity = np.mean([class_report[str(i)]['precision'] for i in range(3)])

        metrics_summary.append({
            'learning_rate': lr,
            'grid_size': f"{grid_x}x{grid_y}",
            'sensitivity': sensitivity,
            'specificity': specificity
        })

        # Zapis błędu uczenia - wykres
        plt.figure()
        plt.plot(som.errors)
        plt.title(f"Error LR={lr}, Grid={grid_x}x{grid_y}")
        plt.xlabel("Epoch")
        plt.ylabel("Total Error")
        plt.grid(True)
        plt.savefig(f"results/error_lr{lr}_grid{grid_x}x{grid_y}.png")
        plt.close()

        # **NOWOŚĆ: Zapis błędów uczenia do txt**
        with open(f"results/training_errors_lr{lr}_grid{grid_x}x{grid_y}.txt", "w") as f_err:
            for epoch_idx, err in enumerate(som.errors):
                f_err.write(f"Epoch {epoch_idx+1}: {err}\n")

# === WIZUALIZACJA PORÓWNANIA METRYK ===
sens = [m['sensitivity'] for m in metrics_summary]
specs = [m['specificity'] for m in metrics_summary]
labels_bar = [f"lr={m['learning_rate']}, {m['grid_size']}" for m in metrics_summary]

x = np.arange(len(metrics_summary))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, sens, width, label='Sensitivity')
plt.bar(x + width/2, specs, width, label='Specificity')
plt.xticks(x, labels_bar, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Sensitivity and Specificity across Configs")
plt.legend()
plt.tight_layout()
plt.savefig("results/sensitivity_specificity_comparison.png")
plt.close()

# === ZAPIS TABELI METRYK ===
with open("results/metrics_summary.txt", "w") as f:
    for m in metrics_summary:
        f.write(f"Learning rate: {m['learning_rate']}, Grid: {m['grid_size']}, "
                f"Sensitivity: {m['sensitivity']:.3f}, Specificity: {m['specificity']:.3f}\n")
        
# Zapis błędów uczenia w txt
with open(f"results/training_errors_lr{lr}_grid{grid_x}x{grid_y}.txt", "w") as f_err:
    for epoch_idx, err in enumerate(som.errors):
        f_err.write(f"Epoch {epoch_idx+1}: {err}\n")


print("Etap 3 DONE: metryki, wykresy i pliki zapisane w folderze 'results'")
