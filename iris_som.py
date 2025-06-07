import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

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

# === ŁADOWANIE DANYCH ===
iris = load_iris()
data = iris.data
labels = iris.target
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# === INICJALIZACJA I TRENING ===
som = SOM(x=10, y=10, input_len=4, learning_rate=0.5, num_epochs=50)
som.train(data_scaled)
mapped = som.map_input(data_scaled)

# === MAPOWANIE KLAS DO NEURONÓW ===
neuron_label_map = defaultdict(list)
for pos, label in zip(mapped, labels):
    neuron_label_map[tuple(pos)].append(label)

# === ZAMIENIAMY NA DOMINUJĄCE KLASY DLA KAŻDEGO NEURONU ===
neuron_class_map = {}
for neuron, classes in neuron_label_map.items():
    dominant_class = np.bincount(classes).argmax()  # Dominująca klasa
    neuron_class_map[neuron] = dominant_class

# === GENERUJEMY PREDYKCJE NA PODSTAWIE MAPOWANIA NEURONÓW ===
predictions = [neuron_class_map[tuple(bmu)] for bmu in mapped]

# === PRINTUJ NEURONY Z DOMINUJĄCYMI KLASAMI ===
for neuron, class_label in list(neuron_class_map.items())[:10]:
    print(f"Neuron {neuron}: Class {class_label}")

# === PREDYKCJA I METRYKI ===
conf_matrix = confusion_matrix(labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Oblicz metryki klasyfikacyjne
print("Classification Report:")
print(classification_report(labels, predictions))

# === WIZUALIZACJE ===

# 1. Błąd treningowy
plt.figure(figsize=(8, 6))
plt.plot(som.errors)
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.title("Learning Error over Time")
plt.grid(True)
plt.savefig("training_error.png")
plt.close()

# 2. Rozkład neuronów z przypisanymi klasami
plt.figure(figsize=(8, 6))
plt.scatter(
    [neuron[0] for neuron in neuron_class_map.keys()],
    [neuron[1] for neuron in neuron_class_map.keys()],
    c=list(neuron_class_map.values()), cmap='viridis')
plt.colorbar(label="Class")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")

plt.title("Neuron Map with Class Labels")
plt.xlabel("Neuron X")
plt.ylabel("Neuron Y")
plt.savefig("neuron_map.png")
plt.close()

# 3. Rozkład danych wejściowych na mapie

# Tworzymy kolorową mapę ręcznie
colors = ['blue', 'green', 'orange']
names = ['setosa', 'versicolor', 'virginica']

for i in range(3):
    plt.scatter(data_scaled[labels == i, 0], data_scaled[labels == i, 1],
                label=names[i], c=colors[i], alpha=0.7)

plt.legend(title="Iris Class")
plt.title("Input Data Distribution")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.savefig("input_data_labeled.png")
plt.close()


# 4. Macierz pomyłek
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(3), [0, 1, 2])
plt.yticks(np.arange(3), [0, 1, 2])
plt.savefig("confusion_matrix.png")
plt.close()

# 5. Raport klasyfikacji - zapis do pliku
with open("classification_report.txt", "w") as f:
    f.write(classification_report(labels, predictions))
    print("Classification report saved as classification_report.txt")
