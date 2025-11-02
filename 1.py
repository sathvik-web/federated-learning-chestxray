
import os
import numpy as np
import tensorflow as tf
import flwr as fl
import matplotlib.pyplot as plt
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.preprocessing.image import ImageDataGenerator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_subset_data(base_path, subset_size=500, img_size=(150, 150), num_clients=3):
    train_dir = os.path.join(base_path, "train")
    test_dir = os.path.join(base_path, "test")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=subset_size,
        class_mode="binary",
        shuffle=True
    )
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=subset_size,
        class_mode="binary",
        shuffle=False
    )

    x_train, y_train = next(train_gen)
    x_test, y_test = next(test_gen)

    client_data = []
    splits = np.array_split(np.arange(len(x_train)), num_clients)
    for i in range(num_clients):
        idx = splits[i]
        client_data.append((x_train[idx], y_train[idx]))

    return client_data, (x_test, y_test)

def create_model(input_shape=(150,150,3)):
    base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # smaller LR for stability
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(acc)}

if __name__ == "__main__":

    BASE_PATH = "C:/Users/sathv/OneDrive/Desktop/chest_xray"  
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5  

    print("Loading subset of data...")
    client_datasets, (x_test, y_test) = load_subset_data(BASE_PATH, subset_size=1000, num_clients=NUM_CLIENTS)

    print("\n Training centralized model for comparison...")
    centralized_model = create_model()
    centralized_model.fit(
        np.concatenate([ds[0] for ds in client_datasets]),
        np.concatenate([ds[1] for ds in client_datasets]),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    _, centralized_acc = centralized_model.evaluate(x_test, y_test)
    print(f"Centralized Model Accuracy: {centralized_acc:.4f}")

    def client_fn(cid: str):
        cid_int = int(cid)
        model = create_model()
        x_train, y_train = client_datasets[cid_int]
        return HospitalClient(model, x_train, y_train, x_test, y_test)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    print("\n Starting Federated Learning Simulation...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy
    )

    print("\n Simulation Complete!")

    print("Recorded metrics:", history.metrics_distributed.keys())

    rounds = list(range(1, len(history.losses_distributed)+1))
    if "accuracy" in history.metrics_distributed:
        accuracies = [acc for _, acc in history.metrics_distributed["accuracy"]]
        plt.figure(figsize=(8,5))
        plt.plot(rounds, accuracies, marker="o")
        plt.title("Federated Learning Accuracy per Round")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.show()
    else:
        print("Accuracy metric not found in history.")

