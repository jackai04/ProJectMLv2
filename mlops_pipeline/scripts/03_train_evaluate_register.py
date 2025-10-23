import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import os
import shutil


# ----------------------------
# Define BetterCNN
# ----------------------------
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------
# Test model only
# ----------------------------
def test_only(
    checkpoint_path="CIFAR10_model.pth",
    batch_size=64,
    device="cuda",
    acc_threshold=0.85,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Clear old mlruns to avoid Windows path conflicts in GitHub Actions
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")

    # Set safe tracking URI for all OS (Linux & Windows)
    mlflow.set_tracking_uri(f"file://{os.path.abspath('./mlruns')}")
    mlflow.set_experiment("CIFAR10 - Test Only")

    with mlflow.start_run(run_name="BetterCNN_test_only"):
        mlflow.set_tag("ml.step", "test_only")
        mlflow.log_param("checkpoint_path", checkpoint_path)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("acc_threshold", acc_threshold)

        # 1. Load test dataset
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        # 2. Load model checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        model = BetterCNN().to(device)

        # Fix for PyTorch 2.6+: weights_only=False allows full checkpoint load
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Support both checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"Loaded checkpoint from: {checkpoint_path}")

        # 3. Evaluate
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"\n Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # 4. Log model in MLflow
        mlflow.pytorch.log_model(model, artifact_path="cnn_model")

        # 5. Register if meets threshold
        if accuracy >= acc_threshold:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/cnn_model"
            registered_model = mlflow.register_model(model_uri, "CNN-CIFAR10")
            print(
                f"Model registered: '{registered_model.name}' version {registered_model.version}"
            )
        else:
            print(
                f"Accuracy {accuracy:.4f} below threshold {acc_threshold}. Not registered."
            )

        print("\n Testing completed successfully.")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Use relative path for portability between Windows & Linux
    ckpt_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "CIFAR10_model.pth")
    )

    test_only(
        checkpoint_path=ckpt_path,
        batch_size=64,
        device="cuda",
        acc_threshold=0.85,
    )
