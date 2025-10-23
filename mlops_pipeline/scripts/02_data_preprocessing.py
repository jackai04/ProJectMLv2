import os
import torch
import torchvision
import torchvision.transforms as transforms
import mlflow


def preprocess_data(batch_size=64):
    """
    Loads CIFAR-10 data, applies transforms, and logs dataset as PyTorch tensors.
    """
    # ตั้ง tracking URI ให้เก็บในโฟลเดอร์ local
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CIFAR10 - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting preprocessing with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # 1. Define preprocessing transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # 2. Load CIFAR-10
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        # จำกัด dataset เหมือนขั้นตอนก่อนหน้า (ถ้าต้องการ)
        trainset = torch.utils.data.Subset(trainset, range(10000))

        # 3. Convert to tensors and save as artifact
        os.makedirs("artifacts", exist_ok=True)

        train_data = torch.stack([data[0] for data in trainset])
        train_labels = torch.tensor([data[1] for data in trainset])
        test_data = torch.stack([data[0] for data in testset])
        test_labels = torch.tensor([data[1] for data in testset])

        torch.save((train_data, train_labels), "artifacts/train.pt")
        torch.save((test_data, test_labels), "artifacts/test.pt")

        # Log artifacts safely
        mlflow.log_artifact("artifacts/train.pt", artifact_path="processed_data")
        mlflow.log_artifact("artifacts/test.pt", artifact_path="processed_data")

        # 4. Save class names
        with open("artifacts/classes.txt", "w") as f:
            for c in testset.classes:
                f.write(c + "\n")
        mlflow.log_artifact("artifacts/classes.txt", artifact_path="processed_data")

        # 5. Log info
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("train_samples", len(trainset))
        mlflow.log_metric("test_samples", len(testset))

        print("Preprocessing finished.")
        print(f"Run ID for next step: {run_id}")


if __name__ == "__main__":
    preprocess_data()
