import os
import torch
import torchvision
import torchvision.transforms as transforms
import mlflow


def preprocess_data(batch_size=64):
    """
    Loads CIFAR-10 data, applies transforms, and logs dataset as PyTorch tensors.
    """
    # ใช้ path แบบ relative (เหมาะกับ GitHub Actions ที่เป็น Linux)
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
    MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

    # สร้างทุกโฟลเดอร์ให้เรียบร้อย
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"ARTIFACT_DIR: {ARTIFACT_DIR}")
    print(f"MLRUNS_DIR: {MLRUNS_DIR}")
    print("-" * 60)

    # ตั้ง tracking URI ในโฟลเดอร์ project เอง (ไม่แตะ root)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CIFAR10 - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting preprocessing with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        # 1. Transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # 2. Load CIFAR-10 (ใช้ data ใน project folder)
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=DATA_DIR, train=False, download=True, transform=transform
        )

        # จำกัด dataset (10,000)
        trainset = torch.utils.data.Subset(trainset, range(10000))

        # 3. Save tensors
        train_path = os.path.join(ARTIFACT_DIR, "train.pt")
        test_path = os.path.join(ARTIFACT_DIR, "test.pt")
        classes_path = os.path.join(ARTIFACT_DIR, "classes.txt")

        train_data = torch.stack([d[0] for d in trainset])
        train_labels = torch.tensor([d[1] for d in trainset])
        test_data = torch.stack([d[0] for d in testset])
        test_labels = torch.tensor([d[1] for d in testset])

        torch.save((train_data, train_labels), train_path)
        torch.save((test_data, test_labels), test_path)

        # log_artifact ใช้ path ที่อยู่ใน workspace เท่านั้น
        mlflow.log_artifact(train_path, artifact_path="processed_data")
        mlflow.log_artifact(test_path, artifact_path="processed_data")

        # Save class names
        with open(classes_path, "w") as f:
            for c in testset.classes:
                f.write(c + "\n")
        mlflow.log_artifact(classes_path, artifact_path="processed_data")

        # Log metadata
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("train_samples", len(trainset))
        mlflow.log_metric("test_samples", len(testset))

        print("Preprocessing finished successfully.")
        print(f"Run ID for next step: {run_id}")


if __name__ == "__main__":
    preprocess_data()
