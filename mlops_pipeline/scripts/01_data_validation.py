# 01_data_validation.py
import mlflow
import torchvision
import torchvision.transforms as transforms


def validate_data():
    """
    Loads the CIFAR-10 dataset using PyTorch,
    performs basic validation checks, and logs the results to MLflow.
    """
    mlflow.set_experiment("CIFAR10 - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Load CIFAR-10 dataset
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        num_train = len(trainset)
        num_test = len(testset)
        num_classes = len(trainset.classes)

        print(f"Train size: {num_train}, Test size: {num_test}, Classes: {num_classes}")

        # 2. ตรวจสอบข้อมูลเบื้องต้น
        first_img, first_label = trainset[0]
        img_shape = tuple(first_img.shape)
        print(f"First image shape: {img_shape}, Label: {trainset.classes[first_label]}")

        # 3. Log metrics and parameters
        mlflow.log_metric("num_train", num_train)
        mlflow.log_metric("num_test", num_test)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("image_shape", str(img_shape))

        validation_status = "Success"
        mlflow.log_param("validation_status", validation_status)

        print("Data validation completed successfully.")


if __name__ == "__main__":
    validate_data()
