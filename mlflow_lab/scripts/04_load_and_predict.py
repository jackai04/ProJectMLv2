# 04_load_and_predict.py
import mlflow
import torch
from torchvision import transforms
from PIL import Image


def load_and_predict(
    image_path=r"C:\sitthisak\gitkub\ProJectMLv2\cat.png",
):
    MODEL_NAME = "CNN-CIFAR10"  # ตรงกับตอน register
    MODEL_STAGE = "Staging"  # เปลี่ยนเป็น "Production" หรือ "Staging" ได้ถ้า promote แล้ว

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    # โหลดโมเดลจาก MLflow
    try:
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # ใช้ normalization เดียวกับตอนเทรน
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    # เปิดภาพ
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # ส่งเข้า device ด้วย

    # ทำนายผล
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    print(f"Predicted class index: {predicted.item()}")
    print(f"Predicted label: {classes[predicted.item()]}")


if __name__ == "__main__":
    load_and_predict()
