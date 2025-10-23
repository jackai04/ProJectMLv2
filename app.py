# http://127.0.0.1:8000/docs#/
# http://127.0.0.1:8000/

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import uvicorn

app = FastAPI(
    title="CNN-CIFAR10 API",
    description="API for classifying images using CIFAR-10 PyTorch model",
)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
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

# Image preprocessing for CIFAR-10
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_latest_model():
    """Load latest model from MLflow or fallback to local .pth"""
    global model
    try:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            "CNN-CIFAR10", stages=["Production"]
        )

        if latest_versions:
            model_uri = f"models:/CNN-CIFAR10/Production"
            model = mlflow.pytorch.load_model(model_uri, map_location=device)
            model.eval()
            print(f"Loaded model from MLflow: {model_uri}")
        else:
            # fallback local model
            import torchvision.models as models

            model = models.resnet18(num_classes=10)
            model.load_state_dict(
                torch.load("models/best_cifar10_model.pth", map_location=device)
            )
            model.to(device)
            model.eval()
            print("Loaded local PyTorch model")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


def preprocess_image(file_bytes):
    """Preprocess uploaded image"""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.on_event("startup")
async def startup_event():
    load_latest_model()


# HTML page for upload
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>CIFAR-10 Prediction</title>
        </head>
        <body>
            <h1>Upload Image for CIFAR-10 Prediction</h1>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "device": str(device),
        "number_of_classes": len(class_names),
        "class_names": class_names,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        img_tensor = preprocess_image(await file.read())
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probs, 1)
            pred_class = pred_class.item()
            conf = round(conf.item(), 4)

        all_predictions = {
            class_names[i]: round(float(probs[0][i]), 6)
            for i in range(len(class_names))
        }

        response = {
            "predicted_class": class_names[pred_class],
            "class_id": pred_class,
            "confidence": conf,
            "all_predictions": all_predictions,
        }

        # Return JSON response (browser will show it)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
