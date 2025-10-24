from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import base64
import uvicorn
from fastapi.responses import JSONResponse


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

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_latest_model():
    """Load latest model from MLflow or fallback"""
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
            import torchvision.models as models

            model = models.resnet18(num_classes=10)
            model.load_state_dict(
                torch.load("models/best_cifar10_model.pth", map_location=device)
            )
            model.to(device)
            model.eval()
            print("Loaded local model")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


def preprocess_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        return img, img_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


@app.on_event("startup")
async def startup_event():
    load_latest_model()


# === HTML Upload Page ===
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>CIFAR-10 Prediction</title>
            <style>
                body { font-family: Arial; background: #f9f9f9; text-align: center; padding: 40px; }
                h1 { color: #333; }
                form { background: white; padding: 20px; border-radius: 12px; display: inline-block; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                input[type="file"] { margin-bottom: 10px; }
                input[type="submit"] { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
                input[type="submit"]:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <h1>CIFAR-10 Image Classifier</h1>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """


# === Predict & Show Result Page ===
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        file_bytes = await file.read()
        img, img_tensor = preprocess_image(file_bytes)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_class = torch.max(probs, 1)
            pred_class = pred_class.item()
            conf = round(conf.item() * 100, 2)

        pred_name = class_names[pred_class]
        predictions = {
            class_names[i]: round(float(probs[0][i]) * 100, 2)
            for i in range(len(class_names))
        }

        # Convert uploaded image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Build HTML result
        html = f"""
        <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial; text-align: center; background: #f4f4f4; padding: 40px; }}
                    .card {{ background: white; padding: 30px; border-radius: 12px; display: inline-block;
                             box-shadow: 0 0 12px rgba(0,0,0,0.1); }}
                    img {{ width: 200px; border-radius: 10px; }}
                    h2 {{ color: #333; }}
                    table {{ margin: 0 auto; border-collapse: collapse; }}
                    th, td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>Prediction Result</h2>
                    <img src="data:image/png;base64,{img_b64}" alt="Uploaded Image"/>
                    <h3>Predicted: <span style="color: #007bff;">{pred_name}</span></h3>
                    <p>Confidence: <b>{conf}%</b></p>
                    <h4>All Class Probabilities</h4>
                    <table>
                        <tr><th>Class</th><th>Confidence (%)</th></tr>
                        {''.join(f"<tr><td>{cls}</td><td>{val}</td></tr>" for cls, val in predictions.items())}
                    </table>
                    <br><a href="/">Try another image</a>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


if __name__ == "__main__":
    uvicorn.run("app2:app", host="127.0.0.1", port=8000, reload=True)
