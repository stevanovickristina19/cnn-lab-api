from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.schemas.model_schemas import ModelConfig, TrainRequest, PredictRequest
from app.services import train_service, predict_service
import os, json, base64

router = APIRouter(prefix="/model", tags=["Model API"])

CONFIG_DIR = "saved_models"
os.makedirs(CONFIG_DIR, exist_ok=True)

current_model = {}


def save_model_config(config: ModelConfig):
    """Save model configuration."""
    config_path = os.path.join(CONFIG_DIR, f"{config.model_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config.dict(), f)
    print(f" Saved model config to {config_path}")


def load_model_config(model_name: str):
    """Load a model configuration if it exists."""
    config_path = os.path.join(CONFIG_DIR, f"{model_name}_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


@router.get("/list_models", summary=" List all trained models")
def list_models():
    """List all available trained models in the 'saved_models' directory."""
    models = []
    for file in os.listdir(CONFIG_DIR):
        if file.endswith(".pth"):
            name = file.replace(".pth", "")
            cfg_path = os.path.join(CONFIG_DIR, f"{name}_config.json")
            models.append({
                "model_name": name,
                "config_exists": os.path.exists(cfg_path)
            })
    if not models:
        return {"message": "No trained models found yet."}
    return {"available_models": models}

@router.post("/create_model", summary=" Create a new model configuration")
def create_model(config: ModelConfig):
    """Create a new model configuration. """
    save_model_config(config)
    current_model["config"] = config
    return {
        "message": f" Model '{config.model_name}' created successfully",
        "details": config.dict()
    }


@router.post("/train", summary=" Train created model")
def train_model(request: TrainRequest):
    """Train model using stored configuration + training parameters."""
    if "config" not in current_model:
        cfg_data = load_model_config(request.model_name)
        if not cfg_data:
            raise HTTPException(status_code=400, detail="No model created yet.")
        current_model["config"] = ModelConfig(**cfg_data)

    config = current_model["config"]
    print(f" Training '{config.model_name}' with lr={request.learning_rate}, epochs={request.epochs}, batch_size={request.batch_size}")

    result = train_service.train_with_params(config, request)
    return {
        "message": " Training completed",
        "metrics": result,
        "plot_url": f"/model/training_plot/{config.model_name}"
    }


@router.get("/load_model", summary=" Load a trained model")
def load_model(model_name: str):
    """Load a previously trained model for prediction/testing."""
    cfg_data = load_model_config(model_name)
    if not cfg_data:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' config not found.")
    current_model["config"] = ModelConfig(**cfg_data)
    print(f" Loaded model '{model_name}' into memory.")
    return {"message": f"Model '{model_name}' loaded successfully and is now active for prediction."}


@router.post("/test_predict", summary=" Upload an image for prediction")
async def test_predict(file: UploadFile = File(...)):
    """Upload an image to test the currently loaded model."""
    if "config" not in current_model:
        raise HTTPException(status_code=400, detail="No model created or loaded yet. Use /load_model first.")

    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    model_name = current_model["config"].model_name

    prediction = predict_service.predict(image_base64, model_name)
    return {"filename": file.filename, "model_used": model_name, "prediction": prediction}

@router.get("/training_plot", summary=" View training accuracy/loss graph")
def get_training_plot(model_name: str):
    """Serve the saved training loss curve as an image."""
    plot_path = os.path.join(CONFIG_DIR, f"{model_name}_training_plot.png")
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Training plot not found. Train the model first.")
    return FileResponse(plot_path, media_type="image/png", filename=f"{model_name}_training_plot.png")
