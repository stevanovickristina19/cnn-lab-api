from pydantic import BaseModel, Field, validator
from typing import List, Tuple, Optional
import math

class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Name of the model, e.g. 'custom_cnn' or 'example_cnn'")
    input_size: Tuple[int, int] = Field(default=(128, 128), description="Input image size (H, W)")
    num_classes: int = Field(default=5, description="Number of output classes")

    num_conv_layers: int = Field(default=3, description="Number of convolutional layers (auto-limited by input size)")
    filters_per_layer: List[int] = Field(default_factory=lambda: [32, 64, 128], description="Filters per conv layer")
    kernel_size: int = Field(default=3, description="Convolution kernel size (odd int ≥1)")
    fc_units: List[int] = Field(default_factory=lambda: [128], description="Units in fully connected layers")
    activation: str = Field(default="relu", description="Activation function ('relu' or 'sigmoid')")


    @validator("input_size")
    def check_input_size(cls, v):
        h, w = v
        if h < 32 or w < 32:
            raise ValueError("Input size too small; must be at least (32, 32).")
        if h > 512 or w > 512:
            raise ValueError("Input size too large; must not exceed (512, 512).")
        return v

    @validator("num_conv_layers")
    def check_num_layers(cls, v, values):
        input_size = values.get("input_size", (128, 128))
        min_side = min(input_size)
        # Max pooling reduces spatial size by 2 per layer → log2(min_side)
        max_layers = int(math.log2(min_side)) - 1
        if not 1 <= v <= max_layers:
            raise ValueError(f"num_conv_layers must be between 1 and {max_layers} for input size {input_size}.")
        return v

    @validator("filters_per_layer")
    def check_filters(cls, v, values):
        if any(f < 8 or f > 512 for f in v):
            raise ValueError("filters_per_layer values must be between 8 and 512.")
        num_layers = values.get("num_conv_layers")
        if num_layers and len(v) != num_layers:
            raise ValueError(f"filters_per_layer length ({len(v)}) must equal num_conv_layers ({num_layers}).")
        return v

    @validator("kernel_size")
    def check_kernel_size(cls, v, values):
        if v % 2 == 0 or v < 1:
            raise ValueError("kernel_size must be an odd integer ≥ 1.")
        min_side = min(values.get("input_size", (128, 128)))
        if v > min_side // 2:
            raise ValueError(f"kernel_size too large for input size {values.get('input_size')}.")
        return v

    @validator("fc_units")
    def check_fc_units(cls, v, values):
        if any(u < 16 or u > 1024 for u in v):
            raise ValueError("fc_units must be between 16 and 1024.")
        if len(v) > 4:
            raise ValueError("Too many fully connected layers (max 4).")
        return v

    @validator("activation")
    def check_activation(cls, v):
        if v.lower() not in ["relu", "sigmoid"]:
            raise ValueError("activation must be either 'relu' or 'sigmoid'.")
        return v.lower()


class TrainRequest(BaseModel):
    model_name: str = Field(default="example_cnn", description="Which model to train")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate for training (> 0)")
    epochs: int = Field(default=5, ge=1, le=100, description="Number of epochs for training (1–100)")
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for training (1–512)")


class PredictRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = Field(default="example_cnn", description="Which model weights to use for prediction")
