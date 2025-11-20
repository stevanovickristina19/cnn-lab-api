import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    def __init__(self, num_conv_layers, filters_per_layer, kernel_size, fc_units, num_classes, activation="relu", input_size=(128, 128)):
        super().__init__()
        self.activation_type = activation.lower()

        # Safety checks
        if not filters_per_layer or any(f <= 0 for f in filters_per_layer):
            raise ValueError(f"filters_per_layer must contain positive integers, got {filters_per_layer}")

        if len(filters_per_layer) < num_conv_layers:
            filters_per_layer += [filters_per_layer[-1]] * (num_conv_layers - len(filters_per_layer))
        elif len(filters_per_layer) > num_conv_layers:
            filters_per_layer = filters_per_layer[:num_conv_layers]

        layers = []
        in_channels = 3
        for i in range(num_conv_layers):
            out_channels = filters_per_layer[i]
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU() if self.activation_type == "relu" else nn.Sigmoid(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # 🔹 Dynamically compute feature map size based on input_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            feat_out = self.conv(dummy)
            self.feature_dim = feat_out.view(1, -1).size(1)

        # Fully connected layers
        fc_layers = []
        in_features = self.feature_dim
        for u in fc_units:
            fc_layers += [nn.Linear(in_features, u), nn.ReLU()]
            in_features = u
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def get_model(config, num_classes=None):
    """
    Create and return a model instance.

    Behavior:
    - If model_name == "example_cnn" → loads predefined CNN_Small
    - Otherwise → builds a CustomCNN with given parameters
    """

    # Handle ModelConfig object
    if hasattr(config, "model_name"):
        model_name = config.model_name.lower()

        if model_name == "example_cnn":
            print(" Using predefined CNN_Small architecture.")
            return CNN_Small(num_classes=config.num_classes)

        print(f" Building custom CNN architecture for '{model_name}'...")
        return CustomCNN(
            num_conv_layers=config.num_conv_layers,
            filters_per_layer=config.filters_per_layer,
            kernel_size=config.kernel_size,
            fc_units=config.fc_units,
            num_classes=config.num_classes,
            activation=config.activation,
            input_size=config.input_size  
        )

    # Handle string case (legacy)
    elif isinstance(config, str):
        name = config.lower()
        if name == "example_cnn":
            return CNN_Small(num_classes=num_classes or 5)
        else:
            return CustomCNN(
                num_conv_layers=3,
                filters_per_layer=[16, 32, 64],
                kernel_size=3,
                fc_units=[128],
                num_classes=num_classes or 5,
                activation="relu"
            )

    else:
        raise TypeError("get_model() expects a ModelConfig object or a string name.")
