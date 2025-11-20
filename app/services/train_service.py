import torch
import torch.nn as nn
import torch.optim as optim
from app.models.models import get_model
from app.utils.dataset_loader import get_dataloaders
import matplotlib.pyplot as plt
import os


def train_with_params(config, request):
    """
    Train a model using both architecture (ModelConfig)
    and hyperparameters (TrainRequest), evaluate on test data,
    and save a training curve PNG for display in Swagger UI.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device}")

    # Build model
    model = get_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=request.learning_rate)

    # Load dataset
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=request.batch_size, input_size=config.input_size)

    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(request.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch+1}/{request.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Save trained model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/{config.model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f" Model saved to {model_path}")

    # Evaluate on test dataset
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f" Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

    # Save training plot
    plot_path = f"saved_models/{config.model_name}_training_plot.png"
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="o")
    plt.title("Training progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    lr_text = f"Learning Rate: {request.learning_rate}"
    plt.text(
    0.02, 0.95, lr_text,
    transform=plt.gca().transAxes,  # position relative to axes (0–1)
    fontsize=10, color="darkblue", weight="bold",
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4ff", edgecolor="blue", alpha=0.5))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved training curve: {plot_path}")

    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "plot_url": f"/model/training_plot/{config.model_name}"
    }


def evaluate(model, dataloader, criterion, device):
    """Compute loss and accuracy on given dataloader."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
