import torch
import numpy as np
from utils import accuracy, reports, adjust_learning_rate

def train_epoch(train_loader, model, criterion, optimizer, epoch, device):
    """Train for one epoch with detailed progress tracking"""
    model.train()
    losses = []
    accs = []
    total_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.append(loss.item())
        acc = accuracy(outputs.data, targets.data)[0].item()
        accs.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch}][{batch_idx + 1}/{total_batches}] '
                  f'Loss: {loss.item():.4f} Acc: {acc:.2f}%')

    return np.mean(losses), np.mean(accs)

def evaluate(loader, model, criterion, device):
    """Evaluate the model with detailed metrics"""
    model.eval()
    losses = []
    accs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            losses.append(loss.item())
            accs.append(accuracy(outputs.data, targets.data, topk=(1,))[0].item())

    return np.mean(losses), np.mean(accs)

def predict(loader, model, device):
    """Get model predictions"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

def train_model(model, train_loader, test_loader, val_loader, config):
    """Train and evaluate the model with comprehensive logging"""
    print(f"\nStarting training on device: {config.DEVICE}")
    print(f"Total epochs: {config.EPOCHS}")
    print(f"Initial learning rate: {config.LEARNING_RATE}")

    device = config.DEVICE
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=True
    )

    best_acc = -1
    for epoch in range(config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{config.EPOCHS-1}")
        print(f"Learning rate: {current_lr:.6f}")

        # Training phase
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, device)

        # Evaluation phase
        if config.USE_VAL:
            eval_loss, eval_acc = evaluate(val_loader, model, criterion, device)
            eval_type = "Validation"
        else:
            eval_loss, eval_acc = evaluate(test_loader, model, criterion, device)
            eval_type = "Test"

        print(f"\nEpoch {epoch} Summary:")
        print(f"Training    - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"{eval_type} - Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.2f}%")

        if eval_acc > best_acc:
            best_acc = eval_acc
            print(f"New best {eval_type.lower()} accuracy: {best_acc:.2f}%")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': eval_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, config.MODEL_SAVE_PATH)

        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch, config)

    # Final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    test_loss, test_acc = evaluate(test_loader, model, criterion, device)
    predictions = predict(test_loader, model, device)
    y_pred = np.argmax(predictions, axis=1)
    y_test = np.array(test_loader.dataset.__labels__())

    classification, confusion, results = reports(y_pred, y_test, config.DATASET)

    return classification, confusion, results