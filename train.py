import torch
import matplotlib.pyplot as plt


def train_model(model, adjmatrix, feature_data, y_train, y_val):
    """
    Train the given GCN or Jumping Knowledge model for node classification.
    
    Args:
        model: PyTorch model (GCN or Jumping GCN).
        adjmatrix: Adjacency matrix (NumPy array).
        feature_data: Node features (PyTorch tensor on CUDA).
        y_train: Ground truth labels for training nodes.
        y_val: Ground truth labels for validation nodes.

    Returns:
        None. Prints and plots training progress.
    """

    

    # Number of training epochs
    epoch_num = 120
    # Move model to GPU
    model = model.to('cuda')

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0
    
    # Logging metrics
    epochs_train_loss = []
    epochs_val_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    # ----------- Training Loop -----------
    for epoch in range(epoch_num):
        # ---------- Forward Pass: Training ----------
        y_pred = model(feature_data, adjmatrix, "train")

        # Compute loss and accuracy for training data
        train_loss = loss_fn(y_pred, y_train)
        epochs_train_loss.append(train_loss.cpu().data)# Store scalar value


        train_acc = torch.sum(torch.argmax(y_pred, dim=1) == y_train).item() / y_train.shape[0]
        epoch_train_acc.append(train_acc)
        
        # ---------- Backward Pass ----------
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # ---------- Forward Pass: Validation ----------
        with torch.no_grad():
            y_pred_val = model(feature_data, adjmatrix, "validation")
            val_loss = loss_fn(y_pred_val, y_val)
            epochs_val_loss.append(val_loss.cpu().data)

            val_acc = torch.sum(torch.argmax(y_pred_val, dim=1) == y_val).item() / y_val.shape[0]
            epoch_val_acc.append(val_acc)


        # ---------- Logging Every 10 Epochs ----------
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epoch_num}")
            print(f"Train loss: {train_loss.item():.4f}")
            print(f"Validation loss: {val_loss.item():.4f}")
            print(f"Train accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")

            # Plot Losses
            plt.figure(figsize=(10, 5))
            plt.plot(epochs_train_loss, label='Train Loss')
            plt.plot(epochs_val_loss, label='Validation Loss')
            plt.title("Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            # Plot Accuracies
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_train_acc, label='Train Accuracy')
            plt.plot(epoch_val_acc, label='Validation Accuracy')
            plt.title("Accuracy over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.show()