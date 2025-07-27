import torch
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay as ConfusionMatrix
import matplotlib.pyplot as plt

def test_model(y_pred: torch.Tensor, y_true: torch.Tensor):
    # model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    # adjmatrix = torch.from_numpy(adjmatrix).float().to('cuda')
    # y_pred = model(feature_data, adjmatrix, "test")
    test_loss = loss_fn(y_pred, y_true)
    test_acc = torch.sum(torch.argmax(y_pred, dim=1) == y_true).item() / y_true.shape[0]
    print('test loss is {}'.format(test_loss))
    print('test acc is {}'.format(test_acc))
    

def evaluate_model_performance(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 7) -> None:
    """
    Computes and displays evaluation metrics for multi-class classification:
    - F1 Score (macro average)
    - Confusion Matrix (visualization)

    Args:
        y_pred (torch.Tensor): Raw model predictions (logits or probabilities).
        y_true (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes in the classification task.
    """
    # Convert predictions to class indices
    predicted_labels = torch.argmax(y_pred, dim=1)

    # Compute F1 score
    f1 = multiclass_f1_score(predicted_labels, y_true, num_classes=num_classes, average='macro')
    print(f"Macro F1 Score: {f1:.4f}")

    # Compute and display confusion matrix
    cm = confusion_matrix(y_true.cpu(), predicted_labels.cpu())
    disp = ConfusionMatrix(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()