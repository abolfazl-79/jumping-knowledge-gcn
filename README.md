
# 🧠 Jumping Knowledge GCN for Node Classification on Cora

This project implements **Graph Convolutional Networks (GCNs)** and the **Jumping Knowledge Network** (JK-Net) for **node classification** on the widely used **Cora citation dataset**. It supports multiple aggregation strategies such as:
- 📌 **Max pooling**
- 📌 **Weighted layer mixture**
- 📌 **LSTM-based attention**

The model is built with **PyTorch** and **NetworkX**, and includes visualization tools for graph statistics and classification results.

---

## 📁 Project Structure

```bash
Jumping-Knowledge-GCN/
│
├── data/
│   ├── cora.content              # Node features + labels
│   └── cora.cites                # Edgelist
│
├── utils/
│   ├── preprocessing.py          # Data loading and splitting
│   ├── visualization.py          # Node degree & graph visualizations
│   └── evaluate.py               # F1-score, confusion matrix plotting
│
├── model/
│   ├── gcn_layer.py              # GCN layer definition
│   ├── gcn.py                    # Traditional multi-layer GCN
│   └── jumping_knowledge.py     # Jumping Knowledge GCN model
│
├── train.py                      # Training script
├── requirements.txt              # Dependencies
└── README.md                     # This file
