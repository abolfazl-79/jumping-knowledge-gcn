import torch
import pandas as pd

# GCN Layer as described in Kipf & Welling (2016)
class GCN_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameter()

    # Initialize weights using Xavier (Glorot) initialization
    def reset_parameter(self):
      torch.nn.init.xavier_uniform_(self.weight)

     # Message passing step: apply weight matrix to input features
    def message_passing(self, x):
      return torch.matmul(x, self.weight)

    # Aggregation step: apply normalized adjacency to messages
    def aggregate(self, normalized_adj, x):
      return torch.matmul(normalized_adj, x)

    # Forward pass = message passing → aggregation
    def forward(self, x, normalized_adj):
      x = self.message_passing(x)
      x = self.aggregate(normalized_adj, x)
      return x


class Traditional_GCN(torch.nn.Module):
  """
    Traditional GCN model with multiple GCN layers followed by linear layers.

    Architecture:
    - 5 GCN layers with batch normalization, ReLU, and dropout
    - 2 fully connected layers for final classification
    """
  def __init__(self, INITIAL_FEATURES: int, node_data: pd.DataFrame):
    super(Traditional_GCN, self).__init__()
    self.node_data = node_data

    # First GCN layer: input features → 32-dim
    self.gcn1 = GCN_layer(INITIAL_FEATURES, 32)
    self.batch_norm1 = torch.nn.BatchNorm1d(32)
    self.relu1 = torch.nn.ReLU()
    self.drop1 = torch.nn.Dropout(0.5)

    # Second GCN layer
    self.gcn2 = GCN_layer(32, 32)
    self.batch_norm2 = torch.nn.BatchNorm1d(32)
    self.relu2 = torch.nn.ReLU()
    self.drop2 = torch.nn.Dropout(0.5)

    # Third GCN layer
    self.gcn3 = GCN_layer(32, 32)
    self.batch_norm3 = torch.nn.BatchNorm1d(32)
    self.relu3 = torch.nn.ReLU()
    self.drop3 = torch.nn.Dropout(0.5)


    # Fourth GCN layer
    self.gcn4 = GCN_layer(32, 32)
    self.batch_norm4 = torch.nn.BatchNorm1d(32)
    self.relu4 = torch.nn.ReLU()
    self.drop4 = torch.nn.Dropout(0.5)


    # Fifth GCN layer
    self.gcn5 = GCN_layer(32, 32)
    self.batch_norm5 = torch.nn.BatchNorm1d(32)
    self.relu5 = torch.nn.ReLU()
    self.drop5 = torch.nn.Dropout(0.5)

    # First fully connected layer
    self.linear6 = torch.nn.Linear(32, 32)
    self.batch_norm6 = torch.nn.BatchNorm1d(32)
    self.relu6 = torch.nn.ReLU()
    self.drop6 = torch.nn.Dropout(0.5)

    # Output layer for classification
    self.linear7 = torch.nn.Linear(32, 32)
    self.softmax = torch.nn.Softmax(dim=1)


  def normalising_adjmatrix(self, adjmatrix):
      """
        Compute symmetric normalized adjacency matrix: Â = D^(-1/2) * (A + I) * D^(-1/2)
      """
      adj_thelda = adjmatrix + torch.eye(adjmatrix.shape[0]).to('cuda')
      D_thelda = torch.sum(adj_thelda, dim=1)
      D_thelda = torch.diag(D_thelda)
      D_thelda_sqrt = torch.sqrt(D_thelda)
      D_thelda_sqrt_inv = torch.linalg.inv(D_thelda_sqrt)
      return torch.matmul(torch.matmul(D_thelda_sqrt_inv, adj_thelda), D_thelda_sqrt_inv)


  def forward(self, x, adjmatrix, node_status):
    """
        Forward pass through all GCN layers and final classifier.
    """
    normalized_adj = self.normalising_adjmatrix(adjmatrix)

    # Layer 1
    x = self.gcn1(x, normalized_adj)
    x = self.batch_norm1(x)
    x = self.relu1(x)
    x = self.drop1(x)

    # Layer 2
    x = self.gcn2(x, normalized_adj)
    x = self.batch_norm2(x)
    x = self.relu2(x)
    x = self.drop2(x)

    # Layer 3
    x = self.gcn3(x, normalized_adj)
    x = self.batch_norm3(x)
    x = self.relu3(x)
    x = self.drop3(x)

    # Layer 4
    x = self.gcn4(x, normalized_adj)
    x = self.batch_norm4(x)
    x = self.relu4(x)
    x = self.drop4(x)

    # Layer 5
    x = self.gcn5(x, normalized_adj)
    x = self.batch_norm5(x)
    x = self.relu5(x)
    x = self.drop5(x)
    
    # Filter only train/val/test nodes
    x = x[self.node_data.where(self.node_data['node_status'] == node_status).dropna().index]

    # Fully connected layers
    x = self.linear6(x)
    x = self.batch_norm6(x)
    x = self.relu6(x)
    x = self.drop6(x)

    x = self.linear7(x)
    x = self.softmax(x)
    return x
  


class Jumping_GCN(torch.nn.Module):

  """
    Jumping Knowledge GCN model as described in the JK-Net paper.
    Combines outputs from multiple GCN layers using one of:
    - Max-pooling
    - Learnable weighted sum (mixture)
    - LSTM-based attention
    
    Architecture:
    - 5 stacked GCN layers (with normalization, ReLU, dropout)
    - Jumping Knowledge layer (pooling/mixture/attention)
    - 2 linear layers for classification
    """
  
  def __init__(self, INITIAL_FEATURES: int, node_data: pd.DataFrame) -> None:
    super(Jumping_GCN, self).__init__()
    self.node_data = node_data

    # ----------- GCN Layers -----------
    self.gcn1 = GCN_layer(INITIAL_FEATURES, 32)
    self.batch_norm1 = torch.nn.BatchNorm1d(32)
    self.relu1 = torch.nn.ReLU()
    self.drop1 = torch.nn.Dropout(0.5)

    self.gcn2 = GCN_layer(32,32)
    self.batch_norm2 = torch.nn.BatchNorm1d(32)
    self.relu2 = torch.nn.ReLU()
    self.drop2 = torch.nn.Dropout(0.5)

    self.gcn3 = GCN_layer(32,32)
    self.batch_norm3 = torch.nn.BatchNorm1d(32)
    self.relu3 = torch.nn.ReLU()
    self.drop3 = torch.nn.Dropout(0.5)

    self.gcn4 = GCN_layer(32,32)
    self.batch_norm4 = torch.nn.BatchNorm1d(32)
    self.relu4 = torch.nn.ReLU()
    self.drop4 = torch.nn.Dropout(0.5)

    self.gcn5 = GCN_layer(32,32)
    self.batch_norm5 = torch.nn.BatchNorm1d(32)
    self.relu5 = torch.nn.ReLU()
    self.drop5 = torch.nn.Dropout(0.5)

    # ----------- Linear Classifier Layers -----------
    self.linear6 = torch.nn.Linear(32,32)
    self.batch_norm6 = torch.nn.BatchNorm1d(32)
    self.relu6 = torch.nn.ReLU()
    self.drop6 = torch.nn.Dropout(0.5)

    # 7-class classification
    self.linear7 = torch.nn.Linear(32,7)
    self.softmax = torch.nn.Softmax(dim=1)

    # ----------- Jumping Knowledge Aggregation Tools -----------
    self.embeddings = [] # Stores final graph representations
    self.alphas = torch.nn.Parameter(torch.ones(4))  # Learnable layer weights for mixture method
    
    # Optional: LSTM-based attention
    self.lstm = torch.nn.LSTM(32, 16, num_layers=1, batch_first=True, bidirectional=True)
    self.linear8 = torch.nn.Linear(32,1) # For computing attention scores

  # ----------- Jumping Knowledge Methods -----------

  def max_pooling(self, gcn_outputs):
    """
        Element-wise max over all GCN layer outputs.
        Input shape: list of [num_nodes, dim]
        Output shape: [num_nodes, dim]
    """
    gcn_outputs = torch.stack(gcn_outputs, dim=0)# [num_layers, num_nodes, dim]
    return torch.max(gcn_outputs, dim=0)[0]


  def mixture_outputs(self, gcn_outputs):
    """
        Weighted sum over layer outputs using learned parameters (alphas).
    """
    gcn_outputs = torch.stack(gcn_outputs, dim=0)  # [num_layers, num_nodes, dim]
    gcn_outputs = gcn_outputs.permute(2,1,0)  # [dim, num_nodes, num_layers]
    alphas_prob = torch.softmax(self.alphas, dim=0)  # Normalize alphas
    final_embbeding = torch.matmul(gcn_outputs, alphas_prob) # [dim, num_nodes]
    return final_embbeding.permute(1,0) # [num_nodes, dim]

  def lstm_attention(self, gcn_outputs):
    """
        Use Bi-LSTM to attend over outputs from each GCN layer.
        Produces a weighted sum of outputs with learned attention weights.
    """
    gcn_outputs = torch.stack(gcn_outputs, dim=0) # [num_layers, num_nodes, dim]
    gcn_outputs = gcn_outputs.permute(1,0,2) # [num_nodes, num_layers, dim]
    output, (hidden_state, cell_state) = self.lstm(gcn_outputs)  # LSTM returns [num_nodes, num_layers, hidden*2]
    linear_output = self.linear8(output)
    linear_output = linear_output.squeeze(-1)# [num_nodes, num_layers]
    Prob = torch.nn.functional.softmax(linear_output, dim=1)
    prob_mean = Prob.mean(dim=0)   # Global attention score for each layer
    gcn_outputs = gcn_outputs.permute(0,2,1) # [num_nodes, dim, num_layers]
    final_gcn_output = torch.matmul(gcn_outputs, prob_mean.squeeze(-1)) # [num_nodes, dim]
    return final_gcn_output

  def normalising_adjmatrix(self, adjmatrix):
      """
        Return symmetric normalized adjacency matrix: Â = D^(-1/2) * (A + I) * D^(-1/2)
      """
      adj_thelda = adjmatrix + torch.eye(adjmatrix.shape[0]).to('cuda')
      D_thelda = torch.sum(adj_thelda, dim=1)
      D_thelda = torch.diag(D_thelda)
      D_thelda_sqrt = torch.sqrt(D_thelda)
      D_thelda_sqrt_inv = torch.linalg.inv(D_thelda_sqrt)
      return torch.matmul(torch.matmul(D_thelda_sqrt_inv, adj_thelda), D_thelda_sqrt_inv)

  # ----------- Forward Pass -----------
  def forward(self, x, adjmatrix, node_status):
    gcn_outputs = []
    normalized_adj = self.normalising_adjmatrix(adjmatrix)

    # -------- Pass through GCN layers --------
    x = self.gcn1(x, normalized_adj)
    x = self.batch_norm1(x)
    x = self.relu1(x)
    x = self.drop1(x)
    gcn_outputs.append(x)

    x = self.gcn2(x, normalized_adj)
    x = self.batch_norm2(x)
    x = self.relu2(x)
    x = self.drop2(x)
    gcn_outputs.append(x)

    x = self.gcn3(x, normalized_adj)
    x = self.batch_norm3(x)
    x = self.relu3(x)
    x = self.drop3(x)
    gcn_outputs.append(x)

    x = self.gcn4(x, normalized_adj)
    x = self.batch_norm4(x)
    x = self.relu4(x)
    x = self.drop4(x)

    x = self.gcn5(x, normalized_adj)
    x = self.batch_norm5(x)
    x = self.relu5(x)
    x = self.drop5(x)
    gcn_outputs.append(x)

    # -------- Jumping Knowledge Aggregation --------
    x = self.max_pooling(gcn_outputs)
    # Alternatives:
    # x = self.mixture_outputs(gcn_outputs)
    # x = self.lstm_attention(gcn_outputs)

    self.embeddings.append(x)  # Save final node embeddings

    # -------- Classification Head --------
    x = x[self.node_data.where(self.node_data['node_status'] == node_status).dropna().index]

    x = self.linear6(x)
    x = self.batch_norm6(x)
    x = self.relu6(x)
    x = self.drop6(x)

    x = self.linear7(x)
    x = self.softmax(x)
    return x