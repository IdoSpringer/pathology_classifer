from TCR_Autoencoder.tcr_autoencoder import PaddingAutoencoder
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class PathologyClassifier(nn.Module):
    def __init__(self, embedding_dim, device, max_len, input_dim, encoding_dim, output_dim,
                 batch_size, ae_file, train_ae=True):
        super(PathologyClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.input_dim = input_dim
        self.batch_size = batch_size
        # TCR Autoencoder
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        checkpoint = torch.load(ae_file)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()
        self.mlp = nn.Sequential(nn.Linear(encoding_dim, 50),
                                 nn.Tanh(),
                                 nn.Linear(50, output_dim),
                                 nn.Softmax(dim=1))

    def forward(self, padded_tcrs):
        # TCR Encoder:
        # Embedding
        concat = padded_tcrs.view(self.batch_size, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)
        # MLP Classifier
        mlp_output = self.mlp(encoded_tcrs)
        return mlp_output

    pass


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(MLP, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, 50),
                                 nn.Tanh(),
                                 nn.Linear(50, output_dim),
                                 nn.Softmax())

    def forward(self, encoded_tcrs):
        # MLP Classifier
        output = self.mlp(encoded_tcrs)
        return output

    pass
