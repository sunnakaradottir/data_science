import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define size variables
num_features = 30  # number of input features (columns) in the dataset

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=num_features, hidden_units=64, latent_features=2):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=in_dim)
        )

    def forward(self, x): 
        
        z = self.encoder(x)
        
        x_hat = self.decoder(z)
        
        return {
            'z': z,
            'x_hat': x_hat
        }


# Choose the shape of the autoencoder
net = AutoEncoder(in_dim=num_features, hidden_units=64, latent_features=2).to(DEVICE)


print(net)

optimizer = optim.Adam(net.parameters(), lr=0.001)

criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction