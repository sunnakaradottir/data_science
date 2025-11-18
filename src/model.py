import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define size variables
num_features = 30  # number of input features (columns) in the dataset

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=num_features, hidden_units=64, latent_features=2, num_layers=1):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.

        def init_encoder_decoder(in_features, out_features, hidden_units, num_layers):
            '''
            Usage: init_encoder_decoder(30, 64, 2, 2)
            Goal: create an encoder / decoder with a dynamic depth

            Input: All features are of type int.
            Output: returns an nn.sequential model for either encoder or decoder
            '''

            # we can use the same function of encoder and decoder
            layers = []

            # First layer
            layers.append(nn.Linear(in_features, hidden_units)),
            layers.append(nn.ReLU())

            for _ in range(num_layers-1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU())

            # Final layer
            layers.append(nn.Linear(hidden_units, out_features))

            return nn.Sequential(*layers)

        self.encoder = init_encoder_decoder(in_dim, latent_features, hidden_units, num_layers)
        self.decoder = init_encoder_decoder(latent_features, in_dim, hidden_units, num_layers)

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