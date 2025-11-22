import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_features = 30  # number of input features (columns) in the dataset


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=num_features, hidden_units=64, latent_features=2, num_layers=1):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.
        
        def init_encoder_decoder(in_features, out_features, hidden_units, num_layers):
            '''
            Create an encoder / decoder with a dynamic number of layers

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

    @staticmethod
    def _parse_config_file(config_path):
        """Parse a model config txt file into a dictionary."""
        config = {}
        with open(config_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    config[key] = value
        return config

    @classmethod
    def from_pretrained(cls, cluster_id, models_dir='models', device='cpu'):
        """
        Load a pretrained AutoEncoder for a specific cluster.

        Args:
            cluster_id: Which cluster's model to load (0-3)
            models_dir: Path to models directory
            device: 'cpu' or 'cuda'

        Returns:
            AutoEncoder with loaded weights, in eval mode
        """
        base_path = f'{models_dir}/ae_cluster_{cluster_id}'
        weights_path = f'{base_path}/ae_cluster_{cluster_id}.pt'
        config_path = f'{base_path}/ae_cluster_{cluster_id}_config.txt'

        config = cls._parse_config_file(config_path)

        # Create model with correct architecture
        model = cls(
            in_dim=num_features,
            hidden_units=config['hidden_dim'],
            latent_features=config['latent'],
            num_layers=config['num_layers']
        )

        # Load weights
        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights)
        model.eval()

        return model