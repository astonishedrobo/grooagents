import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from .model import VAE
from .utils import EarlyStopping
import os
from tqdm import tqdm
import pandas as pd
from .preprocess import load_and_preprocess_data
from .model import VAE, MaskedBetaVAE

def train_vae(model, train_dataloader, val_dataloader, epochs, learning_rate, beta, device, save_path, patience=None, grad_clip=1.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path) if patience else None

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs} (Train)")
        
        for batch_idx, data in progress_bar:
            x = data[0].to(device)  # Correctly access the tensor within the list
            optimizer.zero_grad()
            
            try:
                recon_batch, mu, logvar = model(x)
                loss = model.loss_function(recon_batch, x, mu, logvar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
            except Exception as e:
                print(f"Exception during training in batch {batch_idx}: {e}")
                raise e

        avg_train_loss = train_loss / len(train_dataloader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                x = data[0].to(device)  # Correctly access the tensor
                recon_batch, mu, logvar = model(x)
                loss = model.loss_function(recon_batch, x, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader.dataset)
        print(f"====> Epoch: {epoch+1} Average train loss: {avg_train_loss:.4f}, Average validation loss: {avg_val_loss:.4f}")

        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    if not early_stopping:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

class EmbeddingsTrainer:
    def __init__(
        self,
        data_file,
        save_model_path,
        input_dim=111,
        hidden_dims=[1024, 512, 256, 128],
        latent_dim=50,
        beta=2.0,
        learning_rate=1e-3,
        batch_size=64,
        epochs=200,
        patience=20,
        grad_clip=1.0,
        train_ratio=0.8,
        continuous_cols=None,
        categorical_cols=None,
    ):
        # Configuration parameters
        self.data_file = data_file
        self.save_model_path = save_model_path
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.train_ratio = train_ratio
        
        # Set columns if not provided
        if continuous_cols is None:
            self.continuous_cols = list(pd.read_csv(data_file).columns)
        else:
            self.continuous_cols = continuous_cols
            
        self.categorical_cols = [] if categorical_cols is None else categorical_cols
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_data(self):
        """Load and prepare the data for training"""
        # Load and preprocess data
        df = pd.read_csv(self.data_file)
        
        # Convert to tensor
        X = torch.tensor(df.values, dtype=torch.float32)
        
        # Create datasets
        dataset = TensorDataset(X)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        return self.train_dataloader, self.val_dataloader
    
    def initialize_model(self):
        """Initialize the VAE model"""
        categorical_dims = {}
        self.model = MaskedBetaVAE(
            self.input_dim, 
            self.hidden_dims, 
            self.latent_dim, 
            self.beta, 
            categorical_dims
        )
        return self.model
    
    def train(self):
        """Train the VAE model"""
        print(f"Using device: {self.device}")
        
        # Prepare data if not already done
        if not hasattr(self, 'train_dataloader'):
            self.prepare_data()
            
        # Initialize model if not already done
        if not hasattr(self, 'model'):
            self.initialize_model()
            
        # Train the model
        train_vae(
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.epochs,
            self.learning_rate,
            self.beta,
            self.device,
            self.save_model_path,
            self.patience,
            self.grad_clip
        )
        
        print("Training complete!")
        return self.model