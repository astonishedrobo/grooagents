import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .model import MaskedBetaVAE
from .preprocess import load_and_preprocess_data

class EmbeddingsInference:
    def __init__(
        self,
        data_file,
        model_path,
        variables,
        input_dim=111,
        hidden_dims=[1024, 512, 256, 128],
        latent_dim=50,
        beta=2.0,
        continuous_cols=None,
        categorical_cols=None,
        scaler_type="standard",
    ):
        # Configuration parameters
        self.data_file = data_file
        self.model_path = model_path
        self.variables = variables
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.beta = beta
        self.scaler_type = scaler_type

        if continuous_cols is None:
            self.continuous_cols = list(pd.read_csv(data_file).columns)
        else:
            self.continuous_cols = continuous_cols
            
        self.categorical_cols = [] if categorical_cols is None else categorical_cols

        
        # Initialize placeholders for data
        self.df = None
        self.df_all = None
        self.X = None
        self.embeddings = None
        self.model = None
        
    def prepare_data(self):
        """Load and prepare the data for inference"""
        # Load and preprocess data
        self.df = load_and_preprocess_data(
            self.data_file, 
            self.continuous_cols, 
            self.categorical_cols, 
            scaler_type=self.scaler_type
        )

        # Convert to tensor
        self.X = torch.tensor(self.df.values, dtype=torch.float32)
        
    def load_model(self):
        """Load the trained VAE model"""
        print(f"Loading model from: {self.model_path}")
        
        # Initialize model
        self.model = MaskedBetaVAE(
            self.input_dim, 
            self.hidden_dims, 
            self.latent_dim, 
            self.beta, 
            self.categorical_cols
        )
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(
            self.model_path, 
            map_location=torch.device('cpu')
        ))
        
        # Set model to evaluation mode
        self.model.eval()
        
        return self.model
    
    def generate_embeddings(self):
        """Generate embeddings using the VAE model"""
        # Load data if not already done
        if self.X is None:
            self.prepare_data()
            
        # Load model if not already done
        if self.model is None:
            self.load_model()
            
        # Generate embeddings
        print("Generating Embeddings...")
        with torch.no_grad():
            self.embeddings = self.model.encode(self.X)[0]  # Get the mean (μ) as embeddings
        
        print("Embeddings generated.")
        
        return self.embeddings
    
    def inference(self):
        """Return the results including dataframes and embeddings"""
        # Generate embeddings if not already done
        if self.embeddings is None:
            self.generate_embeddings()
            
        # Convert embeddings to numpy for easier use
        embeddings_np = self.embeddings.numpy()
        
        # # Return all relevant data
        # results = {
        #     'embeddings': embeddings_np,
        #     'df': self.df,
        #     'df_all': self.df_all
        # }
        
        return embeddings_np