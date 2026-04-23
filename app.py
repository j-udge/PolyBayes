import streamlit as st
import torch
import gpytorch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Page Config ---
st.set_page_config(page_title="Polymer Tg Predictor", layout="wide")

# --- Define the Model Class (Must match training script exactly) ---
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- Load Model ---
@st.cache_resource
def load_model():
    # Force loading onto CPU for stable Streamlit web hosting
    device = torch.device('cpu')
    
    # Load the PyTorch payload
    payload = torch.load(r'C:\Tg_Bayesian_Optimization\src\gpytorch_tg_model.pth', map_location=device)
    
    # Extract training data needed to initialize the ExactGP
    train_x = payload['train_data']['x']
    train_y = payload['train_data']['y']
    
    # Initialize the architecture
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Load the trained weights
    model.load_state_dict(payload['model_state'])
    likelihood.load_state_dict(payload['likelihood_state'])
    
    # Set to evaluation mode (Crucial for PyTorch inference!)
    model.eval()
    likelihood.eval()
    
    return model, likelihood, payload['metrics'], device

# Load everything in
model, likelihood, metrics, device = load_model()

# --- Dashboard Header ---
st.title("🧪 Polymer Glass Transition ($T_g$) Predictor")
st.markdown("Powered by GPyTorch & Morgan Fingerprints")

# --- Display Metrics ---
st.header("Model Evaluation")
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{metrics['r2']:.3f}")
col2.metric("RMSE", f"{metrics['rmse']:.2f} K")
col3.metric("Training Samples", metrics['train_size'])
col4.metric("Test Samples", metrics['test_size'])

st.divider()

# --- Prediction Interface ---
st.header("Predict New Polymer")
smiles_input = st.text_input("Enter Polymer SMILES String:", placeholder="e.g., CC(C)(C)c1ccc(O)cc1")

if smiles_input:
    # 1. Convert SMILES to RDKit Molecule
    mol = Chem.MolFromSmiles(smiles_input)
    
    if mol:
        # 2. Generate Morgan Fingerprint 
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        
        # Convert to a 2D numpy array, then to a PyTorch Tensor
        fp_array = np.array(fp, dtype=np.float32).reshape(1, -1)
        fp_tensor = torch.tensor(fp_array).to(device)
        
        # 3. Run Inference with GPyTorch
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(fp_tensor))
            
            # Extract the mean and standard deviation
            pred_tg = observed_pred.mean.item()
            sigma = observed_pred.stddev.item()
        
        # 4. Display Results
        st.subheader("Prediction Results")
        res_col1, res_col2 = st.columns(2)

        # ... (Previous code where res_col1 and res_col2 are defined) ...
        res_col1.metric("Predicted Tg", f"{pred_tg:.2f} K")
        res_col2.metric("Uncertainty (Sigma)", f"± {sigma:.2f} K")
        
        # ---------------------------------------------------------
        # NEW CODE: Generate the Bayesian Distribution Graph
        # ---------------------------------------------------------
        st.subheader("Bayesian Predictive Distribution")
        st.markdown("This curve represents the model's confidence. A tighter peak means higher confidence.")
        
        # 1. Create an X-axis spanning 4 standard deviations on either side of the prediction
        x_axis = np.linspace(pred_tg - 4*sigma, pred_tg + 4*sigma, 200)
        
        # 2. Calculate the Gaussian Probability Density Function (PDF) manually 
        # (This is the math behind the bell curve!)
        y_axis = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - pred_tg) / sigma)**2)
        
        # 3. Package it into a Pandas DataFrame for Streamlit
        # We set the index to x_axis so Streamlit uses it as the bottom label
        chart_df = pd.DataFrame(
            {"Probability Density": y_axis}, 
            index=x_axis
        )
        
        # 4. Draw a slick shaded area chart
        st.area_chart(chart_df)
        # ---------------------------------------------------------

        # 5. Active Learning / Data Acquisition Insight
        st.subheader("Model Confidence Analysis")
        # ... (Rest of your code) ...
        
        # Adjust this threshold based on your typical RMSE
        uncertainty_threshold = 15.0 
        
        if sigma > uncertainty_threshold:
            st.warning(
                f"⚠️ **High Uncertainty!** The model is not confident about this prediction. "
                f"This polymer is in a sparse region of your chemical space. "
                f"**Recommendation:** Synthesize or gather more training data for polymers structurally similar to this one to improve the model."
            )
        else:
            st.success(
                "✅ **High Confidence.** This polymer falls within a well-explored area of your training data."
            )
            
    else:
        st.error("Invalid SMILES string. Please check the structure and try again.")