# PolyBayes: Uncertainty-Aware Glass Transition ($T_g$) Prediction for Polymers

A machine learning pipeline and interactive web dashboard for predicting the glass transition temperature (Tg) of polymers. This project leverages Morgan Fingerprints and GPU-accelerated Gaussian Process Regression to not only predict thermal properties but also quantify model uncertainty, making it highly suitable for active learning and Bayesian optimization workflows in polymer informatics.

## Features

* **Data Preprocessing Pipeline:** Robust automated conversion of polymer SMILES strings into 2048-bit Morgan Fingerprints using RDKit.
* **GPU-Accelerated Training:** Utilizes PyTorch and GPyTorch for exact Gaussian Process inference, dramatically reducing training time for complex chemical datasets compared to CPU-bound alternatives.
* **Bayesian Uncertainty Quantification:** Outputs both a predicted Tg and a standard deviation ($\sigma$), identifying sparse regions in the chemical space and guiding future data acquisition.
* **Interactive Streamlit Dashboard:** A user-friendly web interface that allows users to input SMILES strings, run real-time inference, and visualize the Bayesian predictive distribution via a dynamic bell curve.

## Technology Stack

* **Machine Learning:** PyTorch, GPyTorch, Scikit-Learn
* **Cheminformatics:** RDKit
* **Web Deployment:** Streamlit
* **Data Handling:** Pandas, NumPy, PyArrow/Fastparquet (for efficient `.parquet` storage)


## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/j-udge/PolyBayes.git](https://github.com/j-udge/PolyBayes.git)
   cd PolyBayes
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows use: venv\Scripts\activate
   # On Mac/Linux use: source venv/bin/activate
   ```

3. Install the dependencies:
   *Note: Ensure you have the correct version of PyTorch installed for your CUDA version if you intend to use GPU acceleration.*
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocessing Data (`preprocess.py`)
Before training, raw polymer data (Excel/CSV) must be converted into Morgan Fingerprints. This script handles NaN removal, SMILES validation via RDKit, and bit-vector generation. It outputs a high-performance `.parquet` file.

```bash
python src/preprocess.py
```

### 2. Training the Model (`model_form.py`)
The training script loads the fingerprints, pushes the tensors to the **GPU (CUDA)**, and trains a GPyTorch ExactGP model. It calculates evaluation metrics ($R^2$, $RMSE$) and saves the model state and training data into a deployment-ready `.pth` file.

```bash
python src/model_form.py
```

### 3. Running the Dashboard (`app.py`)
To launch the interactive web interface, use Streamlit. The app loads the trained model and provides a real-time interface for polymer property prediction and uncertainty visualization.

```bash
python -m streamlit run src/app.py
```

## 📊 Bayesian Optimization & Active Learning

This tool is designed to assist in chemical space exploration. When a user inputs a polymer SMILES string, the dashboard evaluates the prediction's uncertainty (σ). High uncertainty flags indicate that the polymer is out-of-distribution relative to the training data. Synthesizing or acquiring data for these specific high-uncertainty polymers provides the maximum information gain for subsequent model retraining.

## 📄 License
[MIT License](LICENSE)
