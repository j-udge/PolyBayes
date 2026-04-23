import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def process_smiles_to_morgan(input_csv, smiles_col, tg_col, output_file):
    print("Loading dataset")
    #Dropping NaN
    print("Droping NaN values..")
    df=pd.read_excel(input_csv)
    df_cleaned=df.dropna(subset=[tg_col])

    # Initialize lists to store our valid data
    valid_smiles = []
    valid_tgs = []
    fingerprints = []
    
    print(f"Processing {len(df_cleaned)} molecules..")
    
    # 2. Iterate through the dataset
    for index, row in df_cleaned.iterrows():
        smiles = row[smiles_col]
        tg = row[tg_col]
        
        # Convert SMILES to an RDKit Mol object
        mol = Chem.MolFromSmiles(str(smiles))
        
        if mol is not None:
            # 3. Generate the Morgan Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            
            # Convert the RDKit vector to a NumPy array so we can save it easily
            fp_array = np.zeros((2048,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, fp_array)
            
            # Store the valid data
            fingerprints.append(fp_array)
            valid_smiles.append(smiles)
            valid_tgs.append(tg)
        else:
            print(f"Warning: Could not parse SMILES at index {index}: {smiles}")

    # 4. Package the processed data into a new DataFrame
    print("Saving SMILES AND TG THAT WERE PROCESSED")
    processed_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Tg': valid_tgs
    })
    
    fp_matrix = np.array(fingerprints)
    
    # Generate the column names (bit_0, bit_1, ... bit_2047)
    bit_columns = [f'bit_{i}' for i in range(fp_matrix.shape[1])]
    
    # Turn the matrix into a DataFrame all at once
    fp_df = pd.DataFrame(fp_matrix, columns=bit_columns)
    
    # Concatenate the SMILES/Tg data with the fingerprint data side-by-side (axis=1)
    processed_df = pd.concat([processed_df, fp_df], axis=1)
    # -----------------------
        
    # 5. Save the final file
    processed_df.to_parquet(output_file, index=False)
    print(f"Successfully saved {len(processed_df)} valid records to {output_file}")

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\Lenovo\Downloads\experiment_polymer_data.xlsx"   
    SMILES_COLUMN_NAME = "PSMILES"         
    TG_COLUMN_NAME = "Tg_K"           
    OUTPUT_FILE = "processed_morgan_fp.parquet"
    process_smiles_to_morgan(INPUT_FILE, SMILES_COLUMN_NAME, TG_COLUMN_NAME, OUTPUT_FILE)

