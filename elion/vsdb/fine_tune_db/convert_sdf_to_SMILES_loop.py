import os
import csv
from rdkit import Chem

# File paths
sdf_folder = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow/pocketflow_1'
csv_file = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow/docking_results_1.csv'
temp_csv = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow/docking_results_temp.csv'
output_csv = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow/docking_results_updated.csv'

# Step 1: Read the CSV, generate SMILES, and write to temporary CSV
updated_rows = []
with open(csv_file, 'r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ['smiles']  # Add 'smiles' column
    for row in reader:
        record = {key: row[key] for key in row}
        sdf_path = os.path.join(sdf_folder, f"No_{row['ligand_id']}.sdf")
        print(f"sdf_path: {sdf_path}")

        # Read the molecule from SDF file
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        mol = next(supplier, None)  # Get the first molecule or None if empty

        # Generate SMILES or set to empty string if molecule fails to load
        if mol is None:
            print(f"Error: Could not load molecule from {sdf_path}")
            record['smiles'] = ''
        else:
            try:
                Chem.SanitizeMol(mol)  # Sanitize to ensure valid SMILES
                record['smiles'] = Chem.MolToSmiles(mol)
                print(f"SMILES: {record['smiles']}")
            except Exception as e:
                print(f"Error generating SMILES for {sdf_path}: {str(e)}")
                record['smiles'] = ''

        updated_rows.append(record)

# Write the temporary CSV with SMILES
with open(temp_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print(f"Temporary CSV with SMILES written to {temp_csv}")

# Step 2: Reformat the CSV to keep only smiles and score columns
reformatted_rows = []
with open(temp_csv, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Select and reorder columns: smiles first, score second
        new_row = {
            'smiles': row['smiles'],
            'score': row['score']
        }
        reformatted_rows.append(new_row)

# Write the final reformatted CSV
with open(output_csv, 'w', newline='') as f:
    fieldnames = ['smiles', 'score']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(reformatted_rows)

print(f"Reformatted CSV written to {output_csv}")