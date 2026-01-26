import os
import re
import subprocess
import csv
from utils.evaluation.docking_vina_modified import VinaDock
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread lock for safe file writing
file_lock = threading.Lock()

def main(ligand_pdbqt, protein_pdbqt, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ligand_id = os.path.splitext(os.path.basename(ligand_pdbqt))[0]
    # Extract the numeric part from ligand_id (e.g., '722' from '722_vins_output')
    match = re.search(r'^(\d+)', ligand_id)
    if match:
        ligand_id = match.group(1)
    else:
        print(f"Could not extract number from {ligand_id}, skipping")
        with file_lock:
            with open(os.path.join(output_dir, 'error_log.txt'), 'a') as log_file:
                log_file.write(f"Could not extract number from {ligand_id}\n")
        return
    
    # Check if ligand_id already exists in docking_results.csv
    results_file = os.path.join(output_dir, 'docking_results.csv')
    with file_lock:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                existing_ids = [row[0] for row in reader]
                if ligand_id in existing_ids:
                    print(f'Skipping {ligand_id} as it already exists in docking results')
                    return
    
    try:
        # Validate PDBQT file
        with open(ligand_pdbqt, 'r') as f:
            content = f.read()
            root_count = len(re.findall(r'^\s*ROOT', content, flags=re.MULTILINE))
            print(f'root_count for {ligand_id}: {root_count}')
            if root_count > 1:
                raise ValueError(f"Invalid PDBQT file {ligand_pdbqt}: Found {root_count} ROOT tags, expected exactly 1")
            if 'ENDROOT' not in content:
                raise ValueError(f"Invalid PDBQT file {ligand_pdbqt}: Missing ENDROOT tag")
        
        dock = VinaDock(ligand_pdbqt, protein_pdbqt)
        pocket_center, box_size = dock._max_min_pdb(protein_pdbqt, buffer=12)
        dock.pocket_center, dock.box_size = pocket_center, box_size
        score, pose = dock.dock(score_func='vina', mode='dock', exhaustiveness=16, save_pose=True)
        pdbqt_path = os.path.join(output_dir, 'pdbqt', f"{ligand_id}_vins_output.pdbqt")
        with file_lock:
            with open(pdbqt_path, 'a') as f:
                f.write(f"MODEL {ligand_id}\n{pose}ENDMDL\n")
            with open(os.path.join(output_dir, 'docking_results.csv'), 'a', newline='') as f:
                writer = csv.writer(f)
                if os.path.getsize(os.path.join(output_dir, 'docking_results.csv')) == 0:
                    writer.writerow(['ligand_id', 'score', 'pdbqt_path'])
                writer.writerow([ligand_id, score, pdbqt_path])
        print(f'score for {ligand_id}: {score}, pose: {pdbqt_path}')
    except Exception as e:
        print(f"Error processing ligand {ligand_id}: {str(e)}")
        with file_lock:
            with open(os.path.join(output_dir, 'error_log.txt'), 'a') as log_file:
                log_file.write(f"Ligand {ligand_id} ({ligand_pdbqt}): {str(e)}\n")
        return

if __name__ == "__main__":
    sdf_folder = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow/pocketflow_2'
    protein_pdbqt = '/blue/lic/huangzihang/repos/common_utils/vina/TEAD38P0M.pdbqt'
    output_dir = '/blue/lic/huangzihang/repos/elion/src/elion/vsdb/fine_tune_db/pocketflow'
    os.makedirs(output_dir, exist_ok=True)

    # Maximum number of worker threads
    max_workers = 64  # Adjust this based on your system's capabilities

    # Collect all tasks
    tasks = []
    for root, dirs, files in os.walk(sdf_folder):
        for file_name in files:
            match = re.search(r"No_(\d+)\.sdf", file_name)
            if match:
                number = match.group(1)
                sdf_path = f'{sdf_folder}/No_{number}.sdf'
                os.makedirs(os.path.join(output_dir, 'mol2'), exist_ok=True)
                mol2_path = os.path.join(output_dir, 'mol2', f'{number}.mol2')
                os.makedirs(os.path.join(output_dir, 'pdbqt'), exist_ok=True)
                pdbqt_path = os.path.join(output_dir, 'pdbqt', f'{number}.pdbqt')
                print(f'Preparing {sdf_path}')
                
                # Convert SDF to MOL2
                result = subprocess.run(['obabel', sdf_path, '-O', mol2_path], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error converting {sdf_path} to MOL2: {result.stderr}")
                    with file_lock:
                        with open(os.path.join(output_dir, 'error_log.txt'), 'a') as log_file:
                            log_file.write(f"SDF to MOL2 conversion failed for {sdf_path}: {result.stderr}\n")
                    continue
                
                # Convert MOL2 to PDBQT
                result = subprocess.run(['obabel', '-imol2', mol2_path, '-opdbqt', '-O', pdbqt_path, '-xh'], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error converting {mol2_path} to PDBQT: {result.stderr}")
                    with file_lock:
                        with open(os.path.join(output_dir, 'error_log.txt'), 'a') as log_file:
                            log_file.write(f"MOL2 to PDBQT conversion failed for {mol2_path}: {result.stderr}\n")
                    continue
                
                # Store task parameters for docking
                tasks.append((pdbqt_path, protein_pdbqt, output_dir))
    
    # Execute docking tasks with a thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: main(*args), tasks)
    
    print(f"All docking tasks completed with {max_workers} worker threads.")