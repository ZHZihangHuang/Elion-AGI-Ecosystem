from properties.CHEMBERT.chembert_TS import chembert_model, SMILES_Dataset_from_file
from pathlib import Path
module_dir = Path(__file__).parent

class activity_predictor:
    
    def __init__(self, properties):
        
        self.base = properties['predictor']
        self.type = properties['predictor_type']

        if self.base == 'CHEM-BERT':
            self.model = chembert_model(properties['state'])
            
    def predict(self, dataset):
        return self.model.predict(dataset)
        
if __name__ == '__main__':
    import time
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    import argparse
    
    #-- Command line arguments
    parser = argparse.ArgumentParser(description='''Tests a CHEMBERT model''')

    parser.add_argument('-s', '--smiles_file',
                        help='Path to the input SMILES file. Must have 2 columns: SMILES, data')

    parser.add_argument('-m', '--model',
                        help='trained model',
                        default=f'{module_dir}/properties/activity/CHEMBERT/model/pretrained_model.pt')

    args = parser.parse_args()
    
    smiles_file   = Path(args.smiles_file)
    trained_model = Path(args.model)

    #---
    output_name = f'{smiles_file.stem}_{trained_model.stem}'
    properties = {'predictor':'CHEM-BERT',
                  'predictor_type':'regressor',
                  'state':trained_model}
    
    start_time = time.time()
    predictor = activity_predictor(properties)

    dataset = SMILES_Dataset_from_file(smiles_file)
    parent_ids = dataset.parent_ids.flatten()
    smiles = dataset.adj_dataset

    # Finally, do the predictions:
    predictions = predictor.predict(dataset)

    results = pd.DataFrame({"SMILES":smiles})
    results['Name'] = parent_ids
    results['Labels'] = predictions
    # if len(labels) == len(predictions):
    #     results['Label'] = labels
    #     error = predictions - labels
    #     rmse = np.linalg.norm(error) / np.sqrt(len(labels))
    #     r2 = (np.corrcoef(predictions,labels)[0,1])**2
    #     print(f"{rmse = :5.2f}, {r2 = :5.2f}")

    results.to_csv(f'{output_name}.csv', float_format='%.2f', index=None)

    elapsed = time.time() - start_time
    print(F"ELAPSED TIME: {elapsed:.5f} seconds")
