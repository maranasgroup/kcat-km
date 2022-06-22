import numpy as np
import os
import argparse
import random
from datetime import datetime
import json
import time
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import pandas as pd
import tensorflow as tf

from models import Conv1DPredictor

def featurize_sequence(seq, pad_length=1024):
    seq_rdic = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}
    code = []
    for aa in seq:
        try:
            code.append(seq_dic[aa])
        except KeyError:
            print(f'Invalid amino-acid code : {aa} found in input sequence! Please check for valid input enzyme Sequence!')
            return None
            
    return np.pad(code, (0,pad_length-len(code)), mode='constant')

def featurize_substrate(smiles, fptype="fp_morgan3"):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except: 
        print('Unable to make rdkit mol object from given SMILES! Please check for valid input substrate SMILES!')
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=3,nBits=2048)
    
    return np.array(fp)

def get_features_from_input_file(input_file):
    print('-'*50)
    print('         Reading inputs...        ')
    features = {}
    input_df = pd.read_csv(input_file)
    for tag, seq, smiles in zip(input_df.tag, input_df.sequence, input_df.smiles):
        sequence_feat = featurize_sequence(seq)
        substrate_feat = featurize_substrate(smiles)
        features[tag] = (substrate_feat, sequence_feat)
    return features

def write_outputs_to_file(outputs, output_file, prediction_type):
    print('-'*50)
    print('         Writing outputs...        ')
    f = open(output_file, 'w')
    if prediction_type=='KCAT': f.write('tag,log[kcat(s^-1)]\n')
    elif prediction_type=='KM': f.write('tag,log[KM(M)]\n')
    
    for tag, pred in outputs.items():
        f.write(f'{tag},{pred}\n ')
    f.close()
    print(f'         Wrote predictions to {output_file}        ')
    print('='*50)
    
def main(args):
    model = os.path.join(args.model_dir,'model.h5')
    
    config = json.load(open(os.path.join(args.model_dir,'config.json')))
    
    predictor = []
    print('-'*50)
    print('         Bulding Model...        ')
    predictor = Conv1DPredictor(config)
    predictor.load_model(model)
    
    features = get_features_from_input_file(args.input_file)
    outputs = {}
    for tag, (substrate_feat, sequence_feat) in tqdm(features.items()):
        print('-'*50)
        print('          Making predictions...        ')
        substrate_feat = tf.expand_dims(substrate_feat, 0, name=None)
        sequence_feat = tf.expand_dims(sequence_feat, 0, name=None)
        prediction = predictor.modelobj((substrate_feat, sequence_feat), training=False)
        outputs[tag] = float(prediction.numpy().flatten()[0])
    
    write_outputs_to_file(outputs, args.output_file, args.prediction_type)
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Make predictions for KCAT/KM")
    
    parser.add_argument("input_file", help="full path to input csv file", type=str)
    parser.add_argument("output_file", help="full path to output csv file", type=str)
    parser.add_argument("model_dir", help="full path to model dir", type=str)
    parser.add_argument("prediction_type", help="prediction type: KCAT/KM", type=str)
    
    args = parser.parse_args()
    assert(args.prediction_type in ['KCAT','KM'])
        
    main(args)