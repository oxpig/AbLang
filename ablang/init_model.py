import os, json, argparse, string
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import anarci

from . import tokenizers, ablang


class init_model():
    """
    Initializes the model.    
    """
    
    def __init__(self, model_dir, random_init=False):
        super().__init__()
        
        with open(os.path.join(model_dir, 'hparams.json'), 'r', encoding='utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))    
        
        self.model_file = os.path.join(model_dir, 'amodel.pt')
        
        self.init_model = ablang.AbLang(self.hparams)
        self.AbLang = ablang.AbLang(self.hparams)
        
        if not random_init:
            self.AbLang.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        
        self.tokenizer = tokenizers.ABtokenizer(os.path.join(model_dir, 'vocab.json'))
        self.AbRep = self.AbLang.AbRep
        
    def freeze(self):
        self.AbLang.eval()
        
    def unfreeze(self):
        self.AbLang.train()
        
    def __call__(self, sequence, mode='sequence', align=False):
        """
        Mode: sequence, residues, reconstruct or mutations.
        """
        
        if isinstance(sequence, str): sequence = [sequence]
        
        aList = []
        splitedSize=50 # THIS IS BASED ON RAM AVAILABLE - HIGHER IS FASTER, BUT IT USES MORE RAM

        for sequence_part in [sequence[x:x+splitedSize] for x in range(0, len(sequence), splitedSize)]:
            
            if mode=='sequence': 
                aList = aList + [self.sequence(sequence_part)]
            
            elif mode=='reconstruct': 
                aList = aList + [self.reconstruct(sequence_part)]
                
            elif mode=='residues': 
                aList = aList + [self.residues(sequence_part, align=align)]
                
            elif mode=='mutations': 
                aList = aList + [self.mutations(sequence_part)]

            else:
                raise SyntaxError("Given mode doesn't exist.")
        
        if mode == 'residues':
            return aList
        
        return np.concatenate(aList)
    
    def sequence(self, seqs):
        """
        Predicts the embeddings per residues and then runs the res_to_seq function for each sequence to achieve a fixed length embedding per sequence.
        """
        
        tokens = self.tokenizer(seqs, pad=True)

        residue_states = self.AbRep(tokens).last_hidden_states
        
        residue_states = residue_states.detach().numpy()
        
        lens = np.vectorize(len)(seqs)
        
        lens = np.tile(lens.reshape(-1,1,1), (residue_states.shape[2], 1))
        
        seq_codings = np.apply_along_axis(res_to_seq, 2, np.c_[np.swapaxes(residue_states,1,2), lens])
        
        del lens
        del residue_states
        
        return seq_codings
    
    def reconstruct(self, seqs):
        """
        Reconstruct sequences
        """
        
        tokens = self.tokenizer(seqs, pad=True)
        
        predictions = self.AbLang(tokens)[:,:,1:21]

        predicted_tokens = torch.max(predictions, -1).indices + 1

        reconstructed_seqs = self.tokenizer(predicted_tokens, encode=False)
        
        return np.array([res_to_seq(seq, 'reconstruct') for seq in np.c_[reconstructed_seqs, np.vectorize(len)(seqs)]])
    
    def mutations(self, seqs):
        """
        Possible Mutations
        """
        
        tokens = self.tokenizer(seqs, pad=True)
        
        predictions = self.AbLang(tokens)[:,:,1:21]
        
        return predictions.detach().numpy()
    
    def residues(self, seqs, align=False):
           
        if align:
            anarci_out = anarci.run_anarci(pd.DataFrame(seqs).reset_index().values.tolist(), ncpu=7, scheme='imgt')
            number_alignment = get_number_alignment(anarci_out)
            
            seqs = np.array([''.join([i[1] for i in onarci[0][0]]).replace('-','') for onarci in anarci_out[1]])
            
            tokens = self.tokenizer(seqs, pad=True)
            residue_states = self.AbRep(tokens).last_hidden_states
            residue_states = residue_states.detach().numpy()
            
            residue_output = np.array([create_alignment(res_embed, oanarci, seq, number_alignment) for res_embed, oanarci, seq in zip(residue_states, anarci_out[1], seqs)])
            del residue_states
            del tokens
            
            return output(aligned_embeds=residue_output, number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values)
            
        else:
            
            tokens = self.tokenizer(seqs, pad=True)
            residue_states = self.AbRep(tokens).last_hidden_states
        
            residue_states = residue_states.detach().numpy()
            
            residue_output = [res_to_list(state, seq) for state, seq in zip(residue_states, seqs)]

            return residue_output

    
@dataclass
class output():
    """
    Dataclass used to store output.
    """

    aligned_embeds: None
    number_alignment: None
    
    
def res_to_list(state, seq):
    return state[1:1+len(seq)]

def res_to_seq(a, mode='mean'):
    """
    Function for how we go from n_values for each amino acid to n_values for each sequence.
    
    We leave out the start, end and padding tokens.
    """
    if mode=='sum':
        return a[1:(1+int(a[-1]))].sum()
    
    elif mode=='mean':
        return a[1:(1+int(a[-1]))].mean()
    
    elif mode=='reconstruct':
        
        return a[0][1:(1+int(a[-1]))]

def get_number_alignment(oanarci):
    """
    Creates a number alignment from the anarci results.
    """
    
    alist = []
    
    for aligned_seq in oanarci[1]:
        alist.append(pd.DataFrame(aligned_seq[0][0])[0])

    unsorted_alignment = pd.concat(alist).drop_duplicates()
    max_alignment = get_max_alignment()
    
    return max_alignment.merge(unsorted_alignment.to_frame(), left_on=0, right_on=0)

def get_max_alignment():
    """
    Create maximum possible alignment for sorting
    """

    sortlist = []

    for num in range(1, 128+1):

        if num==112:
            for char in string.ascii_uppercase[::-1]:
                sortlist.append([(num, char)])

            sortlist.append([(num,' ')])

        else:
            sortlist.append([(num,' ')])
            for char in string.ascii_uppercase:
                sortlist.append([(num, char)])
                
    return pd.DataFrame(sortlist)

def create_alignment(res_embeds, oanarci, seq, number_alignment):

    datadf = pd.DataFrame(oanarci[0][0])

    sequence_alignment = number_alignment.merge(datadf, how='left', on=0).fillna('-')[1]
        
    idxs = np.where(sequence_alignment.values == '-')[0]

    idxs = [idx-num for num, idx in enumerate(idxs)]

    aligned_embeds = pd.DataFrame(np.insert(res_embeds[1:1+len(seq)], idxs , 0, axis=0))

    return pd.concat([aligned_embeds, sequence_alignment], axis=1).values
