import os, json, argparse, string, subprocess
from dataclasses import dataclass

import numpy as np
import torch
import requests

from . import tokenizers, model


class pretrained:
    """
    Initializes AbLang for heavy or light chains.    
    """
    
    def __init__(self, chain="heavy", model_folder="download", random_init=False):
        super().__init__()
        
        if model_folder == "download":
            # Download model to specific place - if already downloaded use it without downloading again
            model_folder = os.path.join(os.path.dirname(__file__), "model-weights-{}".format(chain))
            os.makedirs(model_folder, exist_ok = True)
            
            if not os.path.isfile(os.path.join(model_folder, "amodel.pt")):
                print("Downloading model ...")
                
                url = "http://opig.stats.ox.ac.uk/website/data/downloads/ablang-{}.tar.gz".format(chain)
                tmp_file = os.path.join(model_folder, "tmp.tar.gz")

                with open(tmp_file,'wb') as f: f.write(requests.get(url).content)
                
                subprocess.run(["tar", "-zxvf", tmp_file, "-C", model_folder], check = True) 
                
                os.remove(tmp_file)

        self.hparams_file = os.path.join(model_folder, 'hparams.json')
        self.model_file = os.path.join(model_folder, 'amodel.pt')
        
        with open(self.hparams_file, 'r', encoding='utf-8') as f:
            self.hparams = argparse.Namespace(**json.load(f))    

        self.AbLang = model.AbLang(self.hparams)
        
        if not random_init:
            self.AbLang.load_state_dict(torch.load(self.model_file, map_location=torch.device('cpu')))
        
        self.tokenizer = tokenizers.ABtokenizer(os.path.join(model_folder, 'vocab.json'))
        self.AbRep = self.AbLang.AbRep
        
    def freeze(self):
        self.AbLang.eval()
        
    def unfreeze(self):
        self.AbLang.train()
        
    def __call__(self, sequence, mode='seqcoding', align=False, splitSize=50):
        """
        Mode: sequence, residue, restore or likelihood.
        """
        if not mode in ['rescoding', 'seqcoding', 'restore', 'likelihood']:
            raise SyntaxError("Given mode doesn't exist.")
        
        if isinstance(sequence, str): sequence = [sequence]
        
        aList = []

        for sequence_part in [sequence[x:x+splitSize] for x in range(0, len(sequence), splitSize)]:
            
            aList.append(getattr(self, mode)(sequence_part, align))
        
        if mode == 'rescoding':
            if align==True:
                return aList
            
            return sum(aList, [])
        
        return np.concatenate(aList)
    
    def seqcoding(self, seqs, align=False):
        """
        Sequence specific representations
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
    
    def restore(self, seqs, align=False):
        """
        Restore sequences
        """
        
        tokens = self.tokenizer(seqs, pad=True)
        
        predictions = self.AbLang(tokens)[:,:,1:21]

        predicted_tokens = torch.max(predictions, -1).indices + 1

        restored_tokens = torch.where(tokens==23, predicted_tokens, tokens)

        restored_seqs = self.tokenizer(restored_tokens, encode=False)
        
        return np.array([res_to_seq(seq, 'reconstruct') for seq in np.c_[restored_seqs, np.vectorize(len)(seqs)]])
    
    def likelihood(self, seqs, align=False):
        """
        Possible Mutations
        """
        
        tokens = self.tokenizer(seqs, pad=True)
        
        predictions = self.AbLang(tokens)[:,:,1:21]
        
        return predictions.detach().numpy()
    
    def rescoding(self, seqs, align=False):
        """
        Residue specific representations.
        """
           
        if align:
            
            import pandas as pd
            import anarci
            
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
    
    import pandas as pd
    
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
    
    import pandas as pd

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
    
    import pandas as pd

    datadf = pd.DataFrame(oanarci[0][0])

    sequence_alignment = number_alignment.merge(datadf, how='left', on=0).fillna('-')[1]
        
    idxs = np.where(sequence_alignment.values == '-')[0]

    idxs = [idx-num for num, idx in enumerate(idxs)]

    aligned_embeds = pd.DataFrame(np.insert(res_embeds[1:1+len(seq)], idxs , 0, axis=0))

    return pd.concat([aligned_embeds, sequence_alignment], axis=1).values
