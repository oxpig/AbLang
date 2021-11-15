import json
import torch

class ABtokenizer():
    """
    Tokenizer for proteins. Both aa to token and token to aa.
    """
    
    def __init__(self, vocab_dir):
        self.set_vocabs(vocab_dir)
        self.pad_token = self.vocab_to_token['-']
        
    def __call__(self, sequenceList, encode=True, pad=False, device='cpu'):
        #assert isinstance(sequenceList, list)
        
        if encode: 
            data = [self.encode(seq, device=device) for seq in sequenceList]
            if pad: return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=self.pad_token)
            else: return data
        
        else: return [self.decode(token) for token in sequenceList]
    
    def set_vocabs(self, vocab_dir):
        with open(vocab_dir, encoding="utf-8") as vocab_handle:
            self.vocab_to_token=json.load(vocab_handle)
            
        self.vocab_to_aa = {v: k for k, v in self.vocab_to_token.items()}
     
    def encode(self, sequence, device='cpu'):
        
        encoded = [self.vocab_to_token["<"]]+[self.vocab_to_token[resn] for resn in sequence]+[self.vocab_to_token[">"]]
        
        return torch.tensor(encoded, dtype=torch.long, device=device)
        # Start and Stop token should probably not be added here, but instead earlier
    
    def decode(self, seqtokens):
        
        if torch.is_tensor(seqtokens): seqtokens = seqtokens.cpu().numpy()

        return ''.join([self.vocab_to_aa[token] for token in seqtokens])
    

    
