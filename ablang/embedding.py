import torch


class AbEmbeddings(torch.nn.Module):
    """
    Residue embedding and Positional embedding
    """
    
    def __init__(self, hparams):
        super().__init__()
        self.pad_token_id = hparams.pad_token_id
        
        self.AAEmbeddings = torch.nn.Embedding(hparams.vocab_size, hparams.hidden_size, padding_idx=self.pad_token_id)
        self.PositionEmbeddings = torch.nn.Embedding(hparams.max_position_embeddings, hparams.hidden_size, padding_idx=0) # here padding_idx is always 0
        
        self.LayerNorm = torch.nn.LayerNorm(hparams.hidden_size, eps=hparams.layer_norm_eps)
        self.Dropout = torch.nn.Dropout(hparams.hidden_dropout_prob)

    def forward(self, src):
        
        inputs_embeds = self.AAEmbeddings(src)
        
        position_ids = self.create_position_ids_from_input_ids(src, self.pad_token_id)   
        position_embeddings = self.PositionEmbeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return self.Dropout(self.LayerNorm(embeddings))
        
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Replace non-padding symbols with their position numbers. Padding idx will get position 0, which will be ignored later on.
        """
        mask = input_ids.ne(padding_idx).int()
        
        return torch.cumsum(mask, dim=1).long() * mask