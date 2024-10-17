#!/usr/bin/env python
# coding: utf-8

# # **AbLang Examples**
#
# AbLang is a RoBERTa inspired language model trained on antibody sequences. The following is a set of possible use cases of AbLang.

# In[1]:
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import ablang

# Print module path:

# In[2]:


heavy_ablang = ablang.pretrained("heavy")
heavy_ablang.freeze()


# --------------
# ## **AbLang building blocks**
#
# For easy use we have build the AbLang module (see below), however; for incorporating AbLang into personal codebases it might be more convenient to use the individual building blocks.

# #### AbLang tokenizer

# In[3]:


seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]

print("-" * 100)
print("Input sequences:")
for seq in seqs:
    print("-", seq)

tokens = heavy_ablang.tokenizer(seqs, pad=True)

print("-" * 100)
print("Tokens:")
print(tokens.shape)
print(tokens)

# #### AbLang encoder (AbRep)

rescodings = heavy_ablang.AbRep(tokens)

print("-" * 100)
print("Res-codings:")
print(rescodings)

# #### AbLang full model (AbRep+AbHead)

# In[5]:

likelihoods = heavy_ablang.AbLang(tokens)

print("-" * 100)
print("Likelihoods:")
print(likelihoods)


# -----
# ## **AbLang module: Res-codings**
#
# The res-codings are the 768 values for each residue, describing both a residue's individual properties (e.g. size, hydrophobicity, etc.) and properties in relation to the rest of the sequence (e.g. secondary structure, position, etc.).
#
# To calculate the res-codings, you can use the mode "rescoding" as seen below.

# In[6]:


seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]

rescodings = heavy_ablang(seqs, mode="rescoding")

print("-" * 100)
print("The output shape of a single sequence:", rescodings[0].shape)
print("This shape is different for each sequence, depending on their length.")
print("-" * 100)
print(rescodings)


# ----
# An additional feature, is the ability to align the rescodings. This can be done by setting the parameter align to "True".
#
# Alignment is done by numbering with anarci and then aligning sequences to all unique numberings found in input antibody sequences.
#
# **NB:** You need to install anarci and pandas for this feature.

# In[7]:

try:
    import anarci
except ImportError:
    print("Please install anarci to use the alignment feature.")
    sys.exit()

seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]

rescodings = heavy_ablang(seqs, mode="rescoding", align=True)

print("-" * 100)
print(
    "The output shape for the aligned sequences ('aligned_embeds'):",
    rescodings[0].aligned_embeds.shape,
)
print(
    "This output also includes this numberings ('number_alignment') used for this set of sequences."
)
print("-" * 100)
print(rescodings[0].aligned_embeds)
print(rescodings[0].number_alignment)


# ---------
# ## **AbLang module: Seq-codings**
#
# Seq-codings are a set of 768 values for each sequences, derived from averaging across the res-codings. Seq-codings allow one to avoid sequence alignments, as every antibody sequence, regardless of their length, will be represented with 768 values.

# In[8]:


seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]

seqcodings = heavy_ablang(seqs, mode="seqcoding")
print("-" * 100)
print("The output shape of the seq-codings:", seqcodings.shape)
print("-" * 100)

print(seqcodings)


# -----
# ## **AbLang module: Residue likelihood**
#
# Res- and seq-codings are both derived from the representations created by AbRep. Another interesting representation are the likelihoods created by AbHead. These values are the likelihoods of each amino acids at each position in the sequence. These can be used to explore which amino acids are most likely to be mutated into and thereby explore the mutational space.
#
# **NB:** Currently, the likelihoods includes the start and end tokens and padding.

# In[9]:


seqs = [
    "EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVLGWGSMDVWGQGTTVTVSS",
]

likelihoods = heavy_ablang(seqs, mode="likelihood")
print("-" * 100)
print("The output shape with paddings still there:", likelihoods.shape)
print("-" * 100)
print(likelihoods)


# ### The corresponding amino acids for each likelihood
#
# For each position the likelihood for each of the 20 amino acids are returned. The amino acid order can be found by looking at the ablang vocabulary. For this output the likelihoods for '<', '-', '>' and '\*' have been removed.

# In[10]:


ablang_vocab = heavy_ablang.tokenizer.vocab_to_aa
ablang_vocab


# -----
# ## **AbLang module: Antibody sequence restoration**
#
# In some cases, an antibody sequence is missing some residues. This could be derived from sequencing errors or limitations of current sequencing methods. To solve this AbLang has the "restore" mode, as seen below, which picks the amino acid with the highest likelihood for residues marked with an asterisk (*).

# In[11]:


seqs = [
    "EV*LVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "*************PGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNK*YADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTL*****",
]

print("-" * 100)
print("Restoration of masked residues.")
print("-" * 100)
print(heavy_ablang(seqs, mode="restore"))


# In cases where sequences are missing unknown lengths at the ends, we can use the "align=True" argument.

# In[12]:


seqs = [
    "EV*LVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS",
    "PGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNK*YADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTL",
]

print("-" * 100)
print("Restoration of masked residues and unknown missing end lengths.")
print("-" * 100)
print(heavy_ablang(seqs, mode="restore", align=True))
