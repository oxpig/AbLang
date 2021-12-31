# **AbLang: A language model for antibodies**

AbLang allows users to explore 

-----------

# Install AbLang

AbLang is freely available and can be installed with pip.

~~~.sh
    pip install ablang
~~~

or directly from github.

~~~.sh
    pip install -U git+https://github.com/oxpig/AbLang.git
~~~

----------

# AbLang use cases

**A Jupyter notebook** showing the different use cases of AbLang can be found [here](https://github.com/TobiasHeOl/AbLang/tree/main/examples). 


Currently, AbLang can be used to generate three different representations/encodings for antibody sequences. 

1. **Res-codings:** These encodings are 768 values for each residue, useful for residue specific predictions.

2. **Seq-codings:** These encodings are 768 values for each sequence, useful for sequence specific predictions. The same length of encodings for each sequence, means these encodings also removes the need to align antibody sequences.

3. **Res-likelihoods:** These encodings are the likelihoods of each amino acid at each position in a given antibody sequence, useful for exploring possible mutations.

These representations can be used for a plethora of antibody design applications. As an example, we have used the res-likelihoods from AbLang to restore missing residues in antibody sequences due either to sequencing errors, such as ambiguous bases, or the limitations of the sequencing techniques used.

Restoration of antibody sequences can be done like this:

`python

import ablang

heavy_ablang = ablang.pretrained("heavy") # Use "light" if you are working with light chains
heavy_ablang.freeze()


seqs = [
    'EV*LVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS',
    '*************PGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNK*YADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVT***',
]

heavy_ablang(seqs, mode='restore')

`

```console
array(['EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS',
       'QVQLVESGGGVVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS'],
      dtype='<U121')
```



# Citing this work

AbLang is based on a paper in preparation.