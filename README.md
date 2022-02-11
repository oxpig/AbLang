
---

<div align="center">    
 
# AbLang: A language model for antibodies  

[![DOI:10.1101/2022.01.20.477061](http://img.shields.io/badge/DOI-10.1101/2022.01.20.477061-B31B1B.svg)](https://doi.org/10.1101/2022.01.20.477061)

</div>


General protein language models have been shown to summarise the semantics of protein sequences into representations that are useful for state-of-the-art predictive methods. However, for antibody specific problems, such as restoring residues lost due to sequencing errors, a model trained solely on antibodies may be more powerful. Language models require vast numbers of sequences for training and antibodies are one of the few protein types for which such volumes of data exist, for example in the Observed Antibody Space (OAS) database. Here, we introduce AbLang, a language model trained on the antibody sequences in the OAS database. We demonstrate the power of AbLang by using it to restore missing residues in antibody sequence data, a key issue with BCR-seq data, as seen with over 40% of OAS sequences missing the first 15 amino acids. AbLang restores the missing residues of antibody sequences better than using IMGT germlines or the general protein language model ESM-1b. Further, AbLang does not require knowledge of the germline of the antibody and is seven times faster than ESM-1b.

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


## Antibody sequence restoration

Restoration of antibody sequences can be done using the "restore" mode as seen below.

```{r, engine='python', count_lines}
import ablang

heavy_ablang = ablang.pretrained("heavy") # Use "light" if you are working with light chains
heavy_ablang.freeze()


seqs = [
    'EV*LVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS',
    '*************PGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNK*YADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTL*****',
]

heavy_ablang(seqs, mode='restore')

```

The output of the above is seen below.

```console
array(['EVQLVESGPGLVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS',
       'QVQLVESGGGVVQPGKSLRLSCVASGFTFSGYGMHWVRQAPGKGLEWIALIIYDESNKYYADSVKGRFTISRDNSKNTLYLQMSSLRAEDTAVFYCAKVKFYDPTAPNDYWGQGTLVTVSS'],
      dtype='<U121')
```
-----

### Citation   
```
@article{Olsen2022,
  title={AbLang: An antibody language model for completing antibody sequences},
  author={Tobias H. Olsen, Iain H. Moal and Charlotte M. Deane},
  journal={bioRxiv},
  doi={https://doi.org/10.1101/2022.01.20.477061},
  year={2022}
}
```  