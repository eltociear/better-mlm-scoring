# A Better Way to Do Masked Language Model Scoring
Code accompanying the paper [A Better Way to Do Masked Language Model Scoring](LINK MISSING) by Carina Kauf & Anna Ivanova.

## Paper abstract
Estimating the log-likelihood of a given sentence under an autoregressive language model is straightforward: one can simply apply the chain rule and sum the log-likelihood values for each successive token. However, for masked language models (MLMs), there is no direct way to estimate the log-likelihood of a sentence. To address this issue, [Salazar et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.240.pdf) propose to estimate sentence pseudo-log-likelihood scores (PLLs) by successively masking each sentence token and determining its score from the rest of sentence. Here, we demonstrate that the original PLL method yields inflated scores for  out-of-vocabulary words and propose an adapted metric in which we mask not only the target token, but also all within-word tokens to the right of the target. We show that our adapted metric *PLL-word-l2r*) outperforms both the original metric and a PLL metric in which all intra-word tokens are masked: in particular, it better satisfies theoretical desiderata and better correlates with scores from autoregressive models. Finally, we show that the choice of metric affects even minimal pair evaluation benchmarks, such as BLiMP, underscoring the importance of selecting an appropriate scoring metric for evaluating MLM properties.

<img src="readme_figs/Fig1_conceptual.png" height="150">

We improve on the widely used pseudo-log-likelihood sentence score estimation method for masked language models from [Salazar et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.240.pdf) by making the scoring of multi-token words more cognitively plausible:
<img src="readme_figs/Fig2_motivation_for_adaptation.png" height="250">

## Citation
If you use this work, please cite:

```tex
TODO
```

***
## Repo branches
* `master` : Code accompanying the paper A Better Way to Do Masked Language Model Scoring by Carina Kauf & Anna Ivanova.
* `pr-branch-into-minicons` : Code for including the optimized masked-language modeling metric into the [minicons] library (https://github.com/kanishkamisra/minicons/tree/master) maintained by [Kanishka Misra](https://github.com/kanishkamisra) and supports 
functionalities for **(i)** extracting word representations from contextualized word embeddings and **(ii)** scoring sequences using language model scoring techniques, including masked language models following [Salazar et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.240.pdf).
This repo is a wrapper around the `transformers` [library](https://huggingface.co/transformers) from HuggingFace :hugs:
