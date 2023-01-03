import pandas as pd

from minicons import scorer
from torch.utils.data import DataLoader
import json
import os
import argparse
from tqdm import tqdm
import re


def main():
    """
    This function scores sentences in a given paradigm belonging to the BLiMP benchmark & calculates the model's
    accurately at determining for each minimal sentence pair in the paradigm, which of the two sentences is the
    grammatical one (as determined by higher sentence likelihood)

    We use this function to compare pseudo-log-likelihood scores of a sequence as proposed by Salazar et al. (2019) vs.
    metric that is adjusted for better within-word token scoring
    :param *which_masking*
        > if set to 'original' (default) it calculates the PLL as proposed by Salazar et al. (2020)
        > if set to 'within_word_l2r' it calculates the PLL metric for a given sentence, masking out future word tokens
            for multi-token words
        > if set to 'within_word_mlm' it calculates the PLL metric for a given sentence, masking out all tokens of the
            word to which the current token belongs for multi-token words

    The scoring uses a batch size of 500 as a default (as proposed in the mlm-scoring library for this experiment by
    Salazar et al. (2020)).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch_size', type=int, default=2000) # default comes from Salazar et al. (2020) LibriSpeech expts.
    # https://github.com/awslabs/mlm-scoring/tree/a8fd29f3ca666da386be91eb7c319027603c58a4/examples/lingacc-blimp#ranking
    parser.add_argument('--which_masking', type=str, default="original",
                        help="whether to use the original or adjusted PLL metric or not, default is 'original'."
                             "Other options are 'within_word_l2r' and 'within_word_mlm'")
    args = parser.parse_args()

    if not args.which_masking == "original":
        assert not args.model.startswith('gpt'), 'Adjusted PLL metric is only defined for MLM models!'

    if args.model.startswith('gpt'):
        model = scorer.IncrementalLMScorer(args.model, 'cpu')
    elif re.search('bert', args.model):
        model = scorer.MaskedLMScorer(args.model, 'cpu')
    else:
        raise NotImplementedError

    ########################
    # Score LibriSpeech text
    ########################

    def _apply_tokenizer_opts(sent: str) -> str:
        sent += '.'
        sent = sent.capitalize()
        return sent

    stimuli = []
    with open('librispeech/data/test-clean.am.json') as json_file:
        corpus = json.load(json_file)
        for sent_idx, value in corpus.items():
            ref_stimulus = _apply_tokenizer_opts(value["ref"])
            ref_stim_length = ref_stimulus.split()
            stimuli += [(sent_idx, ref_stimulus, ref_stim_length)]

    stimuli_dl = DataLoader(stimuli, batch_size=args.batch_size)
    sent_ids, stimuli, token_lengths, scores, nr_words = [], [], [], [], []

    for batch in tqdm(stimuli_dl):
        sent_idxs, ref_stimuli, stim_lengths = batch
        if not args.model.startswith('gpt'): #if mlm model
            curr_scores, curr_token_lengths = model.sequence_score(ref_stimuli, which_masking=args.which_masking,
                                                    reduction=lambda x: x.sum().item(), output_num_tokens=True)
        else:
            curr_scores, curr_token_lengths = model.sequence_score(ref_stimulus, reduction=lambda x: x.sum().item(), output_num_tokens=True)

        #results.extend(list(zip(sent_idxs, ref_stimuli, curr_token_lengths, curr_scores)))
        sent_ids.extend(sent_idxs)
        stimuli.extend(ref_stimuli)
        token_lengths.extend(curr_token_lengths)
        nr_words.extend(stim_lengths)
        scores.extend(curr_scores)

    results_df = pd.DataFrame({
        "sentence id":sent_ids,
        "ref sentence":stimuli,
        "PLL score": scores
        "nr. of tokens":token_lengths,
        "nr. of words":nr_words
    })

    ######################
    # Save result to file
    ######################

    # create results folder
    savename = f"results/LibriSpeech/{args.model}/"
    os.makedirs(savename, exist_ok=True)

    # create results filename
    if re.search('bert', args.model):
        if args.which_masking == "within_word_l2r":
            savename += f"LibriSpeech_AdjustedPLL_l2r"
        elif args.which_masking == "within_word_mlm":
            savename += f"LibriSpeech_AdjustedPLL_mlm"
        else:
            savename += f"LibriSpeech_OriginalPLL"
    else:
        savename += f"LibriSpeech"
    savename += ".csv"

    results_df.to_csv(savename, index=False)

if __name__ == '__main__':
    main()
