from minicons import scorer
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import re
from pathlib import Path


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
    parser.add_argument('--file', required=True, help='which BLiMP paradigm is currently being evaluated')
    parser.add_argument('--batch_size', type=int, default=500) # default comes from Salazar et al. (2020) blimp expts.
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

    ######################
    # Run BLiMP experiment
    ######################

    FILE = Path(os.path.join(os.getcwd(),args.file))
    dataset_name = str(FILE).split('/')[-1].split('.jsonl')[0]
    print(dataset_name)

    stimuli = []

    lines = [json.loads(line) for line in FILE.read_text().split('\n') if len(line.strip())]
    for line in lines:
        stimuli.append([line['sentence_good'], line['sentence_bad']])

    for pair in stimuli[:5]:
        print(f"{pair[0]} vs. {pair[1]}")
    print('\n')

    stimuli_dl = DataLoader(stimuli, batch_size=args.batch_size)

    good_scores, bad_scores = [], []

    for batch in tqdm(stimuli_dl):
        good, bad = batch
        if not args.model.startswith('gpt'): #if mlm model
            good_scores.extend(model.sequence_score(good, which_masking=args.which_masking,
                                                    reduction=lambda x: x.sum().item()))
            bad_scores.extend(model.sequence_score(bad, which_masking=args.which_masking,
                                                   reduction=lambda x: x.sum().item()))
        else:
            good_scores.extend(model.sequence_score(good, reduction=lambda x: x.sum().item()))
            bad_scores.extend(model.sequence_score(bad, reduction=lambda x: x.sum().item()))

    dataset_acc = np.mean([g > b for g, b in zip(good_scores, bad_scores)])

    ######################
    # Save result to file
    ######################

    # create results folder
    savename = f"results/BLiMP/{args.model}/"
    os.makedirs(savename, exist_ok=True)

    # create results filename
    if re.search('bert', args.model):
        if args.which_masking == "within_word_l2r":
            savename += f"blimp_{dataset_name}_AdjustedPLL_l2r"
        elif args.which_masking == "within_word_mlm":
            savename += f"blimp_{dataset_name}_AdjustedPLL_mlm"
        elif args.which_masking == "global_l2r":
            savename += f"blimp_{dataset_name}_AdjustedPLL_global_l2r"
        else:
            savename += f"blimp_{dataset_name}_OriginalPLL"
    else:
        savename += f"blimp_{dataset_name}"
    savename += ".txt"

    # Save result to file
    with open(savename, 'w') as outf:
        outf.write(f"{dataset_name}, {dataset_acc}")


if __name__ == '__main__':
    main()
