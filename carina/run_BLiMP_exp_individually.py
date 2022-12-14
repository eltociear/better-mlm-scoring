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
    accurately at determining for each minimal sentence pair in the paradigm, which of the two sentences is the grammatical one
    (as determined by higher sentence likelihood)

    We use this function to compare pseudo-log-likelihood scores of a sequence as proposed by Salazar et al. (2019) vs.
    metric that is adjusted for better within-word token scoring
    :param use_adjusted_metric
        > if set to False (default) it calculates the PLL as proposed by Salazar et al. (2020)
        > if set to True it calculates the PLL metric for a given sentence, masking out future word tokens for multi-
        token words

    The scoring uses a batch size of 500 as a default (as proposed in the mlm-scoring library for this experiment by
    Salazar et al. (2020)).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--file', required=True, help='which BLiMP paradigm is currently being evaluated')
    parser.add_argument('--batch_size', type=int, default=500) # default comes from Salazar et al. (2020) blimp expts.
    # https://github.com/awslabs/mlm-scoring/tree/a8fd29f3ca666da386be91eb7c319027603c58a4/examples/lingacc-blimp#ranking
    parser.add_argument('--use_adjusted_metric', default=False,
                        help="whether to use the adjusted PLL metric or not, default is False", action='store_true')
    args = parser.parse_args()

    if args.use_adjusted_metric:
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

    FILE = Path(os.path.abspath(f"./blimp/data/{args.file}"))
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
            good_scores.extend(model.sequence_score(good, use_adjusted_metric=args.use_adjusted_metric,
                                                    reduction=lambda x: x.sum().item()))
            bad_scores.extend(model.sequence_score(bad, use_adjusted_metric=args.use_adjusted_metric,
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
        if args.use_adjusted_metric:
            savename += f"blimp_{dataset_name}_AdjustedPLL"
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