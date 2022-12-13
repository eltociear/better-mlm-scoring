from minicons import scorer
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import re
import pandas as pd
from pathlib import Path


def main():
    """
    This tests pseudo-log-likelihood scores of a sequence as proposed by Salazar et al. (2019), adjusted for better
    within-word token scoring
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
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

    ####################################
    # Run BLiMP Nr. agreement experiment
    ####################################

    STIMPATH = os.path.abspath("./blimp/data/")

    stimuli = []
    accuracy_dictionary = {}

    i = 0

    for jsonl in tqdm(Path(STIMPATH).glob('*.jsonl')):
        print(jsonl)
        dataset_name = str(jsonl).split('/')[-1].split('.jsonl')[0]

        lines = [json.loads(line) for line in jsonl.read_text().split('\n') if len(line.strip())]
        for line in lines:
            stimuli.append([line['sentence_good'], line['sentence_bad']])

        for pair in stimuli[:5]:
            print(f"{pair[0]} vs. {pair[1]}")

        stimuli_dl = DataLoader(stimuli, batch_size=100)

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

        accuracy_dictionary[dataset_name] = np.mean([g > b for g, b in zip(good_scores, bad_scores)])

        i += 1
        if i == 3:
            break

    # Create a dataframe from the dictionary
    df = pd.DataFrame.from_dict(accuracy_dictionary, orient='index')

    savename = "results/BLiMP/"
    os.makedirs(savename, exist_ok=True)

    if re.search('bert', args.model):
        if args.use_adjusted_metric:
            savename += f"{args.model}_AdjustedPLL"
        else:
            savename += f"{args.model}_OriginalPLL"
    else:
        savename += f"{args.model}"

    # Save the dataframe to a file
    df.to_csv(f'{savename}.csv')


if __name__ == '__main__':
    main()