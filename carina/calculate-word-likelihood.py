from minicons import scorer
import os
import argparse
import re
import pandas as pd

def main(args, context):
    if args.model.startswith('gpt'):
        model = scorer.IncrementalLMScorer(args.model, 'cpu')
    elif re.search('bert', args.model):
        model = scorer.MaskedLMScorer(args.model, 'cpu')
    else:
        raise NotImplementedError

    if args.dataset == "EventsAdapt":
        filepath = os.path.abspath("eventsAdapt/EventsAdapt_vocabulary.txt")
    elif args.dataset == "LibriSpeech":
        filepath = os.path.abspath("librispeech/data/LibriSpeech_vocabulary.txt")
    else:
        raise NotImplementedError

    with open(filepath) as file:
        words = [line.rstrip() for line in file]

    if not context == "":
        words = [f'{context} "{word}".' for word in words]

    if args.which_masking:
        word_token_scores = model.token_score(words, which_masking=args.which_masking)
    else:
        word_token_scores = model.token_score(words)

    if re.search('gpt', args.model):
        assert [elm[0] == ('<|endoftext|>', 0.0) for elm in word_token_scores]
        word_token_scores = [elm[1:] for elm in word_token_scores]
        #CK NOTE: using [1:] because I'm ignoring the EOS token for which the prob is being ignored scorer.py l. 920


    exclude_context = model.tokenizer.tokenize(context)
    exclude_context += ['.', '"', '".']
    exclude_context = [elm.lstrip('Ġ') for elm in exclude_context]
    word_token_scores = [[elm for elm in score_list if elm[0] not in exclude_context] for score_list in word_token_scores]
    words = ["".join([x[0].lstrip("##").lstrip("Ġ").rstrip(".") for x in elm]) for elm in word_token_scores]

    nr_tokens = [len(elm) for elm in word_token_scores]
    tokens = ["_".join([x[0] for x in elm]) for elm in word_token_scores]
    word_scores = [sum([x[1] for x in elm]) for elm in word_token_scores]

    df = pd.DataFrame({
        "word" : words,
        "nr. of tokens" : nr_tokens,
        "tokens" : tokens,
        "word score" : word_scores
    })

    out_dir = f'results/unigram_likelihoods/{args.model}'
    os.makedirs(out_dir, exist_ok=True)

    # create results filename
    if re.search('bert', args.model):
        if args.which_masking == "within_word_l2r":
            savename = f"{args.dataset}_AdjustedPLL_l2r"
        elif args.which_masking == "within_word_mlm":
            savename = f"{args.dataset}_AdjustedPLL_mlm"
        elif args.which_masking == "original":
            savename  = f"{args.dataset}_OriginalPLL"
        else:
            raise NotImplementedError("No masking option supplied!")
    else:
        savename = f"{args.dataset}"

    if not context == "":
        savename += "_context=" + "+".join(context.split())

    savename += ".csv"

    df.to_csv(os.path.join(out_dir, savename), index=False)


if __name__ == '__main__':
    # Note: minicons adds an additional 0.0 log-probability for the first token/word as convention.
    # source: https://github.com/kanishkamisra/minicons/blob/master/examples/surprisals.md
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help="Can be LibriSpeech or EventsAdapt")
    parser.add_argument('--model', required=True)
    parser.add_argument('--which_masking', help="Can be original, within_word_l2r or within_word_mlm")
    args = parser.parse_args()

    contexts = ["I opened a dictionary and randomly picked a word. It was", "I opened the dictionary and picked the word", "My word is"]
    for context in contexts:
        main(args, context)