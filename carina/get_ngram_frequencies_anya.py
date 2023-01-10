import zs
import math
import os
import argparse


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


@Memoize
def get_freq(phrase):
  num_words = len(phrase.split())
  # create an object based on the corpus of interest
  ZS = zs.ZS("/om/data/public/corpora/google-books-v2/eng-us-all/google-books-eng-us-all-20120701-"+str(num_words)+"gram.zs")
  # convert to binary 
  phrase_b = phrase.encode('ascii')
  # get freq
  phraselist = ZS.search(prefix=phrase_b)
  return math.log(len(list(phraselist))+1)    # smooth


def get_word_freq(word):
  word = word.lower()
  word_freq = get_freq(word)
  return word_freq


def main(args):
  vocab_path = {
      "EventsAdapt" : "eventsAdapt/EventsAdapt_vocabulary.txt",
      "LibriSpeech": "libriSpeech/LibriSpeech_vocabulary.txt"
  }

  with open(os.path.abspath(vocab_path[args.dataset])) as file:
    words = [line.rstrip() for line in file]

  with open(os.path.abspath(f"results/unigram_frequencies_anya/{args.dataset}_unigram_frequencies.txt"), "w") as fout:
    fout.write('Index\tWord\tUnigram_freq\n')  # add header

    for ind, word in enumerate(words):
      unigram_frequency = get_word_freq(word)
      fout.write('{}\t{}\t{}\n'.format(ind, word, unigram_frequency))  # add row


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', help="Can be one of EventsAdapt or LibriSpeech")
  args = parser.parse_args()

  out_dir = 'results/unigram_frequencies_anya/'
  os.makedirs(out_dir, exist_ok=True)

  main(args)
