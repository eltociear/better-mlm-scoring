from google_unigram_freq import main as calculate_unigram_freq
import pandas as pd
import numpy as np
import os

def main():
    out_dir = 'unigram_frequencies/'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.abspath("eventsAdapt/EventsAdapt_vocabulary.txt")) as file:
        words = [line.rstrip() for line in file]
    
    with open(os.path.abspath("unigram_frequencies/EventsAdapt_unigram_frequencies.txt"), "w") as fout:
        
        fout.write('Index\tWord\tUnigram_freq\tUnigram_prob\n') #add header
        
        for ind, word in enumerate(words):
            unigram_frequency, unigram_probability = calculate_unigram_freq(word)
            fout.write('{}\t{}\t{}\t{}\n'.format(ind, word, unigram_frequency, unigram_probability)) #add row

if __name__ == "__main__":
    main()
