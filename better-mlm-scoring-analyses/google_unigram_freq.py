import sys, re, os, argparse, zs, math

def stringify_ngram(ngram):
    return " ".join(ngram)

def lookup_ngram_google_books_v2(ngram,start_year=1960,end_year=math.inf):
    z = zs.ZS("/om/data/public/corpora/google-books-v2/eng-us-all/google-books-eng-us-all-20120701-" + str(len(ngram)) + "gram.zs")
    ngram_string = " ".join(ngram)
    total = 0
    for record in z.search(prefix=(ngram_string + "\t").encode()):
       (found_ngram, year_str, tokens_str, _) = record.decode().split("\t")
       if found_ngram != ngram_string:
           break
       year = int(year_str)
       if year >= start_year and year <= end_year:
           total += int(tokens_str)
    return total

def main(word):
    """Look up the joint frequency of an ngram w_1 ... w_n (and the frequency of w_1 ... w_n-1 for n>1) in either the Google Web ngrams or Google Books ngrams."""
    parser = argparse.ArgumentParser(description='Look up the joint frequency of an ngram w_1 ... w_n (and the frequency of w_1 ... w_n-1 for n>1) in either the Google Web ngrams or Google Books ngrams.')
    parser.add_argument('-W', '--web', help='use Google Web 1T (if not specified, will use Google Books as default.  Google Web 1T may not be implemented right now either.)', action="store_true")
    parser.add_argument('-f', '--from_year', help='specify starting year for Google books counts', type=int, default=1960)
    parser.add_argument('-t', '--to_year', help='specify ending year for Google books counts', type=int, default=math.inf)
    args = parser.parse_args()
    ngram = [word]
    #print args
    if args.web:
        f = lookup_ngram
    else:
        f = lambda x: lookup_ngram_google_books_v2(x,start_year=args.from_year, end_year=args.to_year)
    # look up zero-gram overall count
    overall_count = f([])
    if True:
        n = len(ngram)
        event = ngram
        event_freq = f(event)
        event_prob = float(event_freq)/float(overall_count)
    return event_freq, event_prob

if __name__ == "__main__":
    main()

