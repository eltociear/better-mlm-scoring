from minicons import scorer
import os

stimuli = ['The hooligan wrecked the vehicle.',
           'The vehicle wrecked the hooligan.',
           'The vehicle was wrecked by the hooligan.',
           'The hooligan was wrecked by the vehicle.']


##################
## Sentence scores
##################
mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
# CK: added the parameter use_adjusted_metric=True, which specifies whether we should use the new masking approach or not.
print(mlm_model.sequence_score(stimuli, use_adjusted_metric=True, reduction = lambda x: -x.sum(0).item()))
print(mlm_model.sequence_score(stimuli, use_adjusted_metric=False, reduction = lambda x: -x.sum(0).item()))



####################
## Plot token scores
####################

#define global figure settings
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

custom_params = {"axes.spines.right": False,
                 "axes.spines.top": False,
                 'ytick.left': True,
                 'xtick.bottom': True,
                'grid.linestyle': "" #gets rid of horizontal lines
                }
sns.set_theme(font_scale=1, style="white", rc=custom_params)

import matplotlib.pyplot as plt
import numpy as np

for sent_i, sent in enumerate(stimuli):
    for use_adjusted_metric in [True, False]:
        print(f"use_adjusted_metric={use_adjusted_metric} | {sent}")
        scores = mlm_model.token_score(sent, use_adjusted_metric=use_adjusted_metric)[0]
        for score in scores:
            print(f"{score[0]} | {score[1]}")
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        avg = np.mean(y)

        fig, ax = plt.subplots()
        plt.plot(y, marker="o", label=f"Sentence PLL score = {round(sum(y),4)}")
        plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))
        ax.set_xticklabels(x)
        plt.axhline(y=avg, color='r', linestyle='--', label=f"Avg. token PLL score = {round(avg, 2)}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
        plt.ylabel('Token PLL score')
        if use_adjusted_metric:
            plt.title(f"{sent} | Adapted PLL metric")
            savesuffix = 'adaptedPLL'
        else:
            plt.title(f"{sent} | Original PLL metric")
            savesuffix = 'originalPLL'
        plt.tight_layout()

        out_dir = "results/motivationForAdaptation/"
        os.makedirs(out_dir, exist_ok=True)
        filename = f'{out_dir}/sent{sent_i+1}_{savesuffix}.png'
        plt.savefig(filename, dpi=180, bbox_inches='tight')
        plt.show()
        print('\n')