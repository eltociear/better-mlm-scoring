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
reduction = lambda x: -x.sum(0).item()
print(mlm_model.sequence_score(stimuli, which_masking="original", reduction=reduction))
print(mlm_model.sequence_score(stimuli, which_masking="within_word_l2r", reduction=reduction))
print(mlm_model.sequence_score(stimuli, which_masking="within_word_mlm", reduction=reduction))



####################
## Plot token scores
####################

#define global figure settings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
plt.style.use('seaborn-dark-palette')

###################
## Individual plots
###################

for sent_i, sent in enumerate(stimuli):
    for which_masking in ["original", "within_word_l2r", "within_word_mlm"]:
        print(f"which_masking={which_masking} | {sent}")
        scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
        for score in scores:
            print(f"{score[0]} | {score[1]}")
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        avg = np.mean(y)

        fig, ax = plt.subplots()
        plt.plot(y, marker="o", label=f"Sentence PLL score = {round(sum(y),2)}")
        plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))
        ax.set_xticklabels(x)
        plt.axhline(y=avg, color='r', linestyle='--', label=f"Avg. token PLL score = {round(avg, 2)}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
        plt.ylabel('Token PLL score')
        if which_masking == "within_word_l2r":
            plt.title("Adapted PLL metric with left-to-right word token masking")
            savesuffix = 'adaptedPLL_l2r'
        elif which_masking == "within_word_mlm":
            plt.title("Adapted PLL metric with whole-word masking")
            savesuffix = 'adaptedPLL_mlm'
        else:
            plt.title("Token-level masking (original PLL metric)")
            savesuffix = 'originalPLL'
        fig.tight_layout()

        out_dir = "results/motivationForAdaptation/"
        os.makedirs(out_dir, exist_ok=True)
        filename = f'{out_dir}/sent{sent_i+1}_{savesuffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print('\n')

###################
## Subplots fig.
###################

for sent_i, sent in enumerate(stimuli):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for i, which_masking in enumerate(["original", "within_word_l2r", "within_word_mlm"]):
        scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        avg = np.mean(y)

        axs[i].plot(y, marker="o", label=f"Sentence PLL score = {round(sum(y), 2)}")
        axs[i].set_xticks(np.arange(len(x)), np.arange(1, len(x) + 1), minor=False)
        axs[i].set_xticklabels(x)
        axs[i].axhline(y=avg, color='r', linestyle='--', label=f"Avg. token PLL score = {round(avg, 2)}")
        axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
        axs[i].set_ylabel('Token PLL score')
        if which_masking == "within_word_l2r":
            axs[i].set_title("Within-word left-to-right masking")
            savesuffix = 'adaptedPLL_l2r'
        elif which_masking == "within_word_mlm":
            axs[i].set_title(f"Whole-word masking")
            savesuffix = 'adaptedPLL_mlm'
        else:
            axs[i].set_title("Token-level masking (original PLL metric)")
            savesuffix = 'originalPLL'
    fig.suptitle(f"{sent}")
    fig.tight_layout()

    out_dir = "results/motivationForAdaptation/"
    os.makedirs(out_dir, exist_ok=True)
    filename = f'{out_dir}/sent{sent_i + 1}_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print('\n')

###################
## Overlaid plots
###################

import matplotlib.patches as patches

for sent_i, sent in enumerate(stimuli):
    fig, ax = plt.subplots()
    ys = []
    for i, which_masking in enumerate(["original", "within_word_l2r", "within_word_mlm"]):
        scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        ys.append(y)

        # Plot the data
        if which_masking == "within_word_l2r":
            label = "Within-word left-to-right masking"
        elif which_masking == "within_word_mlm":
            label = "Whole-word masking"
        else:
            label = "Original PLL metric (token-level masking)"

        ax.plot(y, marker="o", label=f"{label} | Sentence PLL score = {round(sum(y), 2)}")

    plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))
    ax.set_xticklabels(x)
    plt.ylabel('Token PLL score')

    # Find the indices where the three plots differ
    indices = [i for y1 in ys for y2 in ys for i, a in enumerate(zip(y1, y2)) if a[0] != a[1]]
    indices = sorted(np.unique(indices))

    shared_x = [elm if elm not in indices else None for elm in np.arange(len(x))]
    shared_y = [y[i] if i or i == 0 else None for i in shared_x]
    plt.plot(shared_x, shared_y, color='gray', marker="o", label="Token scores shared across metrics")

    ymin, ymax = ax.get_ylim()
    x = np.arange(1, len(x) + 1)

    # Create a rectangle patch for the box
    # matplotlib.patches.Rectangle(xy, width, height)
    rect = patches.Rectangle([x[indices[0]] - 2.25 , int(ymin)],
                             indices[-1] - indices[0] + 2.5,
                             int(ymax) + 0.5 - int(ymin),
                             linewidth=1, edgecolor='k', facecolor='none', ls='--')

    # Add the rectangle patch to the plot
    ax.add_patch(rect)
    plt.ylim(ymin, ymax + 0.5)

    # Add legend & title
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=1)

    # Save the plot
    plt.tight_layout()
    filename = f'{out_dir}/sent{sent_i + 1}_overlaid.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()