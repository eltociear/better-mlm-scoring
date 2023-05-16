import sys
sys.path.append("..")
from adapted_minicons import scorer
import os

side_by_side_plot_metrics = ["original", "within_word_l2r", "within_word_mlm"]
individual_plot_metrics = ["global_l2r"]

stimuli = ['The hooligan wrecked the vehicle.',
           'The carnivore ate the steak.',
           'The traveler lost the souvenir.']

out_dir = "results/motivationForAdaptation/"
os.makedirs(out_dir, exist_ok=True)


##################
## Sentence scores
##################
mlm_model = scorer.MaskedLMScorer('bert-base-uncased', 'cpu')
reduction = lambda x: -x.sum(0).item()
print(mlm_model.sequence_score(stimuli, which_masking="original", reduction=reduction))
print(mlm_model.sequence_score(stimuli, which_masking="within_word_l2r", reduction=reduction))
print(mlm_model.sequence_score(stimuli, which_masking="within_word_mlm", reduction=reduction))
print(mlm_model.sequence_score(stimuli, which_masking="global_l2r", reduction=reduction))


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
sns.set_theme(font_scale=1.5, style="white", rc=custom_params)
plt.style.use('seaborn-dark-palette')

import matplotlib.patches as patches

RENAME_DICT_METRICS = {
    "original" : "PLL-original",
    "within_word_l2r" : "PLL-word-l2r",
    "within_word_mlm" : "PLL-whole-word",
    "global_l2r" : "PLL-sentence-l2r"
}

###################
## Side-by-side
###################

axhlinecolor = sns.cubehelix_palette(5)[2]
for sent_i, sent in enumerate(stimuli):
    fig, axs = plt.subplots(1, len(side_by_side_plot_metrics), figsize=(20, 6), sharey=True)
    ys = []
    for i, which_masking in enumerate(side_by_side_plot_metrics):
        print(f'#{which_masking}#')
        scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
        for score in scores:
            print(f"{score[0]} | {score[1]}")
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        ys.append(y)
        avg = np.mean(y)

        axs[i].plot(y, marker="o", linewidth=2, markersize=7,
                    label=f"Sentence PLL score = {round(sum(y), 2)}")
        axs[i].set_xticks(np.arange(len(x)), np.arange(1, len(x) + 1), minor=False)
        axs[i].set_xticklabels(x)
        axs[i].tick_params(axis='both', which='major', length=10, width=2)
        axs[i].tick_params(axis='both', which='minor', length=10, width=2)
        axs[i].axhline(y=avg, color=axhlinecolor, linestyle='--', label=f"Avg. token PLL score = {round(avg, 2)}")
        axs[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=1)

        if i == 0:
            axs[i].set_ylabel('Token PLL score')
        else:
            axs[i].set_ylabel('')
            
        axs[i].set_title(f"{RENAME_DICT_METRICS[which_masking]}", fontsize=20, fontweight="bold")

    # overlay shared with different color
    # Find the indices where the three plots differ
    if "global_l2r" in side_by_side_plot_metrics:
        ys = ys[:-1] #exclude last one for shading, because the scores of the last one are all different
    indices = [i for y1 in ys for y2 in ys for i, a in enumerate(zip(y1, y2)) if a[0] != a[1]]
    indices = sorted(np.unique(indices))

    indices_1, indices_2 = None, None

    def checkConsecutive(l):
        return sorted(l) == list(range(min(l), max(l) + 1))

    if not checkConsecutive(indices):
        for ind in range(len(indices)):
            indices_1 = indices[:len(indices)-ind-1]
            if not checkConsecutive(indices_1):
                print(f"List {indices_1} does not contain only consecutive numbers!")
            else:
                indices_2 = [ind for ind in indices if ind not in indices_1]
                break


    not_shared_x = [elm if elm in indices else None for elm in np.arange(len(x))]
    not_shared_ys = []
    for ind, y in enumerate(ys):
        not_shared_ys.append([y[i] if i or i == 0 else None for i in not_shared_x])

    for ind, ax in enumerate(axs):
        #ax.plot(not_shared_x, not_shared_ys[ind], marker="o", color='r')

        ymin, ymax = axs[ind].get_ylim()
        x_ind = np.arange(1, len(x) + 1)
        # Create a rectangle patch for the box
        # matplotlib.patches.Rectangle(xy, width, height)
        if indices_1:
            rect = patches.Rectangle([x_ind[indices_1[0]] - 1.25, int(ymin)],
                                     indices_1[-1] - indices_1[0] + 0.5,
                                     int(ymax) + 0.5 - int(ymin),
                                     linewidth=1, edgecolor='gray', facecolor='none', ls=':')
            ax.add_patch(rect)

            rect = patches.Rectangle([x_ind[indices_2[0]] - 1.25, int(ymin)],
                                     indices_2[-1] - indices_2[0] + 0.5,
                                     int(ymax) + 0.5 - int(ymin),
                                     linewidth=1, edgecolor='gray', facecolor='none', ls=':')
            ax.add_patch(rect)

        else:
            rect = patches.Rectangle([x_ind[indices[0]] - 1.25, int(ymin)],
                                     indices[-1] - indices[0] + 0.5,
                                     int(ymax) + 0.5 - int(ymin),
                                     linewidth=1, edgecolor='whitesmoke', facecolor='whitesmoke', ls='-')
            ax.add_patch(rect)

        bold_x = [x[i] if not_shared_x[i] else None for i in range(len(not_shared_x))]
        for lab in ax.get_xticklabels():
            if lab.get_text() in bold_x:
                lab.set_fontweight('bold')

    plt.ylim(ymin, ymax + 0.5)

    fig.suptitle(f"{sent}", fontweight="bold", fontstyle='italic')
    fig.tight_layout()

    out_dir = "results/motivationForAdaptation/"
    os.makedirs(out_dir, exist_ok=True)
    filename = f'{out_dir}/{sent}_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    filename = f'{out_dir}/{sent}_combined.svg'
    plt.savefig(filename, format="svg", dpi=300, bbox_inches='tight')
    filename = f'{out_dir}/{sent}_combined.pdf'
    plt.savefig(filename, format="pdf", dpi=300, bbox_inches='tight')
    plt.show()

        
###################
## Individual plots
###################

for sent_i, sent in enumerate(stimuli):
    for which_masking in individual_plot_metrics:
        print(f"which_masking={which_masking} | {sent}")
        scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
        for score in scores:
            print(f"{score[0]} | {score[1]}")
        x = [elm[0] for elm in scores]
        y = [elm[1] for elm in scores]
        avg = np.mean(y)

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(y, marker="o", linewidth=2, markersize=7, label=f"Sentence PLL score = {round(sum(y),2)}")
        plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))
        ax.set_xticklabels(x)
        plt.axhline(y=avg, color=axhlinecolor, linestyle='--', label=f"Avg. token PLL score = {round(avg, 2)}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
        plt.ylabel('Token PLL score')
        
        plt.title(f"{RENAME_DICT_METRICS[which_masking]}", fontsize=20, fontweight="bold")
        fig.tight_layout()

        filename = f'{out_dir}/{sent}_{which_masking}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        filename = f'{out_dir}/{sent}_{which_masking}.svg'
        plt.savefig(filename, format="svg", dpi=300, bbox_inches='tight')
        filename = f'{out_dir}/{sent}_{which_masking}.pdf'
        plt.savefig(filename, format="pdf", dpi=300, bbox_inches='tight')
        plt.show()
        
# ###################
# ## Overlaid plots
# ###################

# elif WHICH_PLOT == "overlaid":

#     marker_dict = {
#         "original" : "o",
#         "within_word_l2r" : "P",
#         "within_word_mlm" : "p"
#     }

#     for sent_i, sent in enumerate(stimuli):
#         fig, ax = plt.subplots()
#         ys = []
#         for i, which_masking in enumerate(["original", "within_word_l2r", "within_word_mlm"]):
#             scores = mlm_model.token_score(sent, which_masking=which_masking)[0]
#             x = [elm[0] for elm in scores]
#             y = [elm[1] for elm in scores]
#             ys.append(y)

#             # Plot the data
#             if which_masking == "within_word_l2r":
#                 label = "Within-word left-to-right masking"
#             elif which_masking == "within_word_mlm":
#                 label = "Whole-word masking"
#             else:
#                 label = "Original PLL metric (token-level masking)"

#             ax.plot(y, marker=marker_dict[which_masking], label=f"{label} | Sentence PLL score = {round(sum(y), 2)}")

#         plt.xticks(np.arange(len(x)), np.arange(1, len(x) + 1))
#         ax.set_xticklabels(x)
#         plt.ylabel('Token PLL score')

#         # Find the indices where the three plots differ
#         indices = [i for y1 in ys for y2 in ys for i, a in enumerate(zip(y1, y2)) if a[0] != a[1]]
#         indices = sorted(np.unique(indices))

#         shared_x = [elm if elm not in indices else None for elm in np.arange(len(x))]
#         shared_y = [y[i] if i or i == 0 else None for i in shared_x]
#         plt.plot(shared_x, shared_y, color='gray', marker="s", label="Token scores shared across metrics")

#         ymin, ymax = ax.get_ylim()
#         x = np.arange(1, len(x) + 1)

#         # Create a rectangle patch for the box
#         # matplotlib.patches.Rectangle(xy, width, height)
#         rect = patches.Rectangle([x[indices[0]] - 2.25 , int(ymin)],
#                                  indices[-1] - indices[0] + 2.5,
#                                  int(ymax) + 0.5 - int(ymin),
#                                  linewidth=1, edgecolor='k', facecolor='none', ls='--')

#         # Add the rectangle patch to the plot
#         ax.add_patch(rect)
#         plt.ylim(ymin, ymax + 0.5)

#         # Add legend & title
#         #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=1)
#         ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=1)

#         # Save the plot
#         plt.tight_layout()
#         filename = f'{out_dir}/sent{sent_i + 1}_overlaid.png'
#         plt.savefig(filename, dpi=300, bbox_inches='tight')

#         # Show the plot
#         plt.show()

# else:
#     raise NotImplementedError