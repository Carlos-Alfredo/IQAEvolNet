# %%
from raygun.evaluation.inspect_tests import *

# switch to svg backend
# matplotlib.use("svg")
# update latex preamble
# plt.rcParams.update(
#     {
#         "svg.fonttype": "path",
#         # "font.family": "sans-serif",
#         # "font.sans-serif": "AvenirNextLTPro",  # ["Avenir", "AvenirNextLTPro", "Avenir Next LT Pro", "AvenirNextLTPro-Regular", 'UniversLTStd-Light', 'Verdana', 'Helvetica']
#         "path.simplify": True,
#         # "text.usetex": True,
#         # "pgf.rcfonts": False,
#         # "pgf.texsystem": 'pdflatex', # default is xetex
#         # "pgf.preamble": [
#         #      r"\usepackage[T1]{fontenc}",
#         #      r"\usepackage{mathpazo}"
#         #      ]
#     }
# )

# %%
paths = [
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/test_eval1_metrics.json",
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/*/test_eval1_metrics.json",
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/*/*/test_eval1_metrics.json",
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/*/*/*/test_eval1_metrics.json",
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/*/*/*/*/test_eval1_metrics.json",
    "/nrs/funke/rhoadesj/raygun/experiments/ieee-isbi-2023/03_evaluate/*/*/*/*/*/test_eval1_metrics.json",
]
results, basename, tags = load_json_tests(paths)

print(
    *[
        f"{k}: \n\tvoi_merge={v['voi_merge']} \t voi_split={v['voi_split']}\n\tVOI sum={v['voi_merge']+v['voi_split']}\n\n"
        for k, v in results.items()
    ]
)

raw_dict, sums, means, mins, maxs = make_data_dict(results)

# %%
# Try stats

stats = test_stats(raw_dict, stat=ks_2samp)
fig = plot_stats(stats)

# %%
# PLOTTING ===============================================================================================================
fig, axes = plt.subplots(len(sums), 1, figsize=(7, 15))
for c, (metric, results) in enumerate(sums.items()):
    axes[c].set_title(metric)
    # if c == 0:
    axes[c].set_ylabel("Sum of split and merge scores")
    # x_labels = ["train\npredict\nmean=\nmin=\nmax="]
    x_labels = ["Train on:\nPredict on:\nBest score = "]
    for x, ((train, predict), result) in enumerate(results.items()):
        means[metric][train, predict] = np.mean(result)
        maxs[metric][train, predict] = np.max(result)
        mins[metric][train, predict] = np.min(result)
        axes[c].scatter(
            np.ones_like(result) * x + 1,
            result,
            label=f"train-{train} | predict-{predict}",
        )
        # x_labels.append(
        #     f"{train}\n{predict}\n{means[metric][train, predict]:3.4f}\n{mins[metric][train, predict]:3.4f}\n{maxs[metric][train, predict]:3.4f}"
        # )
        x_labels.append(f"{train}\n{predict}\n{mins[metric][train, predict]:3.4f}")
    axes[c].set_xticks(range(x + 2))
    axes[c].set_xticklabels(x_labels)
fig.tight_layout()

# %%
trains = [
    "real30nm",
    "real90nm",
    "link",
    "split",
]  # set([keys[0] for keys in list(sums.values())[0].keys()])
predicts = [
    "real30nm",
    "link",
    "split",
    "real90nm",
]  # set([keys[1] for keys in list(sums.values())[0].keys()])
fig, axes = plt.subplots(len(sums), 1, figsize=(7, 15))
for c, (metric, results) in enumerate(sums.items()):
    axes[c].set_title(metric)
    # if c == 0:
    axes[c].set_ylabel("Sum of split and merge scores")
    # x_labels = ["train\npredict\nmean=\nmin=\nmax="]
    x_labels = ["Train on:\nPredict on:\nBest score = \nMean = \nWorst score = "]
    # x_labels = ["Train on:\nPredict on:\nBest score = "]
    x = 0
    for train in trains:
        for predict in predicts:
            if (train, predict) not in results.keys():
                continue
            result = results[train, predict]
            means[metric][train, predict] = np.mean(result)
            maxs[metric][train, predict] = np.max(result)
            mins[metric][train, predict] = np.min(result)
            axes[c].scatter(
                np.ones_like(result) * x + 1,
                result,
                label=f"train-{train} | predict-{predict}",
            )
            x_labels.append(
                f"{train}\n{predict}\n{mins[metric][train, predict]:3.4f}\n{means[metric][train, predict]:3.4f}\n{maxs[metric][train, predict]:3.4f}"
            )
            # x_labels.append(f"{train}\n{predict}\n{mins[metric][train, predict]:3.4f}")
            x += 1
    axes[c].set_xticks(range(x + 1))
    axes[c].set_xticklabels(x_labels)
fig.tight_layout()
fig
