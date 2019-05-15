import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from sklearn.metrics import confusion_matrix
from cycler import cycler
import seaborn as sns
import math
import pandas as pd
import numpy as np
import itertools

dark_colors = ["#A51C30", "#808080",
               (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
               (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
               (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
               (0.4, 0.6509803921568628, 0.11764705882352941),
               (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
               (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
               (0.4, 0.4, 0.4)]
SPINE_COLOR = 'black'


def beutify(fig_width=None, fig_height=None, columns=1):

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {"lines.linewidth": 1.0,
              'text.latex.preamble': r'\usepackage{amsmath}',
              'pgf.texsystem': "pdflatex",
              'axes.labelsize': 10,
              'axes.titlesize': 10,
              'font.size': 10,
              'legend.fontsize': 6,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.dpi': 300,
              'font.family': 'serif',
              'axes.prop_cycle': cycler('color', dark_colors),
              "pgf.preamble": [r"\usepackage[utf8x]{inputenc}",
                               r"\usepackage[T1]{fontenc}"],
              'figure.figsize': [fig_width, fig_height]
              }
    rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

#    matplotlib.pyplot.tight_layout()

    return ax


def get_label_distribution(y, title=None):
    label_, counts_ = np.unique(y, return_counts=True)
    beutify()
    fig, ax = plt.subplots(1, 1)
    postion = np.arange(len(label_))
    plt.bar(postion, counts_, align='center')
    plt.xticks(postion, label_)
    ax.set_title(
        "{} Cholera Outbreak \n (0 = continue, 1 = drop)".format(title))
    ax.set_ylabel("Number of Students")
    for p in ax.patches:
        ax.annotate("%.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    format_axes(ax)





def plot_confusion_matrix(y_true, y_pred, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fig_num=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if fig_num is not None:
        plt.subplot(2,2,fig_num)
    fmt =   'd'
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.title("")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def savefig(filename, leg=None, format='.pdf', *args, **kwargs):
    """
    Save in PDF file with the given filename.
    """
    if leg:
        art=[leg]
        plt.savefig(filename + format, additional_artists=art, bbox_inches="tight", *args, **kwargs)
    else:
        plt.savefig(filename + format,  bbox_inches="tight", *args, **kwargs)
    plt.close()
