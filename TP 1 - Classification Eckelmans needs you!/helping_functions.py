#!/usr/bin/env python3
# License: BSD 3 clause 
# Inspired from https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

""" -----------------------------------------------------------------------------------------
Visualize the predicted and true classification
INPUT: 
    - clf: trained classifier
    - X: feature data
    - X_train y_train: training data and objective
    - X_test y_test: testing data and objective
    - score: score of the classifier on X 
    - n_neighbors: value of k in the classifier clf
    - title: title of your graph
    - x_label: name of feature 1
    - y_label: name of feature 2
OUTPUT:
    VOID - visualize the classification.
----------------------------------------------------------------------------------------- """

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

def vis_clf(clf, X, X_train, y_train, X_test, y_test, score, n_neighbors, title="", x_label="", y_label=""):
    
    x_min = np.amin(X, axis=0)  # minima over the columns
    x_max = np.amax(X, axis=0)  # maxima over the columns    
    
    # For better visu
    x_min = x_min - 0.1 * (x_max - x_min)
    x_max = x_max + 0.1 * (x_max - x_min)
    
    h = 20 if len(x_min) == 3 else 100  # number of points in the grid
    
    grid = np.transpose(np.linspace(x_min, x_max, h))
    xx = np.meshgrid(*grid)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh
    Z = clf.predict(np.transpose(list(map(lambda x: x.ravel(), xx))))
    
    # Put the result into a color plot
    if len(xx) == 2:
        fig, ax = plt.subplots()
        Z = Z.reshape(xx[0].shape)
        plt.pcolormesh(*xx, Z, cmap=cmap_light, alpha=.8, shading='auto')
    else:  # 3D image
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        mesh = list(map(lambda x: x.ravel(),xx))
        ax.scatter(*mesh, c=Z, cmap=cmap_light, alpha=0.2)
        
    # Plot also the training and testing points   
    scatter = ax.scatter(*np.transpose(X_train), c=y_train, cmap=cmap_bold, \
                         edgecolor='k', s=20)
    scatter = ax.scatter(*np.transpose(X_test), c=y_test, cmap=cmap_bold, \
                         edgecolor='k', s=20, marker="s")
    
    ax.set_title("{} (k = {})".format(title, n_neighbors))
    ax.autoscale(enable=True, axis='both', tight=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if len(xx) == 3:
        ax.set_zlabel('Scaled Lot Frontage')

    l_e = (scatter.legend_elements()[0], ['Low rated', 'High rated'])
    legend1 = ax.legend(*l_e,
                    loc="upper left", title="Classes")
    ax.add_artist(legend1)
    legend_elements = [Line2D([0], [0], marker='o', color='None', label='Scatter',
                          markeredgecolor='grey', markersize=6),
                   Line2D([0], [0], marker='s', color='None', label='Scatter',
                          markeredgecolor='grey', markersize=6)]
    legend2 = ax.legend(handles=legend_elements, labels=['Train', 'Test'],loc="upper right")
    ax.add_artist(legend2)
    
    position = [0.9, 0.1] if len(xx) == 2 else [0.9, 0.1, 0.1]
    ax.text(*position, '{:.2f}'.format(score), size=15,
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()


""" -----------------------------------------------------------------------------------------
Visualize the relation features-target
INPUT: 
    - data: our data obtained via read_csv
OUTPUT:
    VOID - visualize the relation between each of the feature and the target: OverallQual.
----------------------------------------------------------------------------------------- """

def plot_comparison_target_feature(data, cols=3):

    all_boxes  = []
    cols = 3

    fig = make_subplots(rows=len(data.columns) // cols + 1, cols=cols, shared_yaxes=True)
    for idx, label in enumerate(data): 
        temp = go.Scatter(
            x      = data[label],
            y      = data['OverallQual'],
            name   = label,
            mode   = 'markers',
            marker = dict(color = 'rgba(50,160,150,0.7)', line = dict(color = 'rgb(0,0,0)', width = 1.5)),
            text   = label,
            showlegend = False
        )

        row = idx // cols + 1
        col = idx % cols + 1
        fig.append_trace(temp, row=row , col=col)
        fig.update_xaxes(title_text=label, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="OverallQual", row=row, col=col)

    fig.update_layout(title_text="Features vs. target", height=10000/cols)
    fig.show()

