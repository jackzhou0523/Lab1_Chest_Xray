import matplotlib.pyplot as plt

def plotepoch(list,title):
    plt.plot(list)  # plot the data
    plt.xticks(range(0, len(list) + 1, 5))  # set the tick frequency on x-axis
    plt.xlabel('Epoch')  # set the label for x-axis
    plt.title(title)  # set the title of the graph
    plt.show()

def plot_confusion_matrix (
        cf_matrix
        ):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns


    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Confusion Matrix\n');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Predicted Normal', 'Predicted Pneumonia'])
    ax.yaxis.set_ticklabels(['Actual Normal', 'Actual Pneumonia'])
    ## Display the visualization of the Confusion Matrix.
    plt.show()