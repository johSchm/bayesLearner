import matplotlib.pyplot as plt


def scatterplot(x_data, y_data, x_label="", y_label="", title="", color="r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size, color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, size=10, color=color, alpha=0.75)

    if yscale_log:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

