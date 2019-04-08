import matplotlib.pyplot as plt


def simpleplot(x_data1, y_data1, x_data2=[], y_data2=[]):
    plt.plot(x_data1, y_data1, 'bo', x_data2, y_data2, 'ro')
    plt.show()


def scatterplot(x_data, y_data, x_label="", y_label="", title="", color="r"):
    fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax.scatter(x=x_data, y=y_data, marker='o', c=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


