import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def plot_linked_graph_and_curves(graph, curves, ax_graph, ax_curves, fig):
    # Plot graph
    nx.draw_networkx(graph, with_labels=True, ax=ax_graph)

    # Plot curves
    lines = []
    for curve in curves:
        line, = ax_curves.plot(curve[0], curve[1])  # curve: [xs, ys]
        lines.append(line)

    # Helper function: updates the colors of the curves
    def update_curve_colors(ind):
        for i, l in enumerate(lines):
            if i in ind['ind']:
                l.set_color('r')
                l.set_zorder(np.inf)
            else:
                l.set_color('lightgray')

    # Helper function: handles events from figure
    def hover(event):
        if event.inaxes == ax_graph:
            cont, ind = ax_graph.collections[0].contains(event)
            if cont:
                update_curve_colors(ind)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


# Example usage
if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)

    # Generate graph
    G = nx.watts_strogatz_graph(15, 4, 0.3, seed=seed)
    xs = [i for i in range(10)]
    ys = [[x**2*j for x in xs] for j in range(15)]

    # Generate curves
    curves = []
    for y in ys:
        curves.append([xs, y])

    # Create figure and plot
    f, ax = plt.subplots(nrows=2)
    plot_linked_graph_and_curves(G, curves, ax[0], ax[1], f)
    plt.show()
