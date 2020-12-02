import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


class LinkedPlotter:
    def __init__(self, graph, curves, ax_graph, ax_curves, fig):
        self.ax_graph = ax_graph
        self.ax_curves = ax_curves
        self.fig = fig

        # Plot graph
        nx.draw_networkx(graph, with_labels=True, ax=ax_graph)

        # Plot curves
        self.lines = []
        for curve in curves:
            line, = ax_curves.plot(curve[0], curve[1])  # curve: [xs, ys]
            self.lines.append(line)

        # Save current curves colors and zorders for later 'hover off' update
        self.colors = []
        self.zorders = []
        for line in self.lines:
            self.colors.append(line.get_color())
            self.zorders.append(line.get_zorder())

        fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.needs_refresh_on_hover_off = False

    def update_curve_colors(self, ind):
        for i, l in enumerate(self.lines):
            if i in ind['ind']:
                l.set_color('r')
                l.set_zorder(max(self.zorders)+1)
            else:
                l.set_color('lightgray')
        self.needs_refresh_on_hover_off = True

    def hover(self, event):
        if event.inaxes == self.ax_graph:
            cont, ind = self.ax_graph.collections[0].contains(event)
            if cont:
                self.update_curve_colors(ind)
                self.fig.canvas.draw_idle()
            elif self.needs_refresh_on_hover_off:
                for line, color, zorder in zip(self.lines, self.colors, self.zorders):
                    line.set_color(color)
                    line.set_zorder(zorder)
                self.fig.canvas.draw_idle()
                self.needs_refresh_on_hover_off = False


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
    linked_plotter = LinkedPlotter(G, curves, ax[0], ax[1], f)
    plt.show()
