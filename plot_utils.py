import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import networkx as nx


class LinkedPlotter:
    """
    Plots a graph and line plots in user-specified axes.

    There is a one-to-one pairing between nodes in the graph and curves provided, such that if a user hovers their mouse
    over one of the nodes, the corresponding curve in the line plots will be outlined. Furthermore, each node in the
    graph is colored according to its contribution at time-step k, where k is chosen via a slider in the bottom of the
    figure.

    Params:
        graph: Graph to be plotted.
        curves: List of curves to be plotted. Each element is a list [xs, ys], where xs and ys are respectively a list
        of the x and y coordinates of the points to be plotted.
        ax_graph: The axis in which the graph should be plotted.
        ax_curves: The axis in which the curves should be plotted.
    """
    def __init__(self, graph, curves, ax_graph, ax_curves, fig):
        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.curves = curves
        self.ax_graph = ax_graph
        self.ax_curves = ax_curves
        self.fig = fig

        # Plot graph, nodes are color-coded
        colors = [curve[1][-1] for curve in self.curves]
        sizes = [graph.degree(i)*10 for i in range(graph.order())]
        # Plots nodes in a circle: specially good for the small-world but also looks good for scale-free
        nx.draw_circular(graph, with_labels=False, ax=ax_graph, node_size=sizes, node_color=colors, vmin=0, vmax=100)
        # Plots nodes so that the graph is visualized better and sometimes can be good for identifying clusters
        # nx.draw_kamada_kawai(graph, with_labels=False, ax=ax_graph, node_size=sizes, node_color=colors, vmin=0, vmax=100)


        # Plot curves
        self.lines = []
        for curve in self.curves:
            line, = self.ax_curves.plot(curve[0], curve[1])  # curve: [xs, ys]
            self.lines.append(line)
        n_sim_steps = len(self.curves[0][0])
        self.ax_curves.set_xlim([-0.1, n_sim_steps-1])



        # Save current curves colors and zorders for later 'hover off' update
        self.colors = []
        self.zorders = []
        for line in self.lines:
            self.colors.append(line.get_color())
            self.zorders.append(line.get_zorder())

        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        self.needs_refresh_on_hover_off = False

        # Create a slider for the simulation steps (for node coloring)
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='white')
        self.slider = Slider(ax_slider, 'Simulation step', 0, n_sim_steps-1, valinit=n_sim_steps, valstep=1)
        self.slider.on_changed(self.update_node_colors)

        # Add guiding curve to curves plot
        self.guiding_curve = self.ax_curves.axvline(n_sim_steps-1, linestyle='--', color='r')

    def update_node_colors(self, value):
        # Update color of each node in the graph
        pc = self.ax_graph.collections[0]
        cmap = pc.cmap
        min_v, max_v = pc.get_clim()
        new_colors = np.array([curve[1][int(value)] for curve in self.curves])
        percents = np.clip((new_colors - min_v) / (max_v - min_v), 0, 1)
        pc.set_color(cmap(percents))

        # Adjust guiding curve x-position
        self.guiding_curve.set_xdata([value, value])

        self.fig.canvas.draw_idle()

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
