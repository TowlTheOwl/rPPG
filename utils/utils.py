import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def interactive_graph(results:dict, title:str):
    """
    Plots an interactive graph.

    Inputs:
        results (dict): dictionary in the form: (method name -> (bpm, (graph_x, graph_y)))
            where graph_x is the x axis and graph_y is the y axis of the graph to display.
        title (str): title to display
    """

    print(f"Graphing {title}")
    # Graph all results
    fig, ax = plt.subplots(figsize=(8, 6))

    lines = []
    for key in results:
        line, = ax.plot(*results[key][1], label=key)
        lines.append(line)
        print(f"{key}: {results[key][0]} bpm")

    plt.subplots_adjust(left=0.25)
    ax.set_title(title)
    ax.grid(True)

    legend = ax.legend(loc='upper right')

    # Map each text item in the legend to its corresponding line plot object
    legend_lines = legend.get_lines()
    legend_texts = legend.get_texts()
    lookup_map = {}

    for leg_line, leg_text, line in zip(legend_lines, legend_texts, lines):
        # Allow users to click the text or the small line graphic
        leg_line.set_picker(True)
        leg_text.set_picker(True)
        # Both elements point back to the main data plot line
        lookup_map[leg_line] = line
        lookup_map[leg_text] = line

    # event handler function
    def on_pick(event):
        # Retrieve the specific legend object that was clicked
        clicked_artist = event.artist
        
        # Get the corresponding data plot line from our lookup dictionary
        target_line = lookup_map[clicked_artist]
        
        # Invert the visibility state of the main line graph
        new_visibility = not target_line.get_visible()
        target_line.set_visible(new_visibility)
        
        # Fade the legend item color to show it is hidden
        # Find all legend pieces tied to this target line to dim them
        for leg_item, plot_line in lookup_map.items():
            if plot_line == target_line:
                leg_item.set_alpha(1.0 if new_visibility else 0.2)
                
        # Redraw the canvas to update the layout immediately
        fig.canvas.draw_idle()

    # 5. Connect the event picker system to our custom handler function
    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()