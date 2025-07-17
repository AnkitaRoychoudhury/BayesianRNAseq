import numpy as np
import pandas as pd
import bokeh.plotting
import bokeh.io
from bokeh.layouts import gridplot
import iqplot

def gridplot_ncols(plots, ncols=3):
    """
    Function to split list of plots into sublists and plot

    Args:
        plots: list of bokeh plots
        ncols: number of columns for display

    Returns:
        plots
    """
    
    rows = [plots[i:i + ncols] for i in range(0, len(plots), ncols)]
    return gridplot(rows)

def plot_scatter(true_param_dict, trace_param_dict, every = 1):
    """
    Function to develop scatter plots of posterior mean vs true value for parameters.

    Args:
        true_param_dict: dictionary with true values of parameters from simulated data
        trace_param_dict: dictionary with mean values of parameters from posterior
        every: (int) to save every nth plot for vector parameters

    Returns:
        scatter plots
        
    """

    plots = []
    count = 0

    
    for key in trace_param_dict.keys():
        # get data from each param
        x_vals = true_param_dict[key]
        y_vals = trace_param_dict[key]
        
        if isinstance(x_vals[0], (float, np.float64)):
            if count % every == 0:
                y_vals = [float(y) for y in y_vals]
                max_val = max(max(x_vals), max(y_vals))
                min_val = min(min(x_vals), min(y_vals))
                
                p = bokeh.plotting.figure(title = key, width = 400, height = 350,
                                         x_axis_label = 'true value', y_axis_label= 'posterior mean')
                
                p.scatter(x_vals, y_vals, size = 5, color = 'black')
                p.line(np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100), color = 'black')
                plots.append(p)
            count += 1

        elif isinstance(x_vals[0], np.ndarray):

            K = len(x_vals[0])
            
            for i in range(K):
                if count % every == 0:
                    curr_x_val = [arr[i] for arr in x_vals]
                    curr_y_val = [arr[i] for arr in y_vals]
                    curr_y_val = [float(val) for val in curr_y_val] 
                    
                    #print(key, curr_x_val[:5], curr_y_val[:5])
                    max_val = max((max(curr_x_val), max(curr_y_val)))
                    
                    p = bokeh.plotting.figure(title = key + ' ' + str(i), width = 400, height = 350,
                                             x_axis_label = 'true value', y_axis_label= 'posterior mean')
                    p.scatter(curr_x_val, curr_y_val, size = 5, color = 'black')
                    p.line(np.linspace(0, max_val), np.linspace(0, max_val), color = 'black')
                    plots.append(p)
                count += 1
                

    grid = gridplot_ncols(plots, ncols=3)
    bokeh.io.show(grid)
                
        
def plot_rank_histograms(rank_param_dict, bins=50, every = 1):
    """
    Function to make rank histograms

    Args:
        rank_param_dict: dictionary of ranks for all parameters
        bins: number of bins to use for histograms
        every: (int) to save every nth plot for vector parameters

    Returns:
        rank histogram plots
    """
    plots = []
    count = 0
    
    for key, ranks in rank_param_dict.items():
        if isinstance(ranks[0], list):  # for b_s which is list of lists
            K = len(ranks)
            for k in range(K):
                if count % every == 0:
                    data = ranks[k]
                    max_rank = max(ranks[k])
                    hist, edges = np.histogram(data, bins=bins, range=(0, max_rank))
                    p = bokeh.plotting.figure(title=f'Rank Histogram: {key} {k}', width = 400, height = 350)
                    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                           fill_color='skyblue', line_color='black')
                    plots.append(p)
                count += 1
        else:
            if count % every == 0:
                data = ranks
                max_rank = max(ranks)
                hist, edges = np.histogram(data, bins=bins, range=(0, max_rank))
                p = bokeh.plotting.figure(title=f'Rank Histogram: {key}', width = 400, height = 350)
                p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                       fill_color='skyblue', line_color='black')
                plots.append(p)
            count += 1
            
    grid = gridplot_ncols(plots, ncols=3)
    bokeh.io.show(grid)

def plot_rank_ecdf(rank_param_dict, every = 1):
    """
    Function to make rank histograms

    Args:
        rank_param_dict: dictionary of ranks for all parameters
        bins: number of bins to use for histograms
        every: (int) to save every nth plot for vector parameters

    Returns:
        rank histogram plots
    """

    
    plots = []
    count = 0
    
    for key, ranks in rank_param_dict.items():
        if isinstance(ranks[0], list):  # case for vectors like b_s
            K = len(ranks)
            for k in range(K):
                if count % every == 0:
                    data = pd.DataFrame({'rank': ranks[k]})
                    p = iqplot.ecdf(data, q='rank') #, x_range=(0, num_samples))
                    p.title.text = f"ECDF: {key} {k}"
    
                    max_rank = max(ranks[k])
                    
                    # theoretical line
                    x_line = np.linspace(0, max_rank, 200)
                    y_line = x_line / max_rank
                    p.line(x_line, y_line, line_dash="dashed", line_color="black", legend_label="Uniform CDF")
                    
                    # approximate KS band
                    n = len(ranks[k])
                    epsilon = 1.36 / np.sqrt(n)
                    p.line(x_line, np.clip(y_line + epsilon, 0, 1), line_color="gray", line_dash="dotdash")
                    p.line(x_line, np.clip(y_line - epsilon, 0, 1), line_color="gray", line_dash="dotdash")
    
                    plots.append(p)
                count += 1
                
        else:
            if count % every == 0:
                data = pd.DataFrame({'rank': ranks})
                p = iqplot.ecdf(data, q='rank') #, x_range=(0, num_samples))
                p.title.text = f"ECDF: {key}"
    
                max_rank = max(ranks)
    
                # theoretical line
                x_line = np.linspace(0, max_rank, 200)
                y_line = x_line / max_rank
                p.line(x_line, y_line, line_dash="dashed", line_color="black", legend_label="Uniform CDF")
                
                # approximate KS band
                n = len(ranks)
                epsilon = 1.36 / np.sqrt(n)
                p.line(x_line, np.clip(y_line + epsilon, 0, 1), line_color="gray", line_dash="dotdash")
                p.line(x_line, np.clip(y_line - epsilon, 0, 1), line_color="gray", line_dash="dotdash")
                
                plots.append(p)
            count += 1

            
            
    grid = gridplot(plots, ncols=3, width=300, height=300)
    bokeh.io.show(grid)

