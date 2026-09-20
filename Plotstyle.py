"""
plot_style.py -- front-end styling helpers for the Plotly heatmaps (no simulation logic here).

    from plot_style import add_grid
    fig = add_grid(fig, n)                       # n = number of cells per side (e.g. depth.shape[1])
    st.plotly_chart(fig, use_container_width=True)

Works on BOTH heatmaps (water depth and risk class). It gives the map:
  1. a fine grid: a thin gap between every cell, so each region is a separate tile
  2. a bold major grid every `major_every` cells, drawn on top of the tiles
  3. a frame around the whole map
  4. square cells, row 0 at the top, and readable R/C axis labels
  5. (optional) bright outlines around chosen cells, e.g. the critical ones
"""
import plotly.graph_objects as go

FINE_GRID = "#64748B"                    # colour of the gaps between cells (slate)
MAJOR_GRID = "rgba(34,211,238,0.85)"     # cyan major grid lines
FRAME = "rgba(226,232,240,0.9)"
AXIS_TEXT = "#94A3B8"


def add_grid(fig, n, major_every=5, cell_gap=1.5, fine_color=FINE_GRID, major_color=MAJOR_GRID,
             outline_mask=None, outline_color="#F43F5E", title=None):
    """
    Add distinct grid lines to a heatmap figure and return it.

    n             cells per side (the map is n x n)
    major_every   draw a bold line every this many cells (0 or None turns the major grid off)
    cell_gap      thickness in pixels of the fine grid between cells (0 turns it off)
    outline_mask  optional (n, n) boolean array; True cells get a bright outline
    """
    # 1. fine grid: gaps between tiles let the plot background show through as grid lines
    fig.update_traces(xgap=cell_gap, ygap=cell_gap, selector=dict(type="heatmap"))

    # 2. major grid, on top of the tiles. Cell (r, c) is centred on integer coordinates,
    #    so cell edges sit at k - 0.5.
    lo, hi = -0.5, n - 0.5
    if major_every:
        marks = list(range(0, n, major_every)) + [n]
        for k in marks:
            edge = k - 0.5
            fig.add_shape(type="line", x0=edge, x1=edge, y0=lo, y1=hi, layer="above",
                          line=dict(color=major_color, width=1.4))
            fig.add_shape(type="line", y0=edge, y1=edge, x0=lo, x1=hi, layer="above",
                          line=dict(color=major_color, width=1.4))

    # 3. frame around the whole map
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=lo, y1=hi, layer="above",
                  line=dict(color=FRAME, width=2), fillcolor="rgba(0,0,0,0)")

    # 5. optional outlines around chosen cells (for example the critical ones)
    if outline_mask is not None:
        rows, cols = outline_mask.nonzero()
        for r, c in zip(rows, cols):
            fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5, layer="above",
                          line=dict(color=outline_color, width=1.6), fillcolor="rgba(0,0,0,0)")

    # 4. square cells, row 0 at the top, R/C labels every `major_every` cells
    step = major_every or max(1, n // 6)
    ticks = list(range(0, n, step))
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=[f"C{c}" for c in ticks], side="top",
                     range=[lo, hi], showgrid=False, zeroline=False, constrain="domain",
                     tickfont=dict(size=10, color=AXIS_TEXT), showline=False)
    fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=[f"R{r}" for r in ticks],
                     range=[hi, lo], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
                     constrain="domain", tickfont=dict(size=10, color=AXIS_TEXT), showline=False)
    fig.update_layout(plot_bgcolor=fine_color, margin=dict(l=40, r=10, t=44, b=10))
    if title:
        fig.update_layout(title=title)
    return fig
