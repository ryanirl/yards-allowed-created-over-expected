# Visualizations 
import plotly.graph_objects as go

from typing import Union
from typing import Dict


def get_field_number_bottom() -> go.Scatter:
    # The yard lines on the field
    return go.Scatter(
        x = [20, 30, 40, 50, 60, 70, 80, 90, 100],
        y = [5, 5, 5, 5, 5, 5, 5, 5, 5],
        mode = "text",
        text = ["10", "20", "30", "40", "50", "40", "30", "20", "10"],
        textfont_size = 30,
        textfont_family = "Courier New, monospace",
        textfont_color = "#ffffff",
        showlegend = False,
        hoverinfo = "none"
    )


def get_field_number_top() -> go.Scatter:
    # The yard lines on the field
    return go.Scatter(
        x = [20, 30, 40, 50, 60, 70, 80, 90, 100],
        y = [48.5, 48.5, 48.5, 48.5, 48.5, 48.5, 48.5, 48.5, 48.5],
        mode = "text",
        text = ["10", "20", "30", "40", "50", "40", "30", "20", "10"],
        textfont_size = 30,
        textfont_family = "Courier New, monospace",
        textfont_color = "#ffffff",
        showlegend = False,
        hoverinfo = 'none'
    )


def get_line_of_scrimmage(line) -> go.Scatter:
    return go.Scatter(
        x = [line] * 2,
        y = [0, 53.5],
        line_dash = "dash",
        line_color = "blue",
        showlegend = False,
        hoverinfo = "none"
    )


def get_layout(title: str, n_frames: int, scale: Union[int, float] = 10) -> go.Layout:
    sliders_dict = _get_slider(n_frames)
    updatemenus_dict = _get_updatemenus()
    layout = go.Layout(
        title = title,
        autosize = False,
        width = 120 * scale,
        height = 60 * scale,
        xaxis = dict(
            range = [0, 120], 
            autorange = False, 
            tickmode = "array",
            tickvals = list(range(10, 111, 5)),
            showticklabels = False
        ),
        yaxis = dict(
            range = [0, 53.3], 
            autorange = False,
            showgrid = False,
            showticklabels = False
        ),
        plot_bgcolor = "#00B140",
        updatemenus = [updatemenus_dict],
        sliders = [sliders_dict]
    )

    return layout


def _get_updatemenus() -> Dict:
    # Start and stop buttons for animation
    return {
        "buttons": [
            {
                "args": [None, {
                    "frame": {"duration": 100, "redraw": False}, 
                    "fromcurrent": True, 
                    "transition": {"duration": 0}
                }],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }


def _get_slider_step(t) -> Dict:
    return {
        "args": [[t], {
            "frame": {"duration": 100, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0}
        }],
        "label": str(t),
        "method": "animate"
    }


def _get_slider(n_frames) -> Dict:
    # Slider to show frame position in animation
    return {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Frame:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [_get_slider_step(t + 1) for t in range(n_frames)]
    }


