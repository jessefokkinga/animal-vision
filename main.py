import base64
import io
import logging
from glob import glob

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image

from dashapp import dashapp
from utils import explain_prediction, make_prediction

classes = [
    animal.replace("animals/", "").replace("/", "")
    for animal in glob("animals/*/")
]
models = [
    model.replace("models/", "").replace(".h5", "")
    for model in glob("models/*")
]

dashapp.title = "Animal vision"
logging.basicConfig(level=logging.INFO)
app = dashapp.server
app.config.suppress_callback_exceptions = True

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "0rem 1rem",
    "backgroundColor": "#f8f9fa",
}

sidebar_main = html.Div(
    [
        html.H4("Animal vision tool", style={"margin-top": "10px"}),
        html.Hr(),
        dcc.Upload(
            id="upload-image",
            children=dbc.Button(
                "Upload File",
                color="info",
                className="me-1",
                style={"margin-top": "10px"},
            ),
        ),
        html.Hr(),
        dcc.Dropdown(
            models, id="model-selection", placeholder="Select a model"
        ),
        dbc.Button(
            "Generate prediction",
            id="generate-prediction",
            color="info",
            className="me-1",
            style={"margin-top": "10px"},
        ),
        dbc.Button(
            "Explain prediction",
            id="explain-prediction",
            color="info",
            className="me-1",
            style={"margin-top": "10px"},
        ),
        html.Hr(),
    ],
    style=SIDEBAR_STYLE,
)


@dashapp.callback(
    Output("output-image", "children"), Input("upload-image", "contents")
)
def update_output(image):
    if image is not None:
        children = parse_contents(image)
        return children


@dashapp.callback(
    Output("predicted-value", "children"),
    Input("generate-prediction", "n_clicks"),
    [State("upload-image", "contents"), State("model-selection", "value")],
    prevent_initial_call=True,
)
def generate_prediction(n, image, model):

    data = image.replace("data:image/jpeg;base64,", "")
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    classes.sort()
    pred_class, pred_conf = make_prediction(
        img, model=model, class_names=classes
    )

    return f"Prediction is {pred_class} with a confidence of {pred_conf}%"


@dashapp.callback(
    Output("prediction-explanation", "children"),
    Input("explain-prediction", "n_clicks"),
    [State("upload-image", "contents"), State("model-selection", "value")],
    prevent_initial_call=True,
)
def generate_explanation_plot(n, image, model):

    data = image.replace("data:image/jpeg;base64,", "")
    img = Image.open(io.BytesIO(base64.b64decode(data)))
    classes.sort()
    fig = explain_prediction(model, img, classes, 1)

    fig = parse_contents(fig, jpeg=False)
    return fig


def parse_contents(contents, jpeg=True):
    # Remove 'data:image/png;base64' from the image string,
    # see https://stackoverflow.com/a/26079673/11989081
    if jpeg:
        data = contents.replace("data:image/jpeg;base64,", "")
    else:
        data = contents.replace("data:image/png;base64,", "")
    img = Image.open(io.BytesIO(base64.b64decode(data)))

    fig = px.imshow(np.array(img))

    # Hide the axes and the tooltips
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=0, l=0, r=0),
        xaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
        yaxis=dict(showgrid=False, showticklabels=False, linewidth=0),
        hovermode=False,
    )

    return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": True})])


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "marginLeft": "20rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}
content = html.Div(
    [
        dcc.Loading(
            id="ls-loading-1",
            children=[
                html.Div(id="output-image", style={"textAlign": "center"})
            ],
            type="circle",
        ),
        dcc.Loading(
            id="ls-loading-2",
            children=[
                html.H6(
                    id="predicted-value",
                    children="",
                    style={"textAlign": "center"},
                )
            ],
            type="circle",
        ),
        dcc.Loading(
            id="ls-loading-3",
            children=[
                html.Div(
                    id="prediction-explanation", style={"textAlign": "center"}
                )
            ],
            type="circle",
        ),
    ],
    style=CONTENT_STYLE,
)

dashapp.layout = html.Div([sidebar_main, content])

if __name__ == "__main__":
    dashapp.run_server(debug=True)
