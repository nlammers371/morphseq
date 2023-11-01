import io
import base64
import pickle

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go

from PIL import Image

from sklearn.manifold import TSNE
import numpy as np

# Contains 100 images for each digit from MNIST
mnist_path = 'datasets/mini-mnist-1000.pickle'

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def load_mini_mnist():
    with open(mnist_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the data
data = load_mini_mnist()
images = data['images']
labels = data['labels']

# Flatten image matrices from (28,28) to (784,)
flattenend_images = np.array([i.flatten() for i in images])

# t-SNE Outputs a 3 dimensional point for each image
tsne = TSNE(
    random_state=123,
    n_components=3,
    verbose=0,
    perplexity=40,
    n_iter=300) \
    .fit_transform(flattenend_images)

# Color for each digit
color_map = {
    0: "#E52B50",
    1: "#9F2B68",
    2: "#3B7A57",
    3: "#3DDC84",
    4: "#FFBF00",
    5: "#915C83",
    6: "#008000",
    7: "#7FFFD4",
    8: "#E9D66B",
    9: "#007FFF",
}
colors = [color_map[label] for label in labels]

fig = go.Figure(data=[go.Scatter3d(
    x=tsne[:, 0],
    y=tsne[:, 1],
    z=tsne[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=colors,
    )
)])

fig.update_traces(
    hoverinfo="none",
    hovertemplate=None,
)

app = Dash(__name__)

app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
    ],
)

@callback(
    Output("graph-tooltip-5", "show"),
    Output("graph-tooltip-5", "bbox"),
    Output("graph-tooltip-5", "children"),
    Input("graph-5", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]

    im_matrix = images[num]
    im_url = np_image_to_base64(im_matrix)
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P("MNIST Digit " + str(labels[num]), style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children

if __name__ == "__main__":
    app.run(debug=True)