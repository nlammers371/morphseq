import io
import base64
import pickle
from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
from PIL import Image
import os
from sklearn.manifold import TSNE
import numpy as np
import umap
import pandas as pd
from _archive.functions_folder.utilities import path_leaf
import skimage


# Contains 100 images for each digit from MNIST
vae_data_path = 'datasets/mini-mnist-1000.pickle'

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def load_vae_data(vae_data_path):
    vae_df = pd.read_csv(vae_data_path, index_col=0)
    return vae_df

# Load the data
vae_df = load_vae_data(vae_data_path)
# images = data['images']
labels = vae_df['snip_id']
umap_array = vae_df['snip_id']

# Flatten image matrices from (28,28) to (784,)
flattenend_images = np.array([i.flatten() for i in images])

# # t-SNE Outputs a 3 dimensional point for each image
# tsne = TSNE(
#     random_state=123,
#     n_components=3,
#     verbose=0,
#     perplexity=40,
#     n_iter=300) \
#     .fit_transform(flattenend_images)

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