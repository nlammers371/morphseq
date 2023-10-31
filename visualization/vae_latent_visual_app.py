import io
import base64
import pickle
from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
from PIL import Image
import os
from sklearn.manifold import TSNE
import numpy as np
from functions.pythae_utils import *
import pandas as pd
import plotly.express as px
from _archive.functions_folder.utilities import path_leaf
import skimage
import dash

# Contains 100 images for each digit from MNIST
# vae_data_path = 'datasets/mini-mnist-1000.pickle'

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def get_image_sampler(train_dir, main_dims=None):
    mode_vec = ["train", "eval", "test"]
    if main_dims is None:
        main_dims = (576, 256)

    data_transform = make_dynamic_rs_transform(main_dims)
    data_sampler_vec = []
    for mode in mode_vec:
        ds_temp = MyCustomDataset(
            root=os.path.join(train_dir, mode),
            transform=data_transform,
            return_name=True
        )
        data_sampler_vec.append(ds_temp)

    return data_sampler_vec

# def load_vae_data(vae_data_path):
#     vae_df = pd.read_csv(vae_data_path, index_col=0)
#     return vae_df
#
# # Load the data
# vae_df = load_vae_data(vae_data_path)
# # images = data['images']
# labels = vae_df['snip_id']
# umap_array = vae_df['snip_id']

# Flatten image matrices from (28,28) to (784,)
# flattenend_images = np.array([i.flatten() for i in images])

# # t-SNE Outputs a 3 dimensional point for each image
# tsne = TSNE(
#     random_state=123,
#     n_components=3,
#     verbose=0,
#     perplexity=40,
#     n_iter=300) \
#     .fit_transform(flattenend_images)

# fig = go.Figure(data=[go.Scatter3d(
#     x=tsne[:, 0],
#     y=tsne[:, 1],
#     z=tsne[:, 2],
#     mode='markers',
#     marker=dict(
#         size=2,
#         color=colors,
#     )
# )])
# 
# fig.update_traces(
#     hoverinfo="none",
#     hovertemplate=None,
# )
# 

# app = Dash(__name__)

# app.layout = html.Div(
#     className="container",
#     children=[
#         dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
#         dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
#     ],
# )
# 



def visualize_latent_space(dataRoot, model_architecture, tarining_instance):

    global vae_df

    def load_nucleus_dataset(dataRoot, model_architecture, training_instance):

        # modelPath = dataRoot + 'model.sav'
        # modelDataPath = dataRoot + 'model_data.csv'
        #
        # model = []
        # if os.path.isfile(modelPath):
        #     model = pickle.load(open(modelPath, 'rb'))
        #
        # model_data = []
        # if os.path.isfile(modelDataPath):
        #     model_data = pd.read_csv(modelDataPath)

        # df = vae_df.loc[vae_df["file"] == well_time_index, :]
        df = pd.read_csv(os.path.join(dataRoot, model_architecture, training_instance, "figures", "umap_df.csv"), index_col=0)
        image_sampler = get_image_sampler(dataRoot)
        return {"df": df, "image_sampler_list": image_sampler}


    df_dict = load_nucleus_dataset(dataRoot, model_architecture, training_instance)

    global plot_label_list, plot_partition_list#, image_sampler_list

    plot_label_list = ["predicted_stage_hpf",  "experiment_date", "medium", "master_perturbation", "train_cat", "recon_mse"]
    plot_partition_list = ["all", "biological", "non-biological"]
    df = df_dict["df"]
    image_sampler_list = df_dict["image_sampler_list"]

    ########################
    # App
    app = dash.Dash(__name__)

    def create_figure(df, plot_partition=None, plot_labels=None, plot_dim="3D UMAP"):

        if plot_labels is None:
            plot_labels = "predicted_stage_hpf"
        
        cmap_plot = "ice"

        if plot_dim == "3D UMAP":
            if plot_partition is None:
                fig = px.scatter_3d(df, x="UMAP_00_3", y="UMAP_01_3", z="UMAP_02_3", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "all":
                fig = px.scatter_3d(df, x="UMAP_00_3", y="UMAP_01_3", z="UMAP_02_3", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "biological":
                fig = px.scatter_3d(df, x="UMAP_00_bio_3", y="UMAP_01_bio_3", z="UMAP_02_bio_3", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "non-biological":
                fig = px.scatter_3d(df, x="UMAP_00_n_3", y="UMAP_01_n_3", z="UMAP_02_n_3", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)

        elif plot_dim == "2D UMAP":
            if plot_partition is None:
                fig = px.scatter(df, x="UMAP_00_2", y="UMAP_01_2", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "all":
                fig = px.scatter(df, x="UMAP_00_2", y="UMAP_01_2", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "biological":
                fig = px.scatter(df, x="UMAP_00_bio_2", y="UMAP_01_bio_2", opacity=0.5,
                                    color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            elif plot_partition == "non-biological":
                fig = px.scatter(df, x="UMAP_00_n_2", y="UMAP_01_n_2", opacity=0.5, color=plot_labels,
                                    color_continuous_scale=cmap_plot)
            
               
            # raise Exception("Plot partition options not yet implemented.")

        fig.update_traces(marker=dict(size=3))
        # fig.update_layout(coloraxis_showscale=False)
        # fig.update_coloraxes(showscale=False)
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(range=[xmin, xmax], ),
        #         yaxis=dict(range=[ymin, ymax], ),
        #         zaxis=dict(range=[zmin, zmax], autorange="reversed"),
        #         aspectratio=dict(x=1, y=1, z=0.5)))

        return fig

    f = create_figure(df)

    app.layout = html.Div([
                        dcc.Graph(id='3d_scat', figure=f, clear_on_unhover=True),
                        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
                        # html.Div(id='df_list', hidden=True),
                        html.Div(id='label_list', hidden=True),
                        html.Div(id='partition_list', hidden=True),
                        html.Div(id='plot_dim_list', hidden=True),
                        html.Div([
                            dcc.Dropdown(plot_label_list, plot_label_list[0], id='label-dropdown'),
                            html.Div(id='label-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'}),
                        html.Div([
                            dcc.Dropdown(plot_partition_list, plot_partition_list[0], id='partition-dropdown'),
                            html.Div(id='partition-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'})
                        ,
                        html.Div([
                            dcc.Dropdown(["2D UMAP", "3D UMAP"], "3D UMAP", id='dim-dropdown'),
                            html.Div(id='dim-output-container', hidden=True)
                        ],
                            style={'width': '30%', 'display': 'inline-block'})
                        ]
                        )

    @app.callback(
        Output('label-output-container', 'children'),
        Input('label-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(
        Output('partition-output-container', 'children'),
        Input('partition-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(
        Output('dim-output-container', 'children'),
        Input('dim-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(Output('3d_scat', 'figure'),
                 [Input('partition-output-container', 'children'),
                  Input('label-output-container', 'children'),
                  Input('dim-output-container', 'children')])

    def chart_3d(plot_partition, plot_labels, plot_dim):

        global f

        # check to see which values have changed
        # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        df_dict = load_nucleus_dataset(dataRoot, model_architecture, training_instance)
        df = df_dict["df"]

        f = create_figure(df, plot_labels=plot_labels, plot_partition=plot_partition, plot_dim=plot_dim)

        # f.update_layout(uirevision="Don't change")

        return f

    @callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("3d_scat", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        mode_vec = ["train", "eval", "test"]

        df_dict = load_nucleus_dataset(dataRoot, model_architecture, training_instance)
        df = df_dict["df"]
        image_sampler_list = df_dict["image_sampler_list"]

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        # look the point up in the data frame
        snip_id = df["snip_id"].iloc[num]
        train_cat = df["train_cat"].iloc[num]
        train_cat_ind_vec = np.where(df["train_cat"] == train_cat)[0]
        snip_list_cat = df["snip_id"].iloc[train_cat_ind_vec]
        samp_num = np.where(snip_id == snip_list_cat)[0][0]
        mode_num = [i for i in range(len(mode_vec)) if mode_vec[i] == train_cat][0]

        im_raw = np.asarray(image_sampler_list[mode_num][samp_num][0]).tolist()[0]
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

    app.run_server(debug=True, port=8053)

if __name__ == '__main__':

    # set parameters
    dataRoot = "/Users/nick/Dropbox (Cole Trapnell's Lab)/Nick/morphseq/training_data/20230915_vae/"
    model_architecture = "z100_bs032_ne250_depth05_out16_temperature_sweep2"
    training_instance = "MetricVAE_training_2023-10-27_09-29-34"

    # load image data
    visualize_latent_space(dataRoot, model_architecture, training_instance)
# if __name__ == "__main__":
#     app.run(debug=True)