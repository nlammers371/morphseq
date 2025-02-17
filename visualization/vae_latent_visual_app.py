import io
import base64
# import plotly.graph_objects as go
import os
# from sklearn.manifold import TSNE
from src.functions.dataset_utils import *
import pandas as pd
import plotly.express as px
# from _archive.functions_folder.utilities import path_leaf
# import skimage
from dash import dcc, html, Input, Output, no_update, callback
import glob2 as glob
import numpy as np
import dash
from src.functions.utilities import path_leaf
# import dash_ag_grid as dag


# Contains 100 images for each digit from MNIST
# vae_data_path = 'datasets/mini-mnist-1000.pickle'

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray((256 * np.asarray(im_matrix)).astype(np.uint8))
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def parse_model_options(dataRoot):
    # get list of model architectures
    model_list_raw = glob.glob(dataRoot + "*VAE*")
    model_list_raw = [path_leaf(m) for m in model_list_raw if os.path.isdir(m)]
    # model_name = "SeqVAE_z100_ne250_gamma_temp_self_and_other"

    # get list of training intances
    model_list = []
    for model_name in model_list_raw:
        training_list_raw = [path_leaf(fn) for fn in glob.glob(os.path.join(dataRoot, model_name, "*VAE*"))]
        training_list_raw = [tr for tr in training_list_raw if
                             os.path.isdir(os.path.join(dataRoot, model_name, tr, "figures"))]

        # keep only folders with readable umap_df files
        training_list = []
        for tr in training_list_raw:
            read_path = os.path.join(dataRoot, model_name, tr, "figures", "umap_df.csv")
            try:
                pd.read_csv(read_path)
                training_list.append(tr)
            except:
                pass

        if len(training_list) > 0:
            model_list.append(model_name)

    return model_list

def get_image_sampler(train_dir):   #, main_dims=None):
    # mode_vec = ["train", "eval", "test"]
    # if main_dims is None:
    #     main_dims = (576, 256)

    data_transform = make_dynamic_rs_transform()
    data_sampler_vec = []
    # for mode in mode_vec:
    data_sampler = MyCustomDataset(
        root=train_dir, #os.path.join(train_dir, mode),
        transform=data_transform,
        return_name=True
    )
        # data_sampler_vec.append(ds_temp)

    return data_sampler


def visualize_latent_space(dataRoot, model_architecture, training_instance, preload_flag=False):

    global vae_df, image_dict, figurePath

    figurePath = os.path.join(dataRoot, model_architecture, training_instance, "figures", '')

    def load_latents_dataset(dataRoot, model_architecture, training_instance):

        df = pd.read_csv(os.path.join(dataRoot, model_architecture, training_instance, "figures", "umap_df.csv"),
                         index_col=0)
        image_sampler = get_image_sampler(dataRoot)

        return {"df": df, "image_sampler": image_sampler}

    df_dict = load_latents_dataset(dataRoot, model_architecture, training_instance)

    global plot_label_list, plot_partition_list  # , image_sampler_list

    plot_label_list = ["predicted_stage_hpf",
                       "short_pert_name"]


    model_list = parse_model_options(dataRoot)
    # key_training_dict = {"SeqVAE_z100_ne250_triplet_loss_test_self_and_other":"SeqVAE_training_2024-01-06_03-55-23",
    #                      }
    # model_name = "SeqVAE_z100_ne250_triplet_loss_test_self_and_other"


    # now select initial values to display
    model_name = model_list[0]
    training_list_raw = [path_leaf(fn) for fn in glob.glob(os.path.join(dataRoot, model_name, "*VAE*"))]
    training_list_raw = [tr for tr in training_list_raw if
                         os.path.isdir(os.path.join(dataRoot, model_name, tr, "figures"))]

    # keep only folders with readable umap_df files
    training_list = []
    for tr in training_list_raw:
        read_path = os.path.join(dataRoot, model_name, tr, "figures", "umap_df.csv")
        try:
            pd.read_csv(read_path)
            training_list.append(tr)
        except:
            pass

    plot_partition_list = ["all", "biological", "non-biological"]
    df = df_dict["df"]
    perturbation_index = np.unique(df["short_pert_name"])
    perturbation_list = perturbation_index.tolist()

    # wt_ind = np.where(perturbation_index == 'wck-AB')[0][0]
    # image_sampler_list = df_dict["image_sampler"]

    ########################
    # App
    app = dash.Dash(__name__)

    def create_figure(df, plot_partition=None, plot_labels=None, plot_dim="3D UMAP", plot_class_list=None):

        if plot_labels is None:
            plot_labels = "predicted_stage_hpf"

        if plot_labels == "predicted_stage_hpf":
            cmap_plot = "magma"
        elif plot_labels == "short_pert_name":
            cmap_plot = "plotly"

        marker_opacity = 0.5

        if plot_class_list is not None:
            plot_indices = np.asarray(
                [i for i in range(df.shape[0]) if df.loc[i, "short_pert_name"] in plot_class_list])
        else:
            plot_indices = np.arange(df.shape[0])

        plot_df = df.iloc[plot_indices]

        global plot_variables

        if plot_dim == "3D UMAP":
            if plot_partition is None:
                plot_variables = ["UMAP_00_3", "UMAP_01_3", "UMAP_02_3"]

            elif plot_partition == "all":
                plot_variables = ["UMAP_00_3", "UMAP_01_3", "UMAP_02_3"]

            elif plot_partition == "biological":
                plot_variables = ["UMAP_00_bio_3", "UMAP_01_bio_3", "UMAP_02_bio_3"]

            elif plot_partition == "non-biological":
                plot_variables = ["UMAP_00_n_3", "UMAP_01_n_3", "UMAP_02_n_3"]

            fig = px.scatter_3d(plot_df, x=plot_variables[0], y=plot_variables[1], z=plot_variables[2],
                                opacity=marker_opacity,
                                color=plot_labels,
                                color_continuous_scale=cmap_plot)


            fig.update_layout(scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3")
            )

        elif plot_dim == "2D UMAP":
            if plot_partition is None:
                plot_variables = ["UMAP_00_2", "UMAP_01_2"]

            elif plot_partition == "all":
                plot_variables = ["UMAP_00_2", "UMAP_01_2"]

            elif plot_partition == "biological":
                plot_variables = ["UMAP_00_bio_2", "UMAP_01_bio_2"]

            elif plot_partition == "non-biological":
                plot_variables = ["UMAP_00_n_2", "UMAP_01_n_2"]

            fig = px.scatter(plot_df, x=plot_variables[0], y=plot_variables[1], opacity=marker_opacity, color=plot_labels,
                             color_continuous_scale=cmap_plot)

            fig.update_layout(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2"
                )
            # raise Exception("Plot partition options not yet implemented.")

        if plot_labels == "predicted_stage_hpf":
            fig.update_traces(marker=dict(size=4))
        elif plot_labels == "short_pert_name":
            fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='rgba(70,70,70,0.2)')))
        # selector=dict(mode='markers'))
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )

        return fig

    f = create_figure(df, plot_class_list=["wck-AB"])

    app.layout = html.Div([
        html.Button('Save', id='save-button'),
        html.P(id='save-button-hidden', style={'display': 'none'}),
        dcc.Graph(id='3d_scat', figure=f), #, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        html.Div([dcc.Checklist(id="checklist",
                                options=perturbation_list,
                                inline=True,
                                value=['wck-AB'],
                                labelStyle={'display': 'block'},
                                style={"height": 200, "width": 200, "overflow": "auto"}
                                ),
                  html.Div(id='checklist-output-container', hidden=True)
                  ]),
        # html.Div(id='df_list', hidden=True),
        html.Div(id='label_list', hidden=True),
        html.Div(id='partition_list', hidden=True),
        html.Div(id='plot_dim_list', hidden=True),
        html.Div([
            dcc.Dropdown(plot_label_list, plot_label_list[0], id='label-dropdown'),
            html.Div(id='label-output-container', hidden=True)
        ],
            style={'width': '15%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(plot_partition_list, plot_partition_list[0], id='partition-dropdown'),
            html.Div(id='partition-output-container', hidden=True)
        ],
            style={'width': '10%', 'display': 'inline-block'})
        ,
        html.Div([
            dcc.Dropdown(["2D UMAP", "3D UMAP"], "3D UMAP", id='dim-dropdown'),
            html.Div(id='dim-output-container', hidden=True)
        ],
            style={'width': '10%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(model_list, model_name, id='model-dropdown'),
            html.Div(id='model-output-container', hidden=True)
        ],
            style={'width': '25%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(training_list, training_list[0], id='training-dropdown'),
            html.Div(id='training-output-container', hidden=True)
        ],
            style={'width': '25%', 'display': 'inline-block'})

    ]
    )
    @app.callback(
        Output('model-output-container', 'children'),
        Input('model-dropdown', 'value')
    )
    def load_wrapper(value):
        return value
    @app.callback(
        dash.dependencies.Output('training-dropdown', 'options'),
        [dash.dependencies.Input('model-dropdown', 'value')]
    )
    def update_date_dropdown(name):
        training_list_raw = [path_leaf(fn) for fn in glob.glob(os.path.join(dataRoot, name, "*VAE*"))]
        training_list = []
        for tr in training_list_raw:
            read_path = os.path.join(dataRoot, model_name, tr, "figures", "umap_df.csv")
            try:
                pd.read_csv(read_path)
                training_list.append(tr)
            except:
                pass

        return training_list

    @app.callback(
        Output('training-output-container', 'children'),
        Input('training-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(
        Output('label-output-container', 'children'),
        Input('label-dropdown', 'value')
    )
    def load_wrapper(value):
        return value

    @app.callback(
        Output("checklist-output-container", "children"),
        Input("checklist", "value")
    )
    def change_values(value):
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
                   Input('dim-output-container', 'children'),
                   Input('checklist-output-container', 'children'),
                   Input("model-output-container", "children"),
                   Input("training-output-container", "children")])

    def chart_3d(plot_partition, plot_labels, plot_dim, pert_class_values, model_architecture, training_instance):

        global f

        df_dict = load_latents_dataset(dataRoot, model_architecture, training_instance)
        df = df_dict["df"]

        f = create_figure(df, plot_labels=plot_labels, plot_partition=plot_partition, plot_dim=plot_dim,
                          plot_class_list=pert_class_values)

        # f.update_layout(uirevision="Don't change")

        return f

    # @app.callback(Output("graph-tooltip-5", "children"),
    #               Input('3d_scat', 'relayoutData'),
    #               Input("graph-tooltip-5", "children"))
    # def update_camera(relayout_data, clickOut):
    #     # ctx = callback_context
    #     # caller = None if not ctx.triggered else ctx.triggered[0]['prop_id'].split(".")[
    #     #     0]  # in {'graph1', 'graph2', None}
    #
    #     # initialization, no relayoutData
    #
    #     # graph1 was interacted with: update graph2
    #     if relayout_data is None:
    #         return clickOut
    #
    #     elif 'scene.camera' in relayout_data:
    #         print("reset clickData")
    #
    #         return False, no_update, no_update
    #
    #     else:
    #         return clickOut

    @callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Output("3d_scat", "clickData"),
        Input("3d_scat", 'clickData'),
        Input('3d_scat', 'relayoutData'),
        Input('checklist-output-container', 'children')
    )
    def display_hover(hoverData, relayoutData, plot_class_list):  # relayoutData, ):
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        # if changed_id == '3d_scat.relayoutData':
        #     hoverData = None
        out_args = [False, no_update, no_update]
        #
        if hoverData is None:
            return out_args[0], out_args[1], out_args[2], None
        # ctx = dash.callback_context
        # ids = [c['prop_id'] for c in ctx.triggered]

        if hoverData is not None:
            # if hoverData is not None:
            #     print("check")

            # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            mode_vec = ["train", "eval", "test"]

            df_dict = load_latents_dataset(dataRoot, model_architecture, training_instance)
            df = df_dict["df"]

            if plot_class_list is not None:
                plot_indices = np.asarray(
                    [i for i in range(df.shape[0]) if df.loc[i, "short_pert_name"] in plot_class_list])
            else:
                plot_indices = np.arange(df.shape[0])

            image_sampler = df_dict["image_sampler"]

            # demo only shows the first point, but other points may also be available
            hover_data = hoverData["points"][0]
            bbox = hover_data["bbox"]

            xyz_pt = np.asarray([hover_data["x"], hover_data["y"], hover_data["z"]])
            xyz_array = df.loc[:, plot_variables]
            dist_vec = np.sum((xyz_pt - xyz_array)**2, axis=1)
            num = np.argmin(dist_vec)
            # num = plot_indices[hover_data["pointNumber"]]

            # look the point up in the data frame
            snip_id = df["snip_id"].iloc[num]
            age_hpf = df["predicted_stage_hpf"].iloc[num]
            pert = df["short_pert_name"].iloc[num]
            train_cat = df["train_cat"].iloc[num]

            if not preload_flag:
                # train_cat_ind_vec = np.where(df["train_cat"] == train_cat)[0]
                snip_list = df["snip_id"] #.iloc[train_cat_ind_vec]
                samp_num = np.where(snip_id == snip_list)[0][0]
                # mode_num = [i for i in range(len(mode_vec)) if mode_vec[i] == train_cat][0]

                im_matrix = np.squeeze(np.asarray(image_sampler[samp_num][0]).tolist()[0])
            else:
                im_matrix = image_dict[snip_id]

            # im_matrix = images[num]
            im_url = np_image_to_base64(im_matrix.T)
            children = [
                html.Div([
                    html.Img(
                        src=im_url,
                        style={"width": "150px", 'display': 'block', 'margin': '0 auto'},
                    ),
                    html.P(str(np.round(age_hpf, 1)) + " hpf | " + pert, style={'font-weight': 'bold'})
                ])
            ]

            return True, bbox, children, None

    @app.callback(
        Output('save-button-hidden', 'children'),
        Input('save-button', 'n_clicks'))

    def save_html(n_clicks):

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

        if 'save-button' in changed_id:

            f.write_html(os.path.join(figurePath, 'dynamic_plot_' + str(int(np.random.rand()*1e6)) + '.html'))
        # fig = px.scatter(x=range(10), y=range(10))
        # fig.write_html("path/to/file.html")
        # else:
        #     if hoverData is None:
        #         return False, None, None
        #     else:
        #         hover_data = hoverData["points"][0]
        #         bbox = hover_data["bbox"]
        #         bbox["x0"] = -100
        #         bbox["x1"] = -100
        #         bbox["y0"] = -100
        #         bbox["y1"] = -100
        #         children = []
        #         return True, bbox, children

    # return app

    app.run_server(debug=True, port=np.random.randint(1000, 9999, 1))


if __name__ == '__main__':
    # set parameters
    root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick//morphseq/training_data/20241107_ds/"
    # root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/"
    model_architecture = "SeqVAE_z100_ne150_sweep_01_block01_iter030"
    training_instance = "SeqVAE_training_2024-11-11_15-45-40"

    preload_flag = False

    # load image data
    visualize_latent_space(root, model_architecture, training_instance, preload_flag=preload_flag)

# if __name__ == "__main__":
#     app.run(debug=True)