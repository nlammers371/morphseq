import io
import base64
import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, Dash, no_update
from src.functions.dataset_utils import *
# from src.functions.utilities import os.path.basename, make_dynamic_rs_transform, MyCustomDataset
from PIL import Image
from skimage.transform import resize

# Global variable for image sampler so we load it only once.
global_image_sampler = None
dataRoot_global = None
model_architecture_global = None
training_instance_global = None
snip_index_global = None

def np_image_to_base64(im_matrix):
    im = Image.fromarray((256 * np.asarray(im_matrix)).astype(np.uint8))
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/jpeg;base64," + encoded_image


def parse_model_options(dataRoot):
    model_dirs = [m for m in glob.glob(os.path.join(dataRoot, "*VAE*")) if os.path.isdir(m)]
    model_list = []
    for m in model_dirs:
        model_name = os.path.basename(m)
        training_dirs = [os.path.basename(tr) for tr in glob.glob(os.path.join(dataRoot, model_name, "*VAE*"))
                         if os.path.isdir(os.path.join(dataRoot, model_name, tr, "figures"))]
        valid_trainings = []
        for tr in training_dirs:
            csv_path = os.path.join(dataRoot, model_name, tr, "figures", "umap_df.csv")
            try:
                pd.read_csv(csv_path)
                valid_trainings.append(tr)
            except Exception:
                continue
        if valid_trainings:
            model_list.append(model_name)
    return model_list


def get_image_sampler(train_dir):
    data_transform = make_dynamic_rs_transform()
    return MyCustomDataset(root=os.path.join(train_dir, "images"), transform=data_transform, return_name=True)


def load_latents_dataset(dataRoot, model_architecture, training_instance):
    umap_csv = os.path.join(dataRoot, model_architecture, training_instance, "figures", "umap_df.csv")
    df = pd.read_csv(umap_csv, index_col=0)
    df["experiment_date"] = df["experiment_date"].astype(str)
    image_sampler = get_image_sampler(dataRoot)

    dirname = os.path.dirname(image_sampler[0]["label"][0])

    snip_index = np.asarray(sorted([entry.name.replace(".jpg", "") for entry in os.scandir(dirname)
                   if entry.is_file() and entry.name.lower().endswith('.jpg')]))
    # snip_index = np.asarray([os.path.basename(image_sampler[i]["label"][0]).replace(".jpg", "") for i in range(len(image_sampler))])
    return {"df": df, "image_sampler": image_sampler, "snip_index": snip_index}


def create_figure(df, label_col="predicted_stage_hpf", plot_partition="biological", dim="3D UMAP",
                  pert_class_list=None, exp_date_list=None):
    if pert_class_list:
        plot_df = df[df["short_pert_name"].isin(pert_class_list)]
    else:
        plot_df = df

    if exp_date_list:
        plot_df = plot_df[df["experiment_date"].isin(exp_date_list)]
    else:
        plot_df = plot_df

    if dim == "3D UMAP":
        plot_dict = dict({"all": ["UMAP_00_3", "UMAP_01_3", "UMAP_02_3"],
                          "biological": ["UMAP_00_bio_3", "UMAP_01_bio_3", "UMAP_02_bio_3"],
                          "non-biological": ["UMAP_00_n_3", "UMAP_01_n_3", "UMAP_02_n_3"]})

        plot_variables = plot_dict[plot_partition]
        fig = px.scatter_3d(plot_df, x=plot_variables[0], y=plot_variables[1], z=plot_variables[2],
                            color=label_col, opacity=0.5,
                            color_continuous_scale="magma" if label_col == "predicted_stage_hpf" else "plotly",
                            custom_data = ["snip_id"]
                            )
        fig.update_layout(scene=dict(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            zaxis_title="UMAP 3"
        ))
    else:
        plot_dict = dict({"all": ["UMAP_00_2", "UMAP_01_2"],
                          "biological": ["UMAP_00_bio_2", "UMAP_01_bio_2"],
                          "non-biological": ["UMAP_00_n_2", "UMAP_01_n_2"]})

        plot_variables = plot_dict[plot_partition]

        fig = px.scatter(plot_df, x=plot_variables[0], y=plot_variables[1],
                         color=label_col, opacity=0.5,
                         color_continuous_scale="magma" if label_col == "predicted_stage_hpf" else "plotly")
        fig.update_layout(xaxis_title="UMAP 1", yaxis_title="UMAP 2")
    fig.update_traces(marker=dict(size=4))
    return fig


# Helper: retrieve an image for a given index from the image sampler.
def get_image_from_index(image_sampler, idx):
    # Assumes image_sampler is indexable.
    im_array = np.squeeze(np.asarray(image_sampler[idx][0]).tolist()[0])
    return im_array


def create_app(dataRoot, model_architecture, training_instance):
    global global_image_sampler, dataRoot_global, model_architecture_global, training_instance_global, snip_index_global
    dataRoot_global = dataRoot
    model_architecture_global = model_architecture
    training_instance_global = training_instance

    dataset = load_latents_dataset(dataRoot, model_architecture, training_instance)
    df = dataset["df"]
    global_image_sampler = dataset["image_sampler"]
    snip_index_global = dataset["snip_index"]

    # Store the DataFrame as JSON-serializable dictionary (records format)
    df_store = df.to_dict("records")

    # Get unique perturbation options
    pert_options = sorted(df["short_pert_name"].unique().tolist())

    # make label vec. Check to see if temperature is included
    label_options = ["predicted_stage_hpf", "short_pert_name", "experiment_date"]
    if "temp" in df.columns.tolist():
        label_options += ["temp"]
    # check to see if we have PCA columns
    pca_cols = [col for col in df.columns.tolist() if "PCA" in col]
    dim_option_vec = ["2D UMAP", "3D UMAP"]
    if len(pca_cols) >= 3:
        dim_option_vec += ["2D PCA", "3D PCA"]

    # Define dropdown options
    label_opts = [{"label": l, "value": l} for l in label_options]
    partition_opts = [{"label": p, "value": p} for p in ["all", "biological", "non-biological"]]
    dim_opts = [{"label": d, "value": d} for d in dim_option_vec]
    model_opts = [{"label": m, "value": m} for m in parse_model_options(dataRoot)]

    training_dirs = [os.path.basename(tr) for tr in glob.glob(os.path.join(dataRoot, model_architecture, "*VAE*"))
                     if os.path.isdir(os.path.join(dataRoot, model_architecture, tr, "figures"))]
    training_opts = [{"label": tr, "value": tr} for tr in training_dirs]

    app = Dash(__name__)
    app.layout = html.Div([
        # Store for the DataFrame
        dcc.Store(id="data-store", data=df_store),
        html.Div([
            # Left column (menus)
            html.Div([
                html.Label("Perturbation Type"),
                dcc.Dropdown(
                    id="perturbation-dropdown",
                    multi=True,
                    placeholder="Select experiment dates...",
                    value=["wt_ab"]
                ),
                html.Br(),
                html.Label("Experiment Date"),
                dcc.Dropdown(
                    id="experiment-date-dropdown",
                    multi=True,
                    placeholder="Select experiment dates..."
                ),
                html.Br(),
                html.Label("Label Column"),
                dcc.Dropdown(id="label-dropdown", options=label_opts, value="predicted_stage_hpf"),
                html.Br(),
                html.Label("Partition"),
                dcc.Dropdown(id="partition-dropdown", options=partition_opts, value="biological"),
                html.Br(),
                html.Label("Dimension"),
                dcc.Dropdown(id="dim-dropdown", options=dim_opts, value="3D UMAP"),
                html.Br(),
                html.Label("Model Architecture"),
                dcc.Dropdown(id="model-dropdown", options=model_opts, value=model_architecture),
                html.Br(),
                html.Label("Training Instance"),
                dcc.Dropdown(id="training-dropdown", options=training_opts, value=training_instance),
                html.Br(),
                dcc.Input(id="key-input", type="text", style={"display": "none"})
            ], style={
                "width": "20%",
                "display": "flex",
                "flexDirection": "column",
                "padding": "10px",
                "boxSizing": "border-box"
            }),
            # Right column (plot)
            html.Div([
                dcc.Graph(
                    id="plot-graph",
                    figure=create_figure(df, pert_class_list=[pert_options[0]]),
                    style={"width": "100%", "height": "600px"}
                ),
                dcc.Tooltip(id="graph-tooltip-5", direction="bottom")
            ], style={
                "width": "80%",
                "padding": "10px",
                "boxSizing": "border-box"
            })
        ], style={"display": "flex", "flexDirection": "row"})
    ])

    @app.callback(
        [Output("perturbation-dropdown", "options"),
         Output("experiment-date-dropdown", "options")],
        [Input("perturbation-dropdown", "value"),
         Input("experiment-date-dropdown", "value"),
         Input("data-store", "data")]
    )
    def update_linked_options(selected_perts, selected_dates, stored_data):
        df = pd.DataFrame(stored_data)
        # Determine valid experiment dates based on selected perturbations.
        if selected_perts:
            valid_dates = sorted(df[df["short_pert_name"].isin(selected_perts)]["experiment_date"].unique())
        else:
            valid_dates = sorted(df["experiment_date"].unique())
        # Determine valid perturbations based on selected experiment dates.
        if selected_dates:
            valid_perts = sorted(df[df["experiment_date"].isin(selected_dates)]["short_pert_name"].unique())
        else:
            valid_perts = sorted(df["short_pert_name"].unique())
        pert_options = [{"label": p, "value": p} for p in valid_perts]
        date_options = [{"label": d, "value": d} for d in valid_dates]
        return pert_options, date_options

    @app.callback(
        Output('training-dropdown', 'options'),
        Input('model-dropdown', 'value')
    )
    def update_training_options(selected_model):
        training_dirs = [os.path.basename(tr) for tr in glob.glob(os.path.join(dataRoot, selected_model, "*VAE*"))
                         if os.path.isdir(os.path.join(dataRoot, selected_model, tr, "figures"))]
        return [{"label": tr, "value": tr} for tr in training_dirs]

    @app.callback(
        Output('plot-graph', 'figure'),
        [
            Input('partition-dropdown', 'value'),
            Input('label-dropdown', 'value'),
            Input('dim-dropdown', 'value'),
            Input('perturbation-dropdown', 'value'),
            Input('experiment-date-dropdown', 'value'),
            Input('model-dropdown', 'value'),
            Input('training-dropdown', 'value')
        ]
    )
    def update_figure(partition, label_col, dim, pert_list, exp_list, selected_model, selected_training):
        new_dataset = load_latents_dataset(dataRoot, selected_model, selected_training)
        df_new = new_dataset["df"]
        # Update stored data (could also update dcc.Store if desired)
        return create_figure(df_new, label_col, partition, dim, pert_list, exp_list)

    @app.callback(
        [Output("graph-tooltip-5", "show"),
         Output("graph-tooltip-5", "bbox"),
         Output("graph-tooltip-5", "children")],
        [Input("plot-graph", "clickData"),
         Input("plot-graph", "relayoutData"),
         Input("key-input", "value"),
         Input("data-store", "data")]
    )
    def update_tooltip(clickData, relayoutData, key_value, stored_data, disp_dims=None):

        if disp_dims is None:
            disp_dims = tuple([144, 64])
        # Hide tooltip if a relayout event (e.g. rotating the plot) is detected.
        if relayoutData and any("scene.camera" in k for k in relayoutData.keys()):
            return False, no_update, no_update

        # Hide tooltip if a designated keystroke is detected (e.g. spacebar)
        if key_value and key_value.strip() == "":
            return False, no_update, no_update

        if clickData is None or stored_data is None:
            return False, no_update, no_update

        # Convert stored data back to a DataFrame.
        df_hover = pd.DataFrame(stored_data)
        point = clickData["points"][0]
        bbox = point.get("bbox", None)
        # idx = point.get("pointNumber", None)
        # if idx is None or idx >= df_hover.shape[0]:
        #     return False, no_update, no_update

        snip_id = point["customdata"][0]
        df_idx = np.where(df_hover["snip_id"] == snip_id)[0]
        row = df_hover.iloc[df_idx]
        # row = df_hover.iloc[idx]
        # snip_id = row["snip_id"]
        sampler_idx = np.where(snip_index_global == snip_id)[0]
        if len(sampler_idx) > 0:
            im_matrix = get_image_from_index(global_image_sampler, sampler_idx[0])
            # im_matrix = resize(im_matrix, disp_dims)
        else:
            raise Exception(f"Image not found for {snip_id}")
        im_url = np_image_to_base64(im_matrix)
        tooltip_children = html.Div([
            html.Img(src=im_url, style={"width": "100px", "display": "block", "margin": "0 auto"}),
            html.P(f"{np.round(row['predicted_stage_hpf'].values[0], 1)} hpf | {row['short_pert_name'].values[0]} | {row['experiment_date'].values[0]}"
                   , style={"font-weight": "bold"})
        ])
        return True, bbox, tooltip_children

    return app


if __name__ == '__main__':
    root = "/Users/nick/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/morphseq/training_data/20241107_ds/"
    model_arch = "SeqVAE_z100_ne150_sweep_01_block01_iter030"
    training_inst = "SeqVAE_training_2024-11-11_15-45-40"
    app = create_app(root, model_arch, training_inst)
    app.run_server(debug=True, port=np.random.randint(1000, 9999, 1))
