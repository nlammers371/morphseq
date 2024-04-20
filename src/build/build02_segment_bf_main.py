import os
import torch
import numpy as np
from src.functions.core_utils_segmentation import Dataset, FishModel
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import glob
import ntpath
from tqdm import tqdm
from src.functions.utilities import path_leaf
import skimage.io as io

def apply_unet(root, model_name, n_classes, overwrite_flag=False, segment_list=None, im_dims=None, batch_size=64,
               n_workers=None, make_sample_figures=False, n_sample_figures=100):


    """
    :param root:
    :param model_name:
    :param n_classes:
    :param overwrite_flag:
    :param im_dims:
    :param batch_size:
    :param n_workers:
    :return:
    """

    print("Generating segmentation masks using " + model_name + "....")

    # extract key info about computational resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if n_workers is None:
        n_workers = os.cpu_count()

    np.random.seed(345)  # use random seed for consistency
    if im_dims == None:
        im_dims = [576, 320]  # NL: can I read this off of the loaded model object?

    # generate directory for model predictions
    path_to_labels = os.path.join(root, 'segmentation', model_name + '_predictions', '')

    if make_sample_figures:
        sample_fig_path = os.path.join(path_to_labels, "sample_figures")
        if not os.path.isdir(sample_fig_path):
            os.makedirs(sample_fig_path)

    # get list of images to classify
    path_to_images = os.path.join(root, 'stitched_FF_images', '*')
    if segment_list is None:
        project_list = sorted(glob.glob(path_to_images))
        project_list = [p for p in project_list if "ignore" not in p]
        project_list = [p for p in project_list if os.path.isdir(p)]
    else:
        project_list = [os.path.join(root, 'stitched_FF_images', p) for p in segment_list]

    # select subset of images to label
    image_path_list = []
    label_path_list = []
    exist_flags = []
    for ind, p in enumerate(project_list):
        im_list_temp = glob.glob(os.path.join(p, '*.png')) + glob.glob(os.path.join(p, '*.tif')) + glob.glob(os.path.join(p, '*.jpg'))
        image_path_list += im_list_temp

        _, project_name = ntpath.split(p)
        label_path_root = os.path.join(path_to_labels, project_name)
        if not os.path.isdir(label_path_root):
            os.makedirs(label_path_root)

        for imp in im_list_temp:
            _, tail = ntpath.split(imp)
            label_path = os.path.join(label_path_root, tail)
            label_path = label_path.replace(".png", ".jpg")
            label_path_list.append(label_path)
            exist_flags.append(os.path.isfile(label_path))

    # remove images with previously existing labels if overwrite_flag=False
    if not overwrite_flag:
        image_path_list = [image_path_list[e] for e in range(len(image_path_list)) if not exist_flags[e]]
        label_path_list = [label_path_list[e] for e in range(len(label_path_list)) if not exist_flags[e]]
        n_ex = np.sum(np.asarray(exist_flags) == 1)
        if n_ex > 0:
            print('Skipping ' + str(n_ex) + " previously segmented images. Set 'overwrite_flag=True' to overwrite")

    # generate data loader object
    im_dataset = Dataset(root, image_path_list, im_dims, num_classes=n_classes, predict_only_flag=True)
    im_dataloader = DataLoader(im_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=n_workers)

    # initialize instance of model
    model = FishModel("FPN",
                      "resnet34",
                      in_channels=3,
                      out_classes=n_classes)

    # load trained model weights
    model.load_state_dict(torch.load(os.path.join(root, 'segmentation_models', model_name), map_location=device))
    model = model.to(device)
    model.eval()

    if make_sample_figures:
        figure_indices = np.random.choice(range(len(im_dataset)), np.min([n_sample_figures, len(im_dataset)]))
    else:
        figure_indices = np.asarray([])

    
    # get predictions
    print("Classifying images...")
    with torch.no_grad():
        # data_list = list(im_dataloader)
        it = iter(im_dataloader)
        iter_i = 0
        for idx in tqdm(range(len(im_dataloader))):
            batch = next(it)
            im = batch["image"]
            im = im.to(device)
            # print('im')
            logits = model(im)
            # print('logits')
            pr_probs = logits.sigmoid()
            pr_max = torch.max(pr_probs, axis=1)

            lb_predicted = pr_max.indices + 2
            lb_predicted[pr_max.values < 0.5] = 1

            lb_predicted = lb_predicted / (n_classes+1) * 255
            lb_predicted = np.asarray(lb_predicted.cpu()).astype(np.uint8)  # convert to integer

            # write to file
            im_paths = batch["path"]
            for b in range(lb_predicted.shape[0]):
                lb_temp = np.squeeze(lb_predicted[b, :, :])
                im_path = im_paths[b]
                suffix = im_path.replace(path_to_images[:-1], "")
                out_path = os.path.join(path_to_labels, suffix + ".jpg")
                io.imsave(out_path, lb_temp, check_contrast=False)

                # make figure
                if iter_i in figure_indices:
                    im_plot = np.squeeze(im[b, 0, :, :].cpu())

                    s = im_plot.shape

                    plt.figure(figsize=(10, 5))
                    #
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.flipud(im_plot))

                    plt.subplot(1, 2, 2)
                    y, x = np.mgrid[0:s[0], 0:s[1]]
                    # plt.axes().set_aspect('equal', 'datalim')
                    plt.set_cmap(plt.gray())

                    # plt.pcolormesh(x, y, Image2_mask, cmap='jet')
                    plt.imshow(im_plot)

                    lb_plot = lb_temp / 255 * (n_classes+1)
                    lb_plot = lb_plot
                    plt.imshow(lb_plot, cmap='Set1', alpha=0.5, vmin=0, vmax=n_classes+1, interpolation='none')
                    # plt.axis([x.min(), x.max(), y.min(), y.max()])

                    plt.xlim([x.min(), x.max()])
                    plt.ylim([y.min(), y.max()])
                    plt.colorbar(
                        ticks=range(n_classes+1)
                    )
                    # save
                    im_name = path_leaf(suffix)
                    subfolder = suffix.replace(im_name, "")
                    out_path = os.path.join(sample_fig_path, subfolder)
                    if not os.path.isdir(out_path):
                        os.makedirs(out_path)

                    plt.savefig(os.path.join(out_path, im_name + '_prediction.jpg'))
                    plt.close()

                iter_i += 1





if __name__ == "__main__":
    root = "/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data"

    # first, apply classifier to identify living and dead embryos, as well bubbles
    n_classes = 2
    model_name = "unet_emb_v4_0050"
    make_sample_figures = True
    apply_unet(root, model_name, n_classes)

    # now apply bubble classifier
    n_classes = 1
    model_name = "unet_bubble_v0_0050"
    apply_unet(root, model_name, n_classes)

    # now apply yolk classifier
    n_classes = 1
    model_name = "unet_yolk_v0_0050"
    apply_unet(root, model_name, n_classes)

    # now apply classifier to flag out-of-focus embryos
    n_classes = 1
    model_name = "unet_focus_v2_0050"
    apply_unet(root, model_name, n_classes)
