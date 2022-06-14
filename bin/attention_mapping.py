#!/usr/local/bin/python

import os
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.metrics import  BinaryAccuracy
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_definition import load_compile_model

def load_samples(image_paths, target_size=(224,224)):

    read_img = lambda image: load_img(image, target_size=target_size, color_mode='grayscale')

    images = [np.array(read_img(image)) for image in image_paths]

    images = np.asarray(images)

    images_array_new_shape = tuple([images.shape[0]] + list(target_size) + [1])

    images = images.reshape(images_array_new_shape)

    X = images/255.0

    return (images, X)

def get_attention_maps(model, model_modifier_function, samples, score):
    # pen_ultim = -1 # Required by GradCam and Saliency map

    # Create Saliency object.
    saliency = Saliency(
        model,
        model_modifier=model_modifier_function,
        clone=True
    )

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(
        score,
        samples,
        smooth_samples=100, # The number of calculating gradients iterations.
        smooth_noise=0.2 # noise spread level.
    )

    #######################################################################

    # Create GradCAM++ object
    gradcam = GradcamPlusPlus(
        model,
        model_modifier=model_modifier_function,
        clone=True
    )

    # Generate heatmap with GradCAM++
    cam = gradcam(
        score,
        samples,
        # penultimate_layer=pen_ultim
    )

    #######################################################################

    scorecam = Scorecam(
        model,
        model_modifier=model_modifier_function
    )

    # Generate heatmap with Faster-ScoreCAM
    s_cam = scorecam(
        score,
        samples,
        # penultimate_layer=pen_ultim,
        max_N=-1
    )

    #######################################################################

    maps = {
        "Saliency map": saliency_map,
        "GradCam++": cam,
        "ScoreCam": s_cam
    }

    return maps

def generate_heatmap(axes, image, att_map, i): # i is the image number in the stack of images
    axes.imshow(image, cmap= plt.cm.binary)
    heatmap = np.uint8(cm.jet(att_map[i])[..., :3] * 255)
    axes.imshow(heatmap, cmap='jet', alpha=0.6)
    axes.tick_params(
        axis='both', which='both',
        left=False, right=False,
        labelleft=False, bottom=False,
        top=False, labelbottom=False
    )

def plot_attention_maps(image_titles, images, attention_maps):
    fig = plt.figure(figsize=(9,9), dpi=600, constrained_layout=True)
    gs = fig.add_gridspec(nrows=len(image_titles), ncols=len(attention_maps), wspace=0.02, hspace=0.02)

    for i, title in enumerate(image_titles):

        for j, (map_name, att_map) in enumerate(attention_maps.items()):

            ax = fig.add_subplot(gs[i, j])
            if i == 0:
                ax.set_title(map_name, fontsize=12) # first row

            if map_name == "Original":
                ax.imshow(images[i], cmap="gray")
                ax.set_ylabel(title, rotation=90, fontsize=8)
                ax.tick_params(
                    axis='both', which='both',
                    left=False, right=False,
                    labelleft=False, bottom=False,
                    top=False, labelbottom=False
                )
            elif map_name == "Saliency map":
                generate_heatmap(ax, images[i], att_map, i)
            elif map_name == "GradCam++":
                generate_heatmap(ax, images[i], att_map, i)
            elif map_name == "ScoreCam": # Default to score cam
                generate_heatmap(ax, images[i], att_map, i)

    plt.savefig("attention_map.png")

# image_name_match = lambda entry, image_name: image_name if image_name.__contains__(entry) else None


def item_sorter(item):
    if item.__contains__("false"):
        return item[12:]
    else:
        return item[11:]

def main(path_to_model, csv_file, image_paths):

    model = load_compile_model(path_to_model)
    df = pd.read_csv(csv_file)

    image_paths.sort(key = lambda item: item_sorter(item))

    df.sort_values(by="Entry", inplace=True)

    df["Image path"] = image_paths

    virus_species = df["Species name"].tolist()
    protein_names = df["Protein"].tolist()

    replace2linear = ReplaceToLinear()

    images, processed_samples = load_samples(image_paths)

    probas = model.predict(processed_samples, verbose=0)
    predictions = (probas > 0.5).astype(np.float32).reshape((-1,)).tolist()

    df["Predictions"] = predictions

    classes = df["Infects human"]

    image_titles = list(
        map(
            lambda species_protein: f"{species_protein[0]}\n{species_protein[1]}\nClass: {species_protein[2]}\nPrediction: {species_protein[3]}",
            zip(virus_species, protein_names, classes, predictions)
        )
    )

    df.to_csv("with_predictions.csv", index=False)

    scores = BinaryScore(predictions)
    print("\n")
    print(predictions, end="\n\n")
    print("getting attention maps", end="\n\n")
    maps = get_attention_maps(model, replace2linear, processed_samples, scores)

    attention_maps = {"Original": images}

    attention_maps.update(maps)
    print("Starting plots")
    plot_attention_maps(image_titles, images, attention_maps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Zoonoses Model Attention maps',
                                     description=\
                                     """
                                     Attention mapping of samples using ScoreCam, GradCam++ and Saliency maps
                                     """.strip(),
                                     )

    parser.add_argument('-m', '--model_path',
                        required=True,
                        help='Main directory containing zoonosis data subsplit directories')

    parser.add_argument('-c', '--csv',
                        required=True,
                        help='Main directory containing zoonosis data subsplit directories')

    parser.add_argument('-i', '--images',
                        required=True, nargs='+',
                        help='Main directory containing zoonosis data subsplit directories')

    args = parser.parse_args()

    path_to_model = args.model_path

    csv_file = args.csv
    # print(csv_file)
    # print(path_to_model)
    image_paths = args.images
    # print(image_paths)
    main(path_to_model, csv_file, image_paths)

