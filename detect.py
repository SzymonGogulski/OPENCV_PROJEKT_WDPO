import json
from pathlib import Path
from typing import Dict
import click
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict
import os


def detect_and_cut(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cutouts = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:
            if w > h:
                h = w
            else:
                w = h

            cutout = image[y-10:y+h+20, x-10:x+w+20]
            cutouts.append(cutout)

    return cutouts


def resize_images(images: list, size: tuple):

    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, size)
        resized_images.append(resized_image)
    return resized_images


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def predict(images: list, model):

    predictions = np.array([])
    predictions = predictions.reshape((0, 5))
    for image in images:
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predictions = np.concatenate((predictions, prediction), axis=0)
    return predictions


def count_occurrences(input_list):
    occurrences = defaultdict(int)

    for item in input_list:
        occurrences[item] += 1

    result_list = [{key: value} for key, value in occurrences.items()]
    return result_list
  

def decode_predictions(predictions: np.array):
    labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        labels.append(label)

    switchcase = {
        0: "Aspen",
        1: "Birch",
        2: "Hazel",
        3: "Maple",
        4: "Oak",
    }

    labels = [switchcase.get(label, "Invalid") for label in labels]
    # ['Birch', 'Aspen', 'Hazel', 'Birch', 'Aspen']
    return labels


def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        A dictionary with the number of each object.
    """

    # Rozmiar zdjęcia i ścieżka do modelu
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_directory, 'model_tf')
    input_shape = (160,160)


    # Wczytanie zdjęcia
    leafs = detect_and_cut(img_path)
    are_all_ndarrays = all(isinstance(elem, np.ndarray) for elem in leafs)

    if (len(leafs) == 0 or are_all_ndarrays == False):
        return {'aspen': 0, 'birch': 0, 'hazel': 0, 'maple': 0, 'oak': 0}

    try:
        leafs = resize_images(leafs, input_shape)
    except cv2.error:
        return {'aspen': 0, 'birch': 0, 'hazel': 0, 'maple': 0, 'oak': 0}
    


    # Wczytanie modelu
    model = load_model(model_path)

    # Predykcja
    prediction = predict(leafs, model)

    # Dodanie etykiet do predykcji
    labels = decode_predictions(prediction)

    # policz wyniki
    result = count_occurrences(labels)
    
    aspen = birch = hazel = maple = oak = 0
    
    for item in result:
        key, value = next(iter(item.items()))
        if key == 'Aspen':
            aspen = value
        elif key == 'Birch':
            birch = value
        elif key == 'Hazel':
            hazel = value
        elif key == 'Maple':
            maple = value
        elif key == 'Oak':
            oak = value

    return {'aspen': aspen, 'birch': birch, 'hazel': hazel, 'maple': maple, 'oak': oak}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()












