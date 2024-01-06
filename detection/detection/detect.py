import cv2 
import numpy as np
import tensorflow as tf
np.set_printoptions(precision=4, suppress=True)

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

def predict_one(image, model):
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)

def predict(images: list, model):

    predictions = np.array([])
    predictions = predictions.reshape((0, 5))
    for image in images:
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predictions = np.concatenate((predictions, prediction), axis=0)
    return predictions
    
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

    return labels

def add_labels_to_images(images: list, labels: list):
    enum=0
    for image, label in zip(images, labels):
        enum+=1
        cv2.putText(image, str(enum) + label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)




if __name__ == "__main__":
    
    # Rozmiar zdjęcia i ścierzka do modelu
    input_shape = (160,160)
    model_path = './model_tf'

    # Wczytanie zdjęcia
    leafs = detect_and_cut('data/0002.jpg')
    leafs = resize_images(leafs, input_shape)

    # Wczytanie modelu
    model = load_model('./model_tf')

    # Predykcja
    prediction = predict(leafs, model)

    # Dodanie etykiet do predykcji
    labels = decode_predictions(prediction)

    # Dodanie etykiet do zdjęć
    add_labels_to_images(leafs, labels)

    for leaf in leafs:
        cv2.imshow('leaf', leaf)
        cv2.waitKey(0)

    print("Success!")