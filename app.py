import io
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = None


def load_model():
    # load the Resenet model pre-trained on the imagenet dataset
    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}
    if request.method == "POST":
        if request.files["image"]:
            image = request.files["image"].read()
            print(request.files["imge"])
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(224, 224))
            print(image)
            prediction = model.predict(image)
            results = imagenet_utils.decode_predictions(prediction)
            data["predictions"] = []

            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            data["success"] = True
    return jsonify(data)


if __name__ == '__main__':
    if model is None:
        load_model()
    app.run()
