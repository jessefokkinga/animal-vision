import base64
import math
from io import BytesIO

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

matplotlib.use("Agg")


def make_prediction(image, model, class_names):
    image = load_and_prep_image(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)

    model = keras.models.load_model("models/" + model + ".h5")

    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = int(tf.reduce_max(preds[0]) * 100)

    return pred_class, pred_conf


def load_and_prep_image(img, img_shape=224, rescale=False):

    img = np.array(img)
    img = tf.image.resize(img, [img_shape, img_shape])

    if rescale:
        return img / 255.0
    else:
        return img


def explain_prediction(model, img, classes, top_preds_count):
    classes.sort()
    img = load_and_prep_image(img)

    img = np.array(img)
    model = keras.models.load_model("models/" + model + ".h5")

    explainer = lime_image.LimeImageExplainer(verbose=False)
    image_columns = 3
    image_rows = math.ceil(top_preds_count / image_columns)

    explanation = explainer.explain_instance(
        img,
        classifier_fn=model.predict,
        top_labels=100,
        hide_color=0,
        num_samples=1000,
    )

    preds = model.predict(np.expand_dims(img, axis=0))
    top_preds_indexes = np.flip(np.argsort(preds))[0, :top_preds_count]
    top_preds_values = preds.take(top_preds_indexes)
    top_preds_names = np.vectorize(lambda x: classes[x])(top_preds_indexes)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        image_rows, image_columns, figsize=(image_columns * 5, image_rows * 5)
    )
    [ax.set_axis_off() for ax in axes.flat]

    for i, (index, value, name, ax) in enumerate(
        zip(top_preds_indexes, top_preds_values, top_preds_names, axes.flat)
    ):
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[i],
            positive_only=False,
            num_features=3,
            hide_rest=False,
            min_weight=1 / 25,
        )

        ax.imshow(mark_boundaries(temp / 255, mask))

    fig = fig_to_uri(fig)
    return fig


def fig_to_uri(in_fig, close_all=True, **save_args):

    out_img = BytesIO()
    in_fig.savefig(out_img, format="png", **save_args)
    if close_all:
        in_fig.clf()
        plt.close("all")
    out_img.seek(0)

    encoded = (
        base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    )

    # Assuming base64_str is the string value without 'data:image/jpeg;base64,'
    im = Image.open(BytesIO(base64.decodebytes(bytes(encoded, "utf-8"))))

    # Crop image
    im = im.crop((185, 80, 535, 425))

    # Convert back to base64 string
    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_str = (
        base64.b64encode(buffered.getvalue()).decode("ascii").replace("\n", "")
    )

    return "data:image/png;base64,{}".format(img_str)
