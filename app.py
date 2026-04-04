from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

FLOWER_IMAGES = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    flower_image = ""
    sepal_length = sepal_width = petal_length = petal_width = ""

    if request.method == "POST":
        sepal_length = request.form["sepal_length"]
        sepal_width = request.form["sepal_width"]
        petal_length = request.form["petal_length"]
        petal_width = request.form["petal_width"]

        input_data = np.array([[float(sepal_length), float(sepal_width),
                                float(petal_length), float(petal_width)]])

        result = model.predict(input_data)[0]
        flower_name = iris.target_names[result]
        flower_image = FLOWER_IMAGES[flower_name]

        if flower_name == "setosa":
            prediction = "🌸 Iris Setosa"
        elif flower_name == "versicolor":
            prediction = "🌺 Iris Versicolor"
        else:
            prediction = "🌼 Iris Virginica"

    return render_template("index.html",
                           prediction=prediction,
                           flower_image=flower_image,
                           sepal_length=sepal_length,
                           sepal_width=sepal_width,
                           petal_length=petal_length,
                           petal_width=petal_width)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)