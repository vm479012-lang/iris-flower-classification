from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load dataset and train model
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    flower_name = ""

    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        input_data = np.array([[sepal_length, sepal_width,
                                petal_length, petal_width]])
        result = model.predict(input_data)[0]
        flower_name = iris.target_names[result]

        if flower_name == "setosa":
            prediction = "🌸 Iris Setosa"
        elif flower_name == "versicolor":
            prediction = "🌺 Iris Versicolor"
        else:
            prediction = "🌼 Iris Virginica"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)