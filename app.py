from flask import Flask, render_template, request, jsonify
import joblib
import os

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

# ==========================================
# Load Model
# ==========================================

MODEL_PATH = "iris_model.pkl"

model = joblib.load(MODEL_PATH)

# ==========================================
# Class Names
# ==========================================

classes = [
    "Setosa",
    "Versicolor",
    "Virginica"
]

flower_images = {
    "Setosa": "images/setosa.jpg",
    "Versicolor": "images/versicolor.jpg",
    "Virginica": "images/virginica.jpg"
}

# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")

# ==========================================
# Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        values = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]

        # Validation

        for value in values:

            if value <= 0:

                raise ValueError(
                    "All measurements must be greater than zero."
                )

        sample = [values]

        prediction = model.predict(sample)[0]

        probabilities = model.predict_proba(sample)[0]

        flower = classes[prediction]

        confidence = round(max(probabilities) * 100, 2)

        image = flower_images[flower]

        return render_template(
            "index.html",
            prediction=flower,
            confidence=confidence,
            image=image
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )

# ==========================================
# Health Check API
# ==========================================

@app.route("/health")
def health():

    return jsonify({

        "status": "healthy",

        "application": "Iris Flower Classification",

        "model_loaded": True

    })

# ==========================================
# Model Information API
# ==========================================

@app.route("/model-info")
def model_info():

    return jsonify({

        "Algorithm": "Random Forest Classifier",

        "Dataset": "Iris Dataset",

        "Features": 4,

        "Classes": classes

    })

# ==========================================
# About API
# ==========================================

@app.route("/about-api")
def about():

    return jsonify({

        "Project": "Iris Flower Classification System",

        "Framework": "Flask",

        "Machine Learning": "Scikit-Learn",

        "Language": "Python"

    })

# ==========================================
# Version API
# ==========================================

@app.route("/version")
def version():

    return jsonify({

        "Version": "1.0.0"

    })

# ==========================================
# Error Handling
# ==========================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "index.html",
        error="404 - Page Not Found"
    ), 404

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(
        "index.html",
        error="Internal Server Error"
    ), 500

# ==========================================
# Run App
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
    
