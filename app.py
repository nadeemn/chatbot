from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

@app.get("/")
def index():
    """
    home page api returning the html template.
    """
    return render_template("base.html")


@app.post("/predict")
def predict():
    """
    predict method to accept the API request from frontend and respond back appropriate answer for the user query.
    """
    text = request.get_json().get("message")

    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
