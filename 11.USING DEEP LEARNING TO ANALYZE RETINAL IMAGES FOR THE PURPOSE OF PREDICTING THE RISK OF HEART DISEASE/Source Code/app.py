from flask import Flask, render_template, request, redirect, session, url_for
from db_config import getConnection
from predict import predict_image
import os
import pandas as pd
from model_utils import load_test_data
from model_utils import evaluate_models, generate_accuracy_graph, generate_confusion_matrices
from model_utils import generate_accuracy_graph

X, X_flat, y = load_test_data()



app = Flask(__name__)
app.secret_key = "secret"

UPLOAD_FOLDER = "static/uploads"

# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = getConnection()
        cursor = conn.cursor()

        query = "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)"
        cursor.execute(query,(name,email,password))

        conn.commit()

        cursor.close()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        conn = getConnection()
        cursor = conn.cursor()

        query = "SELECT * FROM users WHERE email=%s AND password=%s"
        cursor.execute(query,(email,password))

        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            return redirect("/dashboard")

        else:
            return render_template("login.html", error="Invalid Email or Password")

    return render_template("login.html")


# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


# ---------------- UPLOAD IMAGE & PREDICT ----------------
@app.route("/upload", methods=["GET","POST"])
def upload():

    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":

        file = request.files["image"]

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        result = predict_image(filepath)

        conn = getConnection()
        cursor = conn.cursor()

        query = """
        INSERT INTO predictions(user_id,image_path,result)
        VALUES(%s,%s,%s)
        """

        cursor.execute(query,(session["user_id"],filepath,result))

        conn.commit()

        cursor.close()
        conn.close()

        return render_template("result.html",result=result,image=filepath)

    return render_template("upload_image.html")



# ---------------- ACCURACY GRAPH ----------------
@app.route("/accuracy_graph")
def accuracy_graph():

    generate_accuracy_graph()

    return render_template("accuracy_graph.html")


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")

@app.route("/upload_dataset", methods=["GET","POST"])
def upload_dataset():

    if request.method=="POST":

        file=request.files["file"]

        path=os.path.join("dataset",file.filename)

        file.save(path)

        return render_template("upload_dataset.html",msg="Dataset Uploaded Successfully")

    return render_template("upload_dataset.html")
@app.route("/view_dataset")
def view_dataset():

    data=[]

    for label in ["normal","risk"]:

        folder=os.path.join("dataset",label)

        for img in os.listdir(folder):

            data.append({"image":img,"class":label})

    return render_template("view_dataset.html",data=data)
@app.route("/preprocess")
def preprocess():

    X, X_flat, y = load_test_data()

    total = len(X)

    normal = sum(y == 0)
    risk = sum(y == 1)

    return render_template(
        "preprocess.html",
        total=total,
        normal=normal,
        risk=risk
    )
@app.route("/run_algorithms")
def run_algorithms():

    results, _ = evaluate_models()

    return render_template(
        "run_algorithms.html",
        results=results
    )


@app.route("/compare_models")
def compare_models():

    results, _ = evaluate_models()

    return render_template(
        "compare_models.html",
        results=results
    )


@app.route("/confusion_matrix")
def confusion_matrix():

    generate_confusion_matrices()

    return render_template("confusion_matrix.html")


# ---------------- RUN APP ----------------
if __name__ == "__main__":

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)