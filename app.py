import os
import plotly.express as px
import pandas as pd

from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

from complaint_topic_extractor import load, run

app = Flask(__name__)
app.secret_key = "reviewinsight-demo-key"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}

results = None


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    global results

    if request.method == "POST":

        if "dataset" not in request.files:
            flash("Please select a CSV file.")
            return redirect(url_for("index"))

        file = request.files["dataset"]

        if file.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Please upload a CSV file only.")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            text_col = request.form.get("text_col") or None
            rating_col = request.form.get("rating_col") or None

            df = load(
                filepath,
                text_col=text_col,
                rating_col=rating_col,
                n_max=5000
            )

            if len(df) < 10:
                raise ValueError(
                    "Not enough low-rated reviews for clustering."
                )

            df, summary, global_bigrams = run(df)

            # Save output reports
            df.drop(columns=["clean"]).to_csv(
                "reviews_with_clusters.csv", index=False
            )

            summary.drop(columns=["examples"]).to_csv(
                "theme_summary.csv", index=False
            )

            # Cluster chart
            cluster_fig = px.bar(
                summary,
                x="cluster",
                y="n_reviews",
                title="Customer Complaints by Cluster",
                labels={
                    "cluster": "Cluster",
                    "n_reviews": "Number of Reviews"
                }
            )

            cluster_chart = cluster_fig.to_html(
                full_html=False,
                include_plotlyjs="cdn"
            )

            # Bigram chart
            bigram_df = pd.DataFrame(
                global_bigrams,
                columns=["Bigram", "Frequency"]
            )

            bigram_fig = px.bar(
                bigram_df.head(10),
                x="Frequency",
                y="Bigram",
                orientation="h",
                title="Top 10 Frequent Bigrams"
            )

            bigram_chart = bigram_fig.to_html(
                full_html=False,
                include_plotlyjs=False
            )

            results = {
                "total": len(df),
                "clusters": summary.to_dict("records"),
                "bigrams": global_bigrams,
                "cluster_chart": cluster_chart,
                "bigram_chart": bigram_chart
            }

        except Exception as e:
            flash(f"Analysis failed: {str(e)}")
            return redirect(url_for("index"))

    return render_template("index.html", results=results)


@app.route("/download/<report>")
def download_report(report):

    reports = {
        "reviews": "reviews_with_clusters.csv",
        "themes": "theme_summary.csv"
    }

    if report not in reports:
        return "Invalid report", 404

    path = reports[report]

    if not os.path.exists(path):
        return "Run an analysis first.", 404

    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)