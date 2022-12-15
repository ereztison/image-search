import sys, json

sys.path.append("src")
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from flask import Flask, request, send_from_directory, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from vecsim import SciKitIndex
import torch
import clip

from src import embedding

__dir__ = Path(__file__).absolute().parent
upload_dir = __dir__ / "upload"
data_dir = __dir__ / "data"
upload_dir.mkdir(exist_ok=True)
NUMBER_OF_RESULTS = 12
app = Flask(__name__)


@dataclass
class Recommendation:
    id: int
    image: str
    title: str
    highlight: bool
    distance: float


@dataclass
class Item:
    id: int
    image: str
    title: str


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('assets', 'favicon.ico')


@app.route('/assets/<path:path>')
def serve_assets(path):
    return send_from_directory('assets', path)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


def embed_image(image_path):
    return embedding.embed_image(model, preprocess, device, image_path)


def embed_text(text):
    return embedding.embed_text(model, device, text)


def get_recommendations(vec, n=NUMBER_OF_RESULTS):
    dists, ids = sim.search(vec, n)
    df_results = df[df["id"].isin(ids)]

    return [
        Recommendation(row["id"], row["primary_image"], row["title"], False, round(d * 100, 3))
        for d, (idx, row) in sorted(zip(dists, df_results.iterrows()))
    ]


@app.route('/imgsearch', methods=['POST','GET'])
def imgsearch():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(upload_dir/filename)

        vec = embed_image(upload_dir/filename)
        (upload_dir/filename).unlink()
        recs = get_recommendations(vec)

        return render_template('index.html', items=recs, recommendations=recs)
    else:
        return redirect(url_for('index'))


@app.route('/')
def index():
    recs = [
    ]
    return render_template('index.html', recommendations=recs)


@app.route('/txtsearch', methods=['POST'])
def txtsearch():
    txt = str(request.form.get('txt', ""))
    vec = embed_text(txt)
    recs = get_recommendations(vec)

    return render_template('results.html', recommendations=recs)


@app.after_request
def add_no_cache(response):
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Pragma"] = "no-cache"
    return response


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html")


if __name__ == "__main__":
    print("Loading data...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # read the ids sampled from the dataset
    with (data_dir/"clip_ids.json").open('r') as f:
        embedding_ids = json.load(f)

    # load the sampled dataset
    df = pd.read_parquet(data_dir/"product_images.parquet")
    df = df[df["primary_image"].str.endswith(".jpg") | df["primary_image"].str.endswith(".png")] \
        .rename(columns={"asin": "id"})
    df["title"] = df["title"].fillna("")
    df["has_emb"] = df["id"].isin(embedding_ids)
    df = df[df["has_emb"]]

    print("Indexing...")
    sim = SciKitIndex("cosine", 512)

    # load the embeddings (separated to texts and images files)
    txt_embedding = np.load(str(data_dir / "clip_txt_emb.npy"))
    sim.add_items(data=txt_embedding, ids=embedding_ids)
    img_embedding = np.load(str(data_dir / "clip_img_emb.npy"))
    sim.add_items(data=img_embedding, ids=embedding_ids)
    sim.init()
    
    print("Starting server...")
    app.run(port=8080, host='0.0.0.0', debug=True)
