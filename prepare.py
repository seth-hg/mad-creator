import argparse
import os

import numpy as np
import sklearn.cluster
import sklearn.preprocessing
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
from tqdm import tqdm


def load_image(path: str) -> Image:
    img = Image.open(path)
    img = img.resize((224, 224))
    return img.convert("RGB")


def embed_images(images, batch_size):
    embeds = []
    for i in tqdm(range(0, len(images), batch_size), desc="Embedding"):
        batch = [load_image(img) for img in images[i : i + batch_size]]
        batch_embeds = (
            pipeline.forward({"img": batch})["img_embedding"]
            / pipeline.model.temperature
        )
        embeds.append(batch_embeds.detach().cpu().numpy())
    return np.concatenate(embeds)


def cluster_embeds(embeds, threshold):
    normalized_embeds = sklearn.preprocessing.normalize(embeds, axis=1, norm="l2")
    birch = sklearn.cluster.Birch(threshold=threshold, n_clusters=None)
    clusters = birch.fit_predict(normalized_embeds)
    return birch.subcluster_centers_, clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create ann-benchmarks dataset from npz file."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        default="damo/multi-modal_clip-vit-base-patch16_zh",
        help="path to model",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="?",
        default="./images",
        help="directory for source images",
    )
    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        default="index.npz",
        help="output npz file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="?",
        default=0.3,
        help="threshold for Birch clustering",
    )

    args = parser.parse_args()

    # load model
    pipeline = pipeline(
        task=Tasks.multi_modal_embedding,
        model=args.model,
        model_revision="v1.0.1",
    )

    image_files = sorted(os.listdir(args.input))
    image_files = [f"{args.input}/{f}" for f in image_files if f.endswith(".jpg")]
    embeds = embed_images(image_files, 32)
    centers, clusters = cluster_embeds(embeds, args.threshold)

    np.savez(
        args.output,
        files=np.array(image_files),
        embeds=embeds,
        centers=centers,
        clusters=clusters,
    )
