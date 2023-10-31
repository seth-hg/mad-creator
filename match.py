import argparse

import numpy as np
import sklearn.preprocessing
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def match(text_embeds, centers, top_k):
    similarities = np.dot(text_embeds, centers.T)
    best = np.argpartition(-similarities, top_k + 1, axis=-1)[:, :top_k]
    return best


def reorder(text_embed, matched_clusters, clusters, image_embeds, image_files):
    matched_image_files = np.array(
        [image_files[clusters == c][0] for c in matched_clusters]
    )
    matched_image_embeds = np.array(
        [image_embeds[clusters == c][0] for c in matched_clusters]
    )
    matched_image_embeds = sklearn.preprocessing.normalize(
        matched_image_embeds, axis=1, norm="l2"
    )
    similarities = np.dot(text_embed, matched_image_embeds.T)
    image_and_similarities = [
        (matched_image_files[i], float(similarities[i]))
        for i in range(len(matched_image_files))
    ]
    image_and_similarities = sorted(
        image_and_similarities, key=lambda x: x[1], reverse=True
    )
    return image_and_similarities


def get_candidates(
    pipeline, input_texts, top_k, image_files, image_embeds, centers, clusters
):
    text_embeddings = pipeline.forward({"text": input_texts})["text_embedding"]
    text_embeddings = text_embeddings.detach().cpu().numpy()
    best_matches = match(text_embeddings, centers, top_k)
    result = []
    for i, text in enumerate(input_texts):
        matched_images = reorder(
            text_embeddings[i], best_matches[i], clusters, image_embeds, image_files
        )
        print(input_texts[i], matched_images)
        result.append(
            (text, [p[0] for p in matched_images], [p[1] for p in matched_images])
        )
    return result


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
        "--index",
        type=str,
        nargs="?",
        default="index.npz",
        help="prebuilt index of image embeddings",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="?",
        default="input.txt",
        help="input texts",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        nargs="?",
        default=5,
        help="get top_k most related image clusters",
    )

    args = parser.parse_args()

    with open(args.input) as f:
        input_texts = f.read().split("\n")
    input_texts = [t for t in input_texts if t != ""]

    files, image_embeds, centers, clusters = np.load(args.index).values()

    # load model
    pipeline = pipeline(
        task=Tasks.multi_modal_embedding,
        model=args.model,
        model_revision="v1.0.1",
    )

    results = get_candidates(
        pipeline, input_texts, args.top_k, files, image_embeds, centers, clusters
    )
    print(results)
