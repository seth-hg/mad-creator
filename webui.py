import argparse
import tempfile

import numpy as np
import streamlit as st
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_select import image_select

from match import get_candidates


@st.cache_resource
def load_model(model_path):
    print("loading model:", model_path)
    p = pipeline(
        task=Tasks.multi_modal_embedding,
        model=model_path,
        model_revision="v1.0.1",
    )
    return p


def animate(selected, duration):
    images = []
    idx = 0
    for text, cand in selected:
        img = Image.open(cand)
        img = img.resize((int(img.width / 2), int(img.height / 2)))
        draw = ImageDraw.Draw(img)

        # 设置字体和大小
        font = "font/LXGWWenKai-Bold.ttf"
        font_size = 14
        font = ImageFont.truetype(font=font, size=font_size)
        # 计算文字的位置（居中）
        x = (img.width - len(text) * font_size) / 2
        y = img.height - font_size * 2

        # 在图片上添加文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        images.append(img)
        # img.save(f"{idx}.jpg")
        idx += 1
    img = images[0]
    output_file = tempfile.mktemp(".gif", "mad-", ".")
    img.save(
        fp=output_file,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=duration,
        loop=0,
    )
    return output_file


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
        "--top_k",
        type=int,
        nargs="?",
        default=5,
        help="get top_k most related image clusters",
    )
    parser.add_argument(
        "--duration",
        type=int,
        nargs="?",
        default=1500,
        help="how many seconds each frame lasts in the generated GIF",
    )

    args = parser.parse_args()

    # load index
    image_files, image_embeds, centers, clusters = np.load(args.index).values()

    st.set_page_config(
        page_title="MAD Creator",
        layout="wide",
    )

    model = load_model(args.model)

    candidates = st.session_state.get("candidates", None)
    if candidates is None:
        text = st.text_area(
            "input text",
            "",
            label_visibility="hidden",
            height=300,
        )

        input_texts = [t for t in text.split("\n") if t != ""]
        if st.button("confirm", disabled=(len(input_texts) == 0)):
            candidates = get_candidates(
                model,
                input_texts,
                args.top_k,
                image_files,
                image_embeds,
                centers,
                clusters,
            )
            st.session_state["candidates"] = candidates
            st.rerun()
    else:
        selected = []
        for cand in candidates:
            line, images, scores = cand
            img = image_select(line, images, [f"{s:.4f}" for s in scores])
            selected.append((line, img))
        if st.button("create"):
            output_file = animate(selected, args.duration)
            st.image(output_file, "Result")
