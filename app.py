import os
import tempfile
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import pandas as pd

from transformers import CLIPModel, CLIPProcessor

from src.search_engine import (
    search_products_filtered,
    search_by_image
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StyleMatch AI",
    page_icon="👕",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    .product-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .product-name {
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }

    .product-info {
        font-size: 0.82rem;
        color: #888;
    }

    .score {
        font-size: 0.80rem;
        margin-top: 4px;
    }

    .section-space {
        margin-top: 2rem;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 4rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">StyleMatch AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered fashion discovery using natural language
    and visual similarity search.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).parent

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "image_embeddings_10000.npy"
)

PRODUCTS_PATH = (
    PROJECT_ROOT
    / "embeddings"
    / "products_10000.csv"
)

IMAGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "catalog_images"
)


# =========================================================
# LOAD CLIP
# =========================================================

@st.cache_resource
def load_clip():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name = "openai/clip-vit-base-patch32"

    model = CLIPModel.from_pretrained(
        model_name
    ).to(device)

    processor = CLIPProcessor.from_pretrained(
        model_name
    )

    model.eval()

    return model, processor, device


# =========================================================
# LOAD CATALOG
# =========================================================

@st.cache_data
def load_catalog():

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    products = pd.read_csv(
        PRODUCTS_PATH
    )

    return embeddings, products


# =========================================================
# LOAD MODEL + DATA
# =========================================================

with st.spinner("Loading StyleMatch AI..."):

    clip_model, clip_processor, device = load_clip()

    image_embeddings, products_df = load_catalog()


# =========================================================
# PRODUCT DISPLAY
# =========================================================

def display_products(results):

    if len(results) == 0:

        st.warning(
            "No matching products found."
        )

        return

    columns = st.columns(
        len(results)
    )

    for column, (_, row) in zip(
        columns,
        results.iterrows()
    ):

        image_path = (
            IMAGES_PATH
            / f"{int(row['id'])}.jpg"
        )

        with column:

            st.markdown(
                '<div class="product-card">',
                unsafe_allow_html=True
            )

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

            st.markdown(
                f"""
                <div class="product-name">
                {row['productDisplayName']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="product-info">
                {row['baseColour']} · {row['usage']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="score">
                Similarity: {row['similarity']:.3f}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# =========================================================
# TABS
# =========================================================

text_tab, image_tab = st.tabs(
    [
        "🔎 Text Search",
        "📷 Image Search"
    ]
)


# =========================================================
# TEXT SEARCH
# =========================================================

with text_tab:

    st.subheader(
        "Search with natural language"
    )

    st.write(
        "Describe the fashion item you want to find."
    )


    query = st.text_input(
        "Search query",
        placeholder="Example: black casual shoes"
    )


    # -----------------------------------------------------
    # FILTERS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    gender_options = [
        "All"
    ] + sorted(
        products_df["gender"]
        .dropna()
        .unique()
        .tolist()
    )


    category_options = [
        "All"
    ] + sorted(
        products_df["masterCategory"]
        .dropna()
        .unique()
        .tolist()
    )


    colour_options = [
        "All"
    ] + sorted(
        products_df["baseColour"]
        .dropna()
        .unique()
        .tolist()
    )


    usage_options = [
        "All"
    ] + sorted(
        products_df["usage"]
        .dropna()
        .unique()
        .tolist()
    )


    with col1:

        gender = st.selectbox(
            "Gender",
            gender_options
        )


    with col2:

        category = st.selectbox(
            "Category",
            category_options
        )


    with col3:

        colour = st.selectbox(
            "Colour",
            colour_options
        )


    with col4:

        usage = st.selectbox(
            "Usage",
            usage_options
        )


    top_k_text = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5,
        key="text_top_k"
    )


    if st.button(
        "Search Products",
        type="primary",
        use_container_width=True,
        key="text_search_button"
    ):

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            with st.spinner(
                "Searching products..."
            ):

                results = search_products_filtered(
                    query=query,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    device=device,
                    image_embeddings=image_embeddings,
                    products_df=products_df,
                    top_k=top_k_text,

                    gender=(
                        None
                        if gender == "All"
                        else gender
                    ),

                    category=(
                        None
                        if category == "All"
                        else category
                    ),

                    colour=(
                        None
                        if colour == "All"
                        else colour
                    ),

                    usage=(
                        None
                        if usage == "All"
                        else usage
                    )
                )


            st.markdown(
                '<div class="section-space"></div>',
                unsafe_allow_html=True
            )

            st.subheader(
                "Recommended Products"
            )

            display_products(
                results
            )


# =========================================================
# IMAGE SEARCH
# =========================================================

with image_tab:

    st.subheader(
        "Find visually similar products"
    )

    st.write(
        "Upload a fashion image and StyleMatch AI "
        "will search the catalog for similar items."
    )


    uploaded_file = st.file_uploader(
        "Upload product image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="image_upload"
    )


    top_k_image = st.slider(
        "Number of similar products",
        min_value=3,
        max_value=10,
        value=5,
        key="image_top_k"
    )


    if uploaded_file is not None:

        preview_col, info_col = st.columns(
            [1, 2]
        )


        with preview_col:

            st.image(
                uploaded_file,
                caption="Uploaded image",
                use_container_width=True
            )


        with info_col:

            st.info(
                "StyleMatch AI converts your image "
                "into a CLIP embedding and compares "
                "it against the product catalog."
            )


        if st.button(
            "Find Similar Products",
            type="primary",
            use_container_width=True,
            key="image_search_button"
        ):

            suffix = Path(
                uploaded_file.name
            ).suffix


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                temp_path = temp_file.name


            try:

                with st.spinner(
                    "Finding similar products..."
                ):

                    image_results = search_by_image(
                        image_path=temp_path,
                        clip_model=clip_model,
                        clip_processor=clip_processor,
                        device=device,
                        image_embeddings=image_embeddings,
                        products_df=products_df,
                        top_k=top_k_image
                    )


                st.markdown(
                    '<div class="section-space"></div>',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "Similar Products"
                )

                display_products(
                    image_results
                )


            finally:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader(
    "How StyleMatch AI Works"
)

how1, how2, how3 = st.columns(3)


with how1:

    st.markdown("### 1. Understand")

    st.write(
        "CLIP converts text queries or uploaded "
        "images into numerical embeddings."
    )


with how2:

    st.markdown("### 2. Compare")

    st.write(
        "The query embedding is compared with "
        "10,000 pre-computed fashion embeddings."
    )


with how3:

    st.markdown("### 3. Recommend")

    st.write(
        "Products with the highest cosine "
        "similarity are returned as recommendations."
    )


# =========================================================
# PROJECT INFO
# =========================================================

st.divider()

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Fashion Products",
        "10,000"
    )

with info2:
    st.metric(
        "Embedding Size",
        "512"
    )

with info3:
    st.metric(
        "Model",
        "CLIP ViT-B/32"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    StyleMatch AI · Multimodal Fashion Search Project
    </div>
    """,
    unsafe_allow_html=True
)