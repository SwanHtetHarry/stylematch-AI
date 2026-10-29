import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# =========================================================
# IMAGE EMBEDDING
# =========================================================

def get_image_embedding(
    image_path,
    clip_model,
    clip_processor,
    device
):
    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():

        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        embedding = clip_model.visual_projection(
            pooled_output
        )

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True
        )

    return embedding.cpu().numpy()


# =========================================================
# NORMAL TEXT SEARCH
# =========================================================

def search_products(
    query,
    clip_model,
    clip_processor,
    device,
    image_embeddings,
    products_df,
    top_k=5
):
    text_inputs = clip_processor(
        text=[query],
        padding=True,
        return_tensors="pt"
    )

    input_ids = text_inputs["input_ids"].to(device)

    attention_mask = text_inputs[
        "attention_mask"
    ].to(device)

    with torch.inference_mode():

        text_outputs = clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_text = text_outputs.pooler_output

        text_embedding = clip_model.text_projection(
            pooled_text
        )

        text_embedding = (
            text_embedding
            / text_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

    text_embedding = text_embedding.cpu().numpy()

    similarities = (
        image_embeddings
        @ text_embedding.T
    ).squeeze()

    top_indices = (
        similarities
        .argsort()[::-1][:top_k]
    )

    results = products_df.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarities[
        top_indices
    ]

    return results


# =========================================================
# FILTERED TEXT SEARCH
# =========================================================

def search_products_filtered(
    query,
    clip_model,
    clip_processor,
    device,
    image_embeddings,
    products_df,
    top_k=5,
    gender=None,
    category=None,
    colour=None,
    usage=None
):
    # -------------------------
    # Convert query to CLIP embedding
    # -------------------------

    text_inputs = clip_processor(
        text=[query],
        padding=True,
        return_tensors="pt"
    )

    input_ids = text_inputs["input_ids"].to(device)

    attention_mask = text_inputs[
        "attention_mask"
    ].to(device)

    with torch.inference_mode():

        text_outputs = clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_text = text_outputs.pooler_output

        text_embedding = clip_model.text_projection(
            pooled_text
        )

        text_embedding = (
            text_embedding
            / text_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

    text_embedding = text_embedding.cpu().numpy()


    # -------------------------
    # Similarity scores
    # -------------------------

    similarities = (
        image_embeddings
        @ text_embedding.T
    ).squeeze()


    # -------------------------
    # Start with all products
    # -------------------------

    mask = np.ones(
        len(products_df),
        dtype=bool
    )


    # -------------------------
    # Apply filters
    # -------------------------

    if gender is not None:
        mask &= (
            products_df["gender"] == gender
        ).to_numpy()

    if category is not None:
        mask &= (
            products_df["masterCategory"] == category
        ).to_numpy()

    if colour is not None:
        mask &= (
            products_df["baseColour"] == colour
        ).to_numpy()

    if usage is not None:
        mask &= (
            products_df["usage"] == usage
        ).to_numpy()


    # -------------------------
    # Get products that passed filters
    # -------------------------

    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return products_df.iloc[[]].copy()


    # -------------------------
    # Rank only filtered products
    # -------------------------

    filtered_scores = similarities[
        valid_indices
    ]

    ranked_positions = (
        filtered_scores
        .argsort()[::-1][:top_k]
    )

    top_indices = valid_indices[
        ranked_positions
    ]


    # -------------------------
    # Results
    # -------------------------

    results = products_df.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarities[
        top_indices
    ]

    return results


# =========================================================
# IMAGE SEARCH
# =========================================================

def search_by_image(
    image_path,
    clip_model,
    clip_processor,
    device,
    image_embeddings,
    products_df,
    top_k=5
):
    query_embedding = get_image_embedding(
        image_path,
        clip_model,
        clip_processor,
        device
    )

    similarities = (
        image_embeddings
        @ query_embedding.T
    ).squeeze()

    top_indices = similarities.argsort()[::-1]

    filtered_indices = []

    for idx in top_indices:

        candidate_path = products_df.iloc[
            idx
        ]["image_path"]

        # Remove exact same image
        if str(candidate_path) != str(image_path):
            filtered_indices.append(idx)

        if len(filtered_indices) == top_k:
            break

    results = products_df.iloc[
        filtered_indices
    ].copy()

    results["similarity"] = similarities[
        filtered_indices
    ]

    return results


# =========================================================
# DISPLAY RESULTS
# =========================================================

def show_search_results(results):

    if len(results) == 0:
        print("No matching products found.")
        return

    plt.figure(
        figsize=(15, 4)
    )

    for i, (_, row) in enumerate(
        results.iterrows()
    ):
        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        plt.subplot(
            1,
            len(results),
            i + 1
        )

        plt.imshow(image)
        plt.axis("off")

        plt.title(
            f"{row['productDisplayName']}\n"
            f"Score: {row['similarity']:.3f}",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()