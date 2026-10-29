# StyleMatch AI

StyleMatch AI is a multimodal fashion search and recommendation system built using **OpenAI CLIP**, PyTorch, and Streamlit.

The project allows users to search a fashion catalog in two different ways:

- Search using natural language, such as **"black casual shoes"**
- Upload an image and find visually similar fashion products

The application currently searches across a catalog of **10,000 fashion products**.

---

## Features

### Text-to-Product Search

Users can describe the fashion item they are looking for using natural language.

Example:

```text
black casual shoes
```

StyleMatch AI converts the text into a CLIP embedding and compares it with pre-computed product image embeddings.

The products with the highest cosine similarity are returned as recommendations.

---

### Image-to-Product Search

Users can upload a fashion product image.

StyleMatch AI converts the uploaded image into an embedding and searches the catalog for visually similar products.

This allows users to discover products based on appearance rather than keywords.

---

### Product Filters

Text search can also be refined using metadata filters:

- Gender
- Product category
- Colour
- Usage

Example:

```text
Query: casual shoes

Gender: Men
Category: Footwear
Colour: Black
Usage: Casual
```

---

### Adjustable Recommendations

Users can choose to display between **3 and 10 recommendations**.

---

## How It Works

StyleMatch AI uses the CLIP model:

```text
openai/clip-vit-base-patch32
```

CLIP can represent both text and images inside the same embedding space.

### Text Search

```text
Text Query
    ↓
CLIP Text Encoder
    ↓
512-Dimensional Embedding
    ↓
Cosine Similarity
    ↓
Product Image Embeddings
    ↓
Top Matching Products
```

### Image Search

```text
Uploaded Image
    ↓
CLIP Vision Encoder
    ↓
512-Dimensional Embedding
    ↓
Cosine Similarity
    ↓
Product Image Embeddings
    ↓
Top Similar Products
```

The 10,000 catalog image embeddings are generated in advance and saved to disk. This prevents the application from recalculating every product embedding each time it starts.

---

## Dataset

This project uses the **Fashion Product Images Dataset** from Kaggle.

The original dataset contains approximately 44,000 fashion products with metadata including:

- Product ID
- Gender
- Master category
- Subcategory
- Article type
- Base colour
- Season
- Usage
- Product name

For the deployed StyleMatch AI catalog, **10,000 products** were sampled and their CLIP embeddings were pre-computed.

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- OpenAI CLIP
- NumPy
- Pandas
- Pillow
- Streamlit
- Matplotlib

---

## Project Structure

```text
stylematch-AI/
│
├── data/
│   └── catalog_images/
│
├── embeddings/
│   ├── image_embeddings_10000.npy
│   └── products_10000.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   └── search_engine.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The original dataset files are excluded from the Git repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SwanHtetHarry/stylematch-AI.git
cd stylematch-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

---

## Search Example

A user can search:

```text
black casual shoes
```

StyleMatch AI:

1. Converts the query into a CLIP text embedding.
2. Compares it against 10,000 pre-computed image embeddings.
3. Calculates cosine similarity.
4. Ranks products by similarity.
5. Displays the most relevant results.

---

## Image Search Example

A user can upload an image of a shoe, shirt, handbag, watch, cap, or another fashion product.

The uploaded image is converted into a CLIP image embedding and compared against the catalog to find visually similar products.

---

## Why CLIP?

Traditional image classifiers are normally trained to recognize a fixed list of classes.

CLIP is different because it learns relationships between **images and natural language**.

This makes it suitable for semantic search applications such as:

```text
"red women's handbag"
"blue men's t-shirt"
"black casual sneakers"
```

without training a new classifier for every possible search phrase.

---

## Current Limitations

This project is designed as an AI/ML portfolio project and has several limitations.

- The deployed catalog uses 10,000 products rather than the entire dataset.
- Search quality depends on CLIP's learned representation.
- Product metadata may contain inconsistent or missing values.
- Similarity scores are ranking scores and should not be interpreted as probabilities.
- The dataset mainly contains studio-style fashion product images, so performance may differ with real-world photos.

---

## Future Improvements

Possible future improvements include:

- FAISS vector search for larger catalogs
- Full 44,000-product catalog
- Combined metadata and embedding ranking
- Personalized recommendations
- User preference profiles
- Product recommendation history
- More advanced filtering
- Improved mobile interface
- GPU-based embedding generation
- Evaluation metrics for retrieval quality

---

## What I Learned

Through this project I practiced:

- Multimodal AI
- CLIP
- Image embeddings
- Text embeddings
- Semantic search
- Cosine similarity
- Recommendation systems
- Batch inference
- Data preprocessing
- Model integration
- Streamlit application development
- Organizing reusable ML code
- Building an end-to-end AI application

---

## Author

**Swan Htet (Harry Li)**

B.Sc. Information Technology Student  
AI / Machine Learning Project

GitHub:  
https://github.com/SwanHtetHarry