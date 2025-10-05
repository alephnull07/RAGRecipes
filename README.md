# RAGRecipes

An AI-powered cooking assistant that uses Retrieval-Augmented Generation (RAG) to craft delicious, data-backed recipes from a simple photo of your ingredients.

![RAGRecipes Demo](https://placehold.co/800x400/orange/white?text=RAGRecipes+App+Screenshot)
*Replace with a screenshot or GIF of your application.*

---

## Inspiration

Large Language Models can generate text with remarkable fluency, but when it comes to recipes, they often fail — producing vague and inaccurate dishes. We wanted to address this inefficiency by combining LLMs with real-world data. That’s where the idea for **RAGRecipes** came from: an AI-powered cooking assistant that uses Retrieval-Augmented Generation (RAG) to craft data-backed recipes in real time.

## What it Does

RAGRecipes allows users to upload an image of their ingredients. The system then:
1.  **Identifies** what’s in the image using computer vision.
2.  **Filters** the list to keep only edible food items.
3.  **Retrieves** the most relevant recipes from a massive recipe database.
4.  **Generates** a detailed, coherent recipe tailored to what the user already has.

Instead of generic or hallucinated suggestions, users get context-aware, grounded recipes pulled from real data.

## How We Built It

RAGRecipes is built on a multi-stage pipeline that integrates computer vision, data sanitization, and a powerful RAG architecture.

### Architecture Flow

1.  **Image Upload (Flask Frontend)**: The user uploads an image of their ingredients through a simple web interface.
2.  **Ingredient Detection (Google Vision API)**: The image is sent to the Google Vision API, which returns a list of detected object labels.
3.  **Data Normalization (Amazon Bedrock)**: The raw labels are passed to an Amazon Bedrock model (Amazon Nova) with a specific prompt to filter out non-edible items (e.g., "bowl", "table", "knife") and return a clean, comma-separated list of ingredients.
4.  **Recipe Retrieval (RAG with Bedrock Knowledge Base)**:
    *   The cleaned ingredient list is used to query an **Amazon Bedrock Knowledge Base**.
    *   This Knowledge Base is connected to a **vector database** containing the **RecipeNLG dataset** (stored in an S3 bucket). The recipes were vectorized using the **Amazon Titan V2 embeddings model**.
    *   The system performs a similarity search to find the most relevant recipes based on the user's ingredients.
5.  **Recipe Generation (Amazon Bedrock LLM)**: The retrieved recipes are passed as context to an Amazon Bedrock foundation model (e.g., Anthropic Claude), which generates a final, well-structured recipe (Title and Instructions) for the user.

### Tech Stack

*   **Backend**: Python, Flask
*   **Frontend**: HTML, CSS
*   **Cloud Services**:
    *   **AWS**:
        *   **Amazon Bedrock**: For LLM-based data normalization, RAG, and final recipe generation.
        *   **Bedrock Knowledge Base**: For managing the RAG pipeline and vector store.
        *   **Amazon Titan V2**: For creating text embeddings of the recipe data.
        *   **Amazon S3**: For storing the RecipeNLG dataset.
    *   **Google Cloud**:
        *   **Vision API**: For object and ingredient detection from images.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

*   Python 3.8+
*   An AWS account with programmatic access (API keys) and access to Amazon Bedrock models.
*   A Google Cloud Platform account with the Vision API enabled and a service account key file.
*   A pre-configured Amazon Bedrock Knowledge Base.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAGRecipes.git
cd RAGRecipes
```

### 2. Set Up a Virtual Environment

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
Flask
google-cloud-vision
boto3
werkzeug
python-dotenv
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of the project directory and add your credentials and configuration:

```
# AWS Credentials
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
AWS_DEFAULT_REGION=us-east-1

# Google Cloud Credentials
GOOGLE_APPLICATION_CREDENTIALS=./path/to/your/gcp-key.json

# Application Configuration
BEDROCK_KB_ID=YOUR_BEDROCK_KNOWLEDGE_BASE_ID
UPLOAD_FOLDER=uploads
```

*   Place your Google Cloud service account JSON key file in the location specified by `GOOGLE_APPLICATION_CREDENTIALS`.
*   The `boto3` library will automatically use the AWS credentials from the environment variables.

### 5. Run the Application

```bash
python main.py
```

The application will be available at `http://127.0.0.1:5000`.

## Accomplishments & Challenges

*   **Accomplishments**:
    *   Successfully implemented a fully functional RAG pipeline from scratch using Amazon Bedrock.
    *   Integrated a large real-world dataset (RecipeNLG) into a searchable, vectorized knowledge base.
    *   Built a clean, functional Flask frontend connected to a Python backend.
    *   Turned a concept into a working prototype that demonstrates the power of RAG in a practical domain.
*   **Challenges**:
    *   **Custom Data Chunking**: We initially struggled with custom chunking of our recipe CSV using AWS Lambda, facing memory and serialization issues. We pivoted to a simpler, fixed-size chunking method for reliability.
    *   **Deployment**: Our initial plans to deploy via AWS Lambda or Google Cloud Run with FastAPI ran into dependency conflicts. We simplified our stack to a Flask-native backend for smoother integration.

## What We Learned

*   How to effectively integrate RAG with real-world datasets.
*   The process of building and tuning an embedding-based retrieval system using Amazon Titan.
*   The value of simplicity: a complex, custom chunking function isn't always worth the development overhead compared to a simpler, effective solution.

## What's Next for RAGRecipes

*   **Improve Chunking**: Revisit a custom chunking mechanism with AWS Lambda to better separate ingredients and instructions.
*   **Real-time Detection**: Integrate a lightweight, real-time computer vision model for instant ingredient detection.
*   **Expand Database**: Grow our recipe database beyond RecipeNLG to include user-generated and trending recipes.
*   **Personalization**: Factor in user preferences, dietary restrictions, and available kitchen equipment.
*   **Production Deployment**: Deploy a scalable version using AWS Lambda + API Gateway and refine the retrieval architecture for faster query times.
