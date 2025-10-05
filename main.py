from google.cloud import vision
import os
import boto3
import json
import re
from typing import Dict, List, Optional
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# --- Configuration ---
# Load environment variables for configuration
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", './camer-cooked-key.json')
BEDROCK_KB_ID = os.environ.get("BEDROCK_KB_ID", "WBEE2GZJPH") # Default KB ID
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")

# Initialize the client with the service account key
client = vision.ImageAnnotatorClient.from_service_account_json('./camer-cooked-key.json')
client = vision.ImageAnnotatorClient.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)

def get_image_labels(image_path):
    try:
        # Read the image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        # Create the image object
        image = vision.Image(content=content)
        
        # Perform label detection
        response = client.label_detection(image=image)
        labels = response.label_annotations
        
        # Format the labels for better readability
        formatted_labels = []
        for label in labels:
            formatted_labels.append({
                'description': label.description
            })
        
        return formatted_labels
    except Exception as error:
        print(f'Error: {error}')
        raise error


def AI_Inference(model_id, system_prompts, data):
    bedrock_runtime = boto3.client(
      service_name="bedrock-runtime",
      region_name="us-east-1",
    )

    temp = 0.5
    inference_config = {"temperature": temp}

    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=data,
        system=system_prompts,
        inferenceConfig=inference_config,
    )

    text = response["output"]["message"]["content"][0]["text"]

    return text

def normalize_data(data):
    model_id = "us.amazon.nova-pro-v1:0"
    system_prompts = [
        {"text": "You are a helpful assistant that normalizes data for a recipe website. You identify edible ingredients from image labels and remove non-edible items like objects, furniture, or non-food items."}
    ]
    
    # Convert the list of dictionaries to a simple string list
    ingredient_list = [item['description'] for item in data]
    ingredient_string = ", ".join(ingredient_list)
    
    message_1 = {
        "role": "user",
        "content": [{"text": f"The following is a list of items detected in an image: {ingredient_string}. Please identify which items are edible ingredients that could be used in cooking/recipes. Return ONLY the edible ingredients as a comma-separated list, removing any non-edible items like objects, furniture, or non-food items."}]
    }

    messages = [message_1]

    result = AI_Inference(model_id, system_prompts, messages)

    return result 

model_ids = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama3-1-405b-instruct-v1:0",
    "meta.llama3-1-8b-instruct-v1:0",
    "mistral.mistral-large-2402-v1:0",
    "us.amazon.nova-pro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.amazon.nova-micro-v1:0",
]


def parse_ingredient_list(text):
    # Convert a comma-separated string into a cleaned Python list
    if not isinstance(text, str):
        return []
    return [item.strip() for item in text.split(',') if item and item.strip()]

def generate_recipe_with_kb(ingredients: List[str], knowledge_base_id: str, region: str = 'us-east-1') -> str:
    """Use Bedrock Agent Runtime Knowledge Base to generate a recipe using ingredients as context.

    Returns only the Title and Instructions in markdown format.
    """
    bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)

    ingredients_str = ", ".join(ingredients)

    prompt = f"""
You are a helpful recipe assistant.

Using ONLY the provided context and any relevant knowledge from the knowledge base, generate:
- Title (one line)
- Instructions (numbered steps)

Do NOT include extra sections like ingredients, time, or descriptions. Output strictly:

Title: <title>
Instructions:
1. ...
2. ...

Context (ingredients): {ingredients_str}
"""

    # Try multiple commonly available foundation model ARNs; use the first that works
    candidate_model_arns = [
        # Anthropic Claude 3.5 Sonnet v2
        'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0',
        # Anthropic Claude 3.5 Haiku
        'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-haiku-20241022-v1:0',
        # Meta Llama 3.1 70B Instruct
        'arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-1-70b-instruct-v1:0',
        # Amazon Nova Pro
        'arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0',
        # Mistral Large 24.02
        'arn:aws:bedrock:us-east-1::foundation-model/mistral.mistral-large-2402-v1:0',
    ]

    last_error = None
    for model_arn in candidate_model_arns:
        try:
            response = bedrock_agent.retrieve_and_generate(
                input={'text': prompt},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': knowledge_base_id,
                        'modelArn': model_arn,
                    }
                }
            )
            # Success
            break
        except Exception as e:
            last_error = e
            response = None
            continue

    if response is None:
        return f"Bedrock retrieve_and_generate failed across models. Last error: {last_error}"

    # Best-effort extraction of the generated text
    output_text = None
    if isinstance(response, dict):
        output_text = response.get('output', {}).get('text')
        if not output_text:
            # Fallback shapes
            output_text = response.get('responseText') or response.get('generatedText')
    return output_text if output_text else str(response)

app = Flask(__name__)
app.secret_key = "dev"

UPLOAD_FOLDER = "/Users/aaravmaheshwari/Hackathon/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])    
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])    
def upload():
    if "image" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(saved_path)

        # Pipeline: Vision -> Normalize -> KB RAG -> Output
        detected_labels = get_image_labels(saved_path)
        normalized_text = normalize_data(detected_labels)
        ingredients = parse_ingredient_list(normalized_text)
        KB_ID = "WBEE2GZJPH"
        recipe_text = generate_recipe_with_kb(ingredients, KB_ID)
        recipe_text = generate_recipe_with_kb(ingredients, BEDROCK_KB_ID)
        return render_template("index.html", recipe_text=recipe_text, ingredients=ingredients)
    else:
        flash("File type not allowed. Please upload a PNG/JPG/JPEG.")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
