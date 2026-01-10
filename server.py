import streamlit as st
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# -----------------------------
# Load env variables
# -----------------------------
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# -----------------------------
# LLM Clients
# -----------------------------
def get_llm_client():
    if LLM_PROVIDER == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL)

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI

        return OpenAI(api_key=OPENAI_API_KEY)

    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


llm_client = get_llm_client()


# -----------------------------
# MongoDB connection
# -----------------------------
@st.cache_resource
def get_mongo_collections():
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[DB_NAME]
    return db["recipes"], db["recipe_ingredients"], db["ingredient_food_map"], db["raw_foods"]


(
    collection_recipes,
    collection_recipe_ingredients,
    collection_ingredient_food_map,
    collection_raw_foods,
) = get_mongo_collections()


# Helpers
def is_recipe_query(query: str) -> bool:
    keywords = ["ingredient", "ingredients", "recipe", "how to make", "what is in"]
    return any(k in query.lower() for k in keywords)


def fetch_full_recipe_data(user_query: str) -> dict:
    recipe = collection_recipes.find_one(
        {
            "$expr": {
                "$regexMatch": {
                    "input": user_query,
                    "regex": "$recipe_name",
                    "options": "i",
                }
            }
        }
    )

    if not recipe:
        return None

    recipe_id = recipe["recipe_id"]

    ingredients = list(
        collection_recipe_ingredients.aggregate([
            {
                "$match": {"recipe_id": recipe_id}
            },
            {
                "$lookup": {
                    "from": "ingredient_food_map", 
                    "localField": "food_id",         
                    "foreignField": "food_id",       
                    "as": "food_info"
                }
            },
            {
                "$unwind": {
                    "path": "$food_info",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "ingredient_name": 1,
                    "quantity_g": 1,
                    "food_id": 1,
                    "canonical_name": "$food_info.canonical_name"
                }
            },
            {
                "$lookup": {
                    "from": "raw_foods",
                    "localField": "food_id",
                    "foreignField": "food_id",
                    "as": "nutrition"
                }
            },
            {
                "$unwind": {
                    "path": "$nutrition",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "ingredient_name": 1,
                    "quantity_g": 1,
                    "food_id": 1,
                    "canonical_name": 1,
                    "nutrition": {
                        "energy_kcal_100g": "$nutrition.energy_kcal_100g",
                        "protein_g_100g": "$nutrition.protein_g_100g",
                        "fat_g_100g": "$nutrition.fat_g_100g",
                        "carbs_g_100g": "$nutrition.carbs_g_100g"
                    }
                }
            }
        ])
    )
    
    return {
        "recipe_name": recipe["recipe_name"],
        "cuisine": recipe["cuisine"],
        "ingredients": ingredients,
    }


def build_recipe_context(recipe):
    lines = [
        f"Recipe Name: {recipe['recipe_name']}",
        f"Cuisine: {recipe.get('cuisine', 'N/A')}",
        "",
        "Ingredients & Nutrition:",
    ]

    for i, ing in enumerate(recipe["ingredients"], start=1):
        lines.append(
            f"""{i}. {ing['ingredient_name']}
                Quantity: {ing.get('quantity_g', 'N/A')} g
                Energy (kcal/100g): {ing['nutrition'].get('energy_kcal_100g', 'N/A') if ing.get('nutrition') else 'N/A'}
                Protein (g/100g): {ing['nutrition'].get('protein_g_100g', 'N/A') if ing.get('nutrition') else 'N/A'}
                Fat (g/100g): {ing['nutrition'].get('fat_g_100g', 'N/A') if ing.get('nutrition') else 'N/A'}
                Carbs (g/100g): {ing['nutrition'].get('carbs_g_100g', 'N/A') if ing.get('nutrition') else 'N/A'}
            """
        )

    return "\n".join(lines)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MongoDB Chatbot", layout="centered")
st.title("🤖 MongoDB-Powered LLM Chatbot")

st.caption(f"Provider: **{LLM_PROVIDER.upper()}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# LLM Call Abstraction
# -----------------------------
def generate_response(system_prompt, messages):
    if LLM_PROVIDER == "gemini":
        chat = llm_client.start_chat(
            history=[
                (
                    {"role": "user", "parts": m["content"]}
                    if m["role"] == "user"
                    else {"role": "model", "parts": m["content"]}
                )
                for m in messages[:-1]
            ]
        )

        response = chat.send_message(
            f"{system_prompt}\n\nUser: {messages[-1]['content']}"
        )
        return response.text

    elif LLM_PROVIDER == "openai":
        response = llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=0.3,
        )
        return response.choices[0].message.content


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    mongo_context = ""

    if is_recipe_query(user_input):
        recipe = fetch_full_recipe_data(user_input)

        if recipe:
            mongo_context = build_recipe_context(recipe)
        else:
            mongo_context = "No recipe found in database."

    system_prompt = f"""
        You are a professional nutrition and cooking assistant.

        Rules:
        - Use ONLY the data provided.
        - Never invent ingredients or nutrition values.
        - If data is missing, clearly say so.
        - Answer the user's question accurately.

        Recipe Data:
        {mongo_context}
    """

    assistant_reply = "No Data."
    
    if mongo_context == "No recipe found in database.":
        assistant_reply = "No Data."
    else:
        assistant_reply = generate_response(system_prompt, st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
