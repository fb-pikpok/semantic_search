import logging
import openai
import os

from dotenv import load_dotenv

from helper.prompt_templates import prompt_template_cluster_naming

# 0) Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()
logger.info("Environment variables loaded.")

chat_model_name = 'gpt-4o-mini'
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()



def generate_cluster_name(topics_list):
    """
    Generates a concise name for the cluster using OpenAI API.

    Args:
        topics_list (list): List of topics to summarize.

    Returns:
        str: Generated cluster name.
    """
    if not topics_list:
        logger.warning("No topics found to generate a cluster name.")
        return "Unknown"

    topics = ", ".join(topics_list)
    prompt_cluster_naming = prompt_template_cluster_naming.format(topics=topics)

    try:
        response = client.chat.completions.create(
            model=chat_model_name,
            messages=[
                {"role": "system", "content": "You are an expert at summarizing topics into concise names."},
                {"role": "user", "content": prompt_cluster_naming},
            ],
            max_tokens=100
        )

        # Extract cluster name
        if response.choices and response.choices[0].message.content:
            cluster_name = response.choices[0].message.content.strip()
            return cluster_name
        else:
            logger.error("Unexpected API response structure.")
            return "Error"
    except Exception as e:
        logger.error(f"Error in generate_cluster_name function: {e}")
        return "Error"