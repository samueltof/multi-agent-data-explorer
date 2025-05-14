import csv
import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.config.logger import logger # Assuming logger is configured and accessible

DEFAULT_NUM_EXAMPLES = 10
QUERY_EXAMPLES_CSV_PATH_RELATIVE_TO_PROJECT_ROOT = "databases/vdjdb/query_examples.csv"

def get_few_shot_examples(
    user_query: str, 
    embed_model: Any, 
    num_examples: int = DEFAULT_NUM_EXAMPLES
) -> str:
    """
    Retrieve and format a few-shot example set based on user input
    using RAG from a CSV file.
    """
    try:
        current_file = Path(__file__).resolve()
        # src/agents/v1_data_explorer/rag_utils.py -> project_root
        project_root = current_file.parent.parent.parent.parent 
        fewshot_examples_path = project_root / QUERY_EXAMPLES_CSV_PATH_RELATIVE_TO_PROJECT_ROOT

        if not fewshot_examples_path.exists():
            logger.warning(f"Few-shot examples CSV not found at: {fewshot_examples_path}")
            return ""

        logger.info(f"Loading few-shot examples from: {fewshot_examples_path}")
        
        data_dict = {}
        with open(fewshot_examples_path, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            # Ensure expected columns are present
            if not reader.fieldnames or not ("example_input_question" in reader.fieldnames and "example_output_query" in reader.fieldnames):
                logger.error(f"CSV file {fewshot_examples_path} is missing expected columns 'example_input_question' or 'example_output_query'.")
                return ""
            for row in reader:
                data_dict[row["example_input_question"]] = row
        
        if not data_dict:
            logger.warning(f"No data found in few-shot examples CSV: {fewshot_examples_path}")
            return ""

        documents = [
            Document(page_content=json.dumps(question)) # Store the question itself for lookup
            for question in data_dict.keys()
        ]
        
        logger.info(f"Building FAISS index for {len(documents)} few-shot examples.")
        vectorstore = FAISS.from_documents(documents, embed_model)

        logger.info(f"Performing similarity search for user query: '{user_query[:100]}...'")
        similar_docs = vectorstore.similarity_search(user_query, k=num_examples)

        result_strs = []
        for doc in similar_docs:
            question = json.loads(doc.page_content)
            example_data = data_dict.get(question)
            if example_data:
                # Format based on the user's example structure: "Key: Value"
                # We are interested in 'example_input_question' and 'example_output_query'
                formatted_example = (
                    f"User Question: {example_data['example_input_question']}\\n"
                    f"SQL Query: {example_data['example_output_query']}"
                )
                result_strs.append(formatted_example)
            else:
                logger.warning(f"Could not find example data for retrieved question: {question}")
        
        if not result_strs:
            logger.info("No similar examples found after RAG.")
            return ""

        example_set = "\\n\\n---\\n\\n".join(result_strs)
        logger.info(f"Retrieved {len(result_strs)} few-shot examples.")
        return f"\\n\\n{example_set}\\n\\n"

    except Exception as e:
        logger.error(f"Error in get_few_shot_examples: {e}", exc_info=True)
        return ""

if __name__ == '__main__':
    # Example Usage (requires a mock embedding model or actual setup)
    class MockEmbeddings:
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            # Simple hashing to simulate embeddings
            return [[float(sum(bytearray(text.encode()))) / 1e5] for text in texts]

        def embed_query(self, text: str) -> List[float]:
            return [float(sum(bytearray(text.encode()))) / 1e5]

    mock_embed_model = MockEmbeddings()
    
    # Create a dummy csv for testing
    dummy_csv_path = Path("dummy_query_examples.csv")
    with open(dummy_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["example_input_question", "example_output_query"])
        writer.writerow(["How many users?", "SELECT COUNT(*) FROM users;"])
        writer.writerow(["List all products.", "SELECT * FROM products;"])
        writer.writerow(["Oldest user?", "SELECT name FROM users ORDER BY age DESC LIMIT 1;"])

    # Point the util to use this dummy CSV for the test
    QUERY_EXAMPLES_CSV_PATH_RELATIVE_TO_PROJECT_ROOT = str(dummy_csv_path.resolve())
    # Note: For this __main__ block to work correctly, you'd need to adjust how 
    # QUERY_EXAMPLES_CSV_PATH_RELATIVE_TO_PROJECT_ROOT is used in get_few_shot_examples,
    # or temporarily modify the global for testing, or pass path as an argument.
    # For simplicity, this example assumes get_few_shot_examples will find it if project_root calculation is correct.
    # Actual testing would require placing this dummy file in the expected project structure.

    # This test setup is a bit complex due to Path(__file__) and project root assumptions.
    # A simpler test would be to pass the CSV path directly to get_few_shot_examples.
    # For now, this is just a conceptual placeholder for testing.
    
    # Test with a query
    # user_q = "How many items are there?"
    # examples = get_few_shot_examples(user_q, mock_embed_model)
    # print("--- Retrieved Examples ---")
    # print(examples)
    # print("--- End of Examples ---")
    
    # Cleanup dummy file
    # if dummy_csv_path.exists():
    #     dummy_csv_path.unlink()
    pass 