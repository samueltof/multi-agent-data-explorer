from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import List, Dict, Any
import os
from pathlib import Path
import pandas as pd
from langchain_core.embeddings import Embeddings
from src.config.settings import get_settings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.chroma import Chroma
from ..services.embeddings import get_embeddings_service
from .vector_store import VectorStoreType

# Get settings from centralized configuration
settings = get_settings()

# Custom BedrockEmbeddings class that handles the API differently
class BedrockEmbeddings(Embeddings):
    """Custom implementation for Bedrock embeddings"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "amazon.titan-embed-text-v1",
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
            
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Bedrock"""
        embeddings = []
        # Process each text individually since Bedrock doesn't accept arrays
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Bedrock"""
        import time
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Make a direct API call with a single string
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                if 'rate limit' in str(e).lower() and attempt < max_retries - 1:
                    print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

# Initialize our custom embeddings model using portkey_bedrock settings from LLMSettings
embeddings_model = BedrockEmbeddings(
    api_key=settings.llm.portkey_bedrock.api_key,
    base_url=settings.llm.portkey_bedrock.base_url,
    model=settings.llm.portkey_bedrock.embedding_model
)

# Define query pair schema with vector embedding
class QueryPair(LanceModel):
    question: str
    query: str
    vector: Vector(1536)  # Titan embedding size

# VectorStoreService class to provide vector database operations
class VectorStoreService:
    """Service for managing vector database operations with examples and similar query search."""
    
    def __init__(self, agent_name=None, embedding_provider=None, embedding_model=None):
        """
        Initialize the VectorStoreService.
        
        Args:
            agent_name: Name of the agent folder to use. If provided, will use agent-specific paths.
            embedding_provider: Optional provider for embeddings (anthropic, openai, etc.)
            embedding_model: Optional embedding model to use
        """
        self.agent_name = agent_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.project_root = Path(__file__).parent.parent.parent
        
        if agent_name:
            # Use agent-specific paths
            self.db_path = str(self.project_root / "src" / "agents" / agent_name / "data" / "lancedb")
            self.csv_path = str(self.project_root / "src" / "agents" / agent_name / "data" / "query_examples.csv")
        else:
            # Use default paths from settings
            self.db_path = str(self.project_root / settings.database.vector_db_path)
            self.csv_path = str(self.project_root / settings.database.query_examples_path)
        
        self.table_name = settings.database.vector_table_name
        self.db = None
        self._ensure_db_connection()
    
    def _ensure_db_connection(self):
        """Ensure connection to the database is established"""
        if self.db is None:
            os.makedirs(self.db_path, exist_ok=True)
            self.db = lancedb.connect(self.db_path)
    
    def create_vector_database(self, csv_path=None, db_path=None, override=False):
        """
        Create or update the vector database with query examples.
        
        Args:
            csv_path: Optional path to CSV file with examples. Defaults to instance path.
            db_path: Optional path to database. Defaults to instance path.
            override: If True, recreates the database even if it exists.
            
        Returns:
            tuple: (db, query_table) - References to the database and table
        """
        # Use instance paths if not specified
        csv_path = csv_path or self.csv_path
        db_path = db_path or self.db_path
        
        # Create database directory if needed
        os.makedirs(db_path, exist_ok=True)
        
        # Connect to database
        db = lancedb.connect(db_path)
        self.db = db
        
        # Load query examples
        query_pairs = load_query_examples(csv_path)
        
        # Check if table exists
        if self.table_name in db.table_names() and not override:
            return self._update_existing_table(db, query_pairs)
        else:
            return self._create_new_table(db, query_pairs, override)
    
    def _update_existing_table(self, db, query_pairs):
        """Update existing table with new entries"""
        query_table = db.open_table(self.table_name)
        
        # Find new entries
        new_entries = find_new_entries(query_pairs, query_table)
        
        if new_entries:
            # Generate embeddings for new questions
            for pair in new_entries:
                pair["vector"] = embeddings_model.embed_query(pair["question"])
            
            # Add new entries
            query_table.add(new_entries)
        
        return db, query_table
    
    def _create_new_table(self, db, query_pairs, override):
        """Create a new table for vector storage"""
        # Drop existing table if override is True
        if override and self.table_name in db.table_names():
            db.drop_table(self.table_name)
        
        # Generate embeddings for all questions
        for pair in query_pairs:
            pair["vector"] = embeddings_model.embed_query(pair["question"])
        
        # Create table and add data
        if len(query_pairs) > 0:
            query_table = db.create_table(self.table_name, schema=QueryPair)
            query_table.add(query_pairs)
        else:
            # Create empty table to avoid errors
            query_table = db.create_table(self.table_name, schema=QueryPair)
        
        return db, query_table
    
    def search_similar_queries(self, question, limit=2):
        """
        Search for similar questions in the vector database.
        
        Args:
            question: The question to search for
            limit: Maximum number of results to return
            
        Returns:
            pandas.DataFrame: Search results from the vector database
        """
        self._ensure_db_connection()
        
        if not self.table_name in self.db.table_names():
            raise ValueError(f"Table '{self.table_name}' does not exist in the database")
        
        query_table = self.db.open_table(self.table_name)
        
        # Generate embedding for the question
        question_embedding = embeddings_model.embed_query(question)
        
        # Search for similar questions
        search_results = query_table.search(question_embedding).limit(limit).to_pandas()
        
        return search_results
    
    def get_all_examples(self):
        """
        Return all examples from the vector database.
        
        Returns:
            pandas.DataFrame: All examples in the database
        """
        self._ensure_db_connection()
        
        if not self.table_name in self.db.table_names():
            raise ValueError(f"Table '{self.table_name}' does not exist in the database")
        
        query_table = self.db.open_table(self.table_name)
        return query_table.to_pandas()
    
    def get_few_shot_examples(self, query, num_examples=3, agent_name=None):
        """
        Get few-shot examples similar to the given query.
        
        Args:
            query: The query to find similar examples for
            num_examples: Number of examples to retrieve
            agent_name: Optional agent name to override instance value
            
        Returns:
            str: Formatted string with few-shot examples
        """
        if agent_name:
            self.agent_name = agent_name
            
        try:
            # Search for similar queries
            results = self.search_similar_queries(query, limit=num_examples)
            
            if results.empty:
                return "No similar examples available."
                
            # Format the examples
            examples_text = ""
            for i, row in results.iterrows():
                examples_text += f"Question: {row['question']}\n"
                examples_text += f"SQL: {row['query']}\n\n"
                
            return examples_text
            
        except Exception as e:
            import traceback
            print(f"Error getting few-shot examples: {str(e)}")
            print(traceback.format_exc())
            return "No similar examples available."

# Keep existing functions for backward compatibility
def load_query_examples(csv_path):
    """Load query examples from CSV file"""
    print(f"Loading query examples from {csv_path}")
    
    try:
        # First, let's read the raw file to see what we're dealing with
        with open(csv_path, 'r') as f:
            content = f.read()
        
        print(f"CSV file size: {len(content)} bytes")
        print(f"First 200 characters: {content[:200]}")
        
        # Create a clean version of the file that removes the duplicate header
        clean_content = content.replace('"example_input_question,example_output_query"', '')
        
        # Create a temporary file with the cleaned content
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp:
            temp.write(clean_content)
            temp_path = temp.name
        
        # Simple direct approach - read all lines and process manually
        with open(temp_path, 'r') as f:
            lines = f.readlines()
        
        # Skip the header line
        data_lines = lines[1:] if lines[0].startswith('example_input_question') else lines
        
        # Process the CSV line by line, extracting question-query pairs
        query_pairs = []
        i = 0
        while i < len(data_lines):
            line = data_lines[i].strip()
            if not line:  # Skip empty lines
                i += 1
                continue
                
            # Find the comma that separates the question from the query
            # We need to handle cases where there are commas in the quoted strings
            comma_pos = None
            in_quotes = False
            for j, char in enumerate(line):
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    comma_pos = j
                    break
            
            if comma_pos is None:  # No comma found, malformed line
                i += 1
                continue
                
            # Extract the question and query
            question = line[:comma_pos].strip('"').strip()
            query = line[comma_pos+1:].strip('"').strip()
            
            # Skip if question is empty or starts with SELECT (indicating parsing error)
            if not question or question.upper().startswith('SELECT'):
                i += 1
                continue
                
            # Skip if query is empty or doesn't look like SQL
            if not query or not query.upper().startswith('SELECT'):
                i += 1
                continue
                
            # Add to pairs
            query_pairs.append({
                "question": question,
                "query": query
            })
            
            i += 1
        
        # Remove duplicates by converting to dict and back to list
        unique_pairs = {}
        for pair in query_pairs:
            unique_pairs[pair["question"]] = pair["query"]
        
        query_pairs = [{"question": q, "query": qry} for q, qry in unique_pairs.items()]
        
        print(f"Extracted {len(query_pairs)} unique query pairs")
        
        # Show a sample of what we extracted
        if query_pairs:
            print("\nSample of extracted pairs:")
            for i in range(min(2, len(query_pairs))):
                print(f"\nPair {i+1}:")
                print(f"Question: {query_pairs[i]['question']}")
                print(f"Query: {query_pairs[i]['query']}")
        
        # Clean up the temporary file
        import os
        os.unlink(temp_path)
        
        if not query_pairs:
            print("WARNING: No valid query pairs were extracted. Using a default example.")
            query_pairs = [{
                "question": "How many patients are enrolled in each clinical trial phase?",
                "query": "SELECT ct.phase, COUNT(DISTINCT te.patient_id) as patient_count FROM clinical_trials ct JOIN trial_enrollment te ON ct.trial_id = te.trial_id GROUP BY ct.phase"
            }]
        
        return query_pairs
    
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        # Return a default pair as fallback
        return [{
            "question": "How many patients are enrolled in each clinical trial phase?",
            "query": "SELECT ct.phase, COUNT(DISTINCT te.patient_id) as patient_count FROM clinical_trials ct JOIN trial_enrollment te ON ct.trial_id = te.trial_id GROUP BY ct.phase"
        }]

def find_new_entries(query_pairs: List[Dict[str, str]], existing_table) -> List[Dict[str, Any]]:
    """Find entries in query_pairs that don't exist in the database table
    
    Args:
        query_pairs: List of query pairs loaded from CSV
        existing_table: LanceDB table containing existing records
        
    Returns:
        List of query pairs that are not in the database
    """
    # Get existing questions from the table
    existing_df = existing_table.to_pandas()
    existing_questions = set(existing_df['question'].tolist())
    
    # Identify new entries
    new_entries = []
    for pair in query_pairs:
        if pair["question"] not in existing_questions:
            new_entries.append(pair)
    
    return new_entries

def create_vector_database(csv_path: str = None, db_path: str = None, override: bool = False, agent_name: str = None):
    """
    Create or update a vector database with query examples.
    
    Args:
        csv_path: Path to the CSV file with query examples. If None, the default path is used.
        db_path: Path to the LanceDB database. If None, the default path is used.
        override: If True, recreates the database from scratch even if it exists already.
        agent_name: Name of the agent folder to use. If provided, will use agent-specific paths.
        
    Returns:
        tuple: (db, query_table) - References to the database and table
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Set default CSV path if not provided
    if csv_path is None:
        if agent_name:
            # Use agent-specific path - include 'src' in the path
            csv_path = str(project_root / "src" / "agents" / agent_name / "data" / "query_examples.csv")
        else:
            # Use path from settings - relative to project root
            csv_path = str(project_root / settings.database.query_examples_path)
    
    # Set default DB path if not provided
    if db_path is None:
        if agent_name:
            # Use agent-specific path - include 'src' in the path
            db_path = str(project_root / "src" / "agents" / agent_name / "data" / "lancedb")
        else:
            # Use path from settings - relative to project root
            db_path = str(project_root / settings.database.vector_db_path)
    
    print(f"Using vector database path: {db_path}")
    print(f"Using query examples path: {csv_path}")
    
    # Create database directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    # Create a LanceDB database - connect directly to the provided path
    db = lancedb.connect(db_path)
    
    # Load query examples from CSV
    query_pairs = load_query_examples(csv_path)
    
    # Validate we have data before proceeding
    if not query_pairs:
        raise ValueError("No query pairs were loaded from the CSV file. Please check the format.")
    
    # Table name for storing query pairs from settings
    table_name = settings.database.vector_table_name
    
    # Check if table already exists
    if table_name in db.table_names() and not override:
        print(f"Table '{table_name}' already exists. Checking for new entries...")
        
        # Get reference to existing table
        query_table = db.open_table(table_name)
        
        # Find new entries that aren't in the database yet
        new_entries = find_new_entries(query_pairs, query_table)
        
        if new_entries:
            print(f"Found {len(new_entries)} new entries to add to the database")
            
            # Generate embeddings for new questions
            print("Generating embeddings for new questions...")
            for pair in new_entries:
                pair["vector"] = embeddings_model.embed_query(pair["question"])
                print(f"Embedded new question: {pair['question'][:50]}...")
            
            # Add new entries to the table
            query_table.add(new_entries)
            print(f"Added {len(new_entries)} new query pairs to '{table_name}' table")
        else:
            print("No new entries to add. Database is up to date.")
    else:
        # If override is True or table doesn't exist, create new table
        if override and table_name in db.table_names():
            print(f"Override flag set. Dropping existing '{table_name}' table...")
            db.drop_table(table_name)
            print(f"Table '{table_name}' dropped.")
            
        print(f"Creating new table '{table_name}'...")
        
        # Generate embeddings for all questions
        print("Generating embeddings for questions...")
        for pair in query_pairs:
            pair["vector"] = embeddings_model.embed_query(pair["question"])
            print(f"Embedded question: {pair['question'][:50]}...")
        
        # Create the table and add all data only if we have pairs
        if len(query_pairs) > 0:
            query_table = db.create_table(table_name, schema=QueryPair)
            query_table.add(query_pairs)
            print(f"Added {len(query_pairs)} query pairs to '{table_name}' table")
        else:
            print("Warning: No query pairs to add to the database!")
            # Create empty table to avoid errors
            query_table = db.create_table(table_name, schema=QueryPair)
    
    print(f"\nVector database initialization complete. Database stored at {db_path}")
    
    return db, query_table

def test_vector_database(db=None, table_name=None, test_question=None, agent_name=None):
    """
    Test the vector database by running a similarity search on a sample question.
    
    Args:
        db: LanceDB database connection. If None, connects to default location.
        table_name: Name of the table to query. Defaults to setting from config.
        test_question: The question to use for testing. If None, uses a default question.
        agent_name: Name of the agent folder to use. If provided, will use agent-specific paths.
        
    Returns:
        pandas.DataFrame: Search results from the vector database
    """
    if db is None:
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        if agent_name:
            # Use agent-specific path - include 'src' in the path
            db_path = str(project_root / "src" / "agents" / agent_name / "data" / "lancedb")
        else:
            # Use path from settings - relative to project root
            db_path = str(project_root / settings.database.vector_db_path)
            
        db = lancedb.connect(db_path)
    
    # Use table name from settings if not provided
    if table_name is None:
        table_name = settings.database.vector_table_name
    
    if not table_name in db.table_names():
        raise ValueError(f"Table '{table_name}' does not exist in the database")
    
    query_table = db.open_table(table_name)
    
    # Use default question if none provided
    if test_question is None:
        test_question = "What is the most common diagnosis among patients?"
    
    print(f"\nTesting vector database with question: '{test_question}'")
    
    # Generate embedding for the test question
    test_embedding = embeddings_model.embed_query(test_question)
    
    # Search for similar questions
    search_results = query_table.search(test_embedding).limit(2).to_pandas()
    print("\nSearch Results:")
    print(search_results[["question", "query"]])
    
    return search_results

def vector_database_as_pandas(db=None, table_name=None, agent_name=None):
    """
    Return the entire vector database as a pandas DataFrame.
    
    Args:
        db: LanceDB database connection. If None, connects to default location.
        table_name: Name of the table to query. Defaults to setting from config.
        agent_name: Name of the agent folder to use. If provided, will use agent-specific paths.
        
    Returns:
        pandas.DataFrame: The entire contents of the vector database table
    """
    if db is None:
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        if agent_name:
            # Use agent-specific path - include 'src' in the path
            db_path = str(project_root / "src" / "agents" / agent_name / "data" / "lancedb")
        else:
            # Use path from settings - relative to project root
            db_path = str(project_root / settings.database.vector_db_path)
            
        db = lancedb.connect(db_path)
    
    # Use table name from settings if not provided
    if table_name is None:
        table_name = settings.database.vector_table_name
    
    if not table_name in db.table_names():
        raise ValueError(f"Table '{table_name}' does not exist in the database")
    
    query_table = db.open_table(table_name)
    
    # Convert the entire table to a pandas DataFrame
    df = query_table.to_pandas()
    
    return df

def main():
    """Legacy wrapper for create_vector_database with default parameters"""
    db, query_table = create_vector_database()
    # Test the database after creation
    test_vector_database(db)
    return db, query_table

if __name__ == "__main__":
    main()
