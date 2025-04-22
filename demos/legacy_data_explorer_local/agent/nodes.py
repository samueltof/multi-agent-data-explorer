import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END
from services.database import DatabaseManager
from services.llm import LLM
from agents.legacy_data_explorer_local.agent.prompts import (
    get_parse_question_intent,
    get_parse_question_prompt,
    get_schema_description_prompt,
    get_generate_sql_prompt,
    get_validate_and_fix_sql_prompt,
    get_format_results_prompt,
    get_llm_response_prompt,
    get_choose_visualization_prompt,
    get_visualization_prompt,
)
from utils.local_vector_search import get_few_shot_examples
from config.database_config import DatabaseSettings
from config.logger import logger

load_dotenv()

class Nodes:
    def __init__(self):
        logger.info("Initializing Nodes class")
        self.db_settings = DatabaseSettings()
        self.db_manager = DatabaseManager(settings=self.db_settings)
        self.llm = LLM()
        self.default_provider = (
            "anthropic" if "anthropic" in self.llm.available_providers else None
        )
        # self.default_provider = (
        #     "azure" if "azure" in self.llm.available_providers else None
        # )

    def parse_question_intent(self, state: dict) -> dict:
        """Parse user question and identify the intention."""
        logger.info(f"🚩 Parsing question intent for: {state['question']}")
        question = state["question"]

        prompt = get_parse_question_intent()
        output_parser = JsonOutputParser()

        logger.info("🚩 Invoking LLM for intent parsing")
        response = self.llm.invoke(
            prompt, provider=self.default_provider, question=question
        )
        parsed_intent = output_parser.parse(response)
        logger.info(f"Parsed intent: {parsed_intent}")
        return {"intent": parsed_intent}

    def route_based_on_intent(self, state: dict) -> str:
        """Route the conversation based on the user's intention."""
        intent_category = state["intent"]["category"].lower()
        logger.info(f"Routing based on intent category: {intent_category}")

        if intent_category == "ambiguous":
            logger.warning("Ambiguous intent detected, routing to default_llm_response")
            return "default_llm_response"
        elif intent_category == "structure":
            logger.info(
                "Structure-related question detected, routing to schema_description"
            )
            return "schema_description"

        logger.info("Standard question detected, routing to parse_question_prompt")
        return "parse_question_prompt"

    def parse_question_prompt(self, state: dict) -> dict:
        """Parse user question and identify relevant tables and columns."""
        logger.info(f"🚩 Parsing question prompt for: {state['question']}")
        question = state["question"]
        schema = self.db_manager.load_schema_description()
        logger.info(f"🚩 Loaded schema for question parsing: {schema[:100]}...")

        prompt = get_parse_question_prompt()

        output_parser = JsonOutputParser()

        response = self.llm.invoke(
            prompt, provider=self.default_provider, schema=schema, question=question
        )
        parsed_response = output_parser.parse(response)
        return {"parsed_question": parsed_response}

    def schema_description(self, state: dict) -> dict:
        """Loads and formats the schema description from YAML file."""
        question = state["question"]
        intent = state["intent"]

        prompt = get_schema_description_prompt()

        schema_description = self.db_manager.load_schema_description()

        response = self.llm.invoke(
            prompt,
            provider=self.default_provider,
            question=question,
            intent=intent,
            schema=schema_description,
        )
        return {"answer": response}

    def get_random_subsample(self, state: dict) -> dict:
        """Retrieve a random subsample of rows from relevant tables and columns."""
        SQL_RANDOM_FUNCTIONS = {
            'sqlite': 'RANDOM()',
            'athena': 'rand()',
            'postgresql': 'RANDOM()',
            'mysql': 'RAND()'
        }
        parsed_question = state.get("parsed_question", {})
        logger.info(
            f"🚩 Getting random subsamples for parsed question:\n{parsed_question}"
        )

        # If the question isn't relevant, return an empty sample dict.
        if not parsed_question.get("is_relevant", False):
            return {"samples": {}}

        # Get DB type and corresponding random function
        db_type = self.db_settings.database_type.value
        random_func = SQL_RANDOM_FUNCTIONS.get(db_type, 'RANDOM()')

        samples = {}
        for table_info in parsed_question.get("relevant_tables", []):
            table_name = table_info.get("table_name", "")
            # Clean table name - remove any database prefix and use just the base table name
            table_name = table_name.split('.')[-1].strip()
            
            noun_columns = table_info.get("noun_columns", [])
            if table_name and noun_columns:
                # Join column names with commas
                column_names = ", ".join(noun_columns)
                
                # Create query using cleaned table name
                query = f"SELECT {column_names} FROM {table_name} ORDER BY {random_func} LIMIT 3"

                try:
                    # Execute the query and store the result
                    results = self.db_manager.execute_query(query)
                    samples[table_name] = results
                except Exception as e:
                    logger.error(f"Failed to execute SQLite query: {e}")
                    continue

        return {"samples": samples}

    def generate_sql(self, state: dict) -> dict:
        """Generate SQL query based on parsed question and unique nouns."""
        question = state["question"]
        parsed_question = state["parsed_question"]
        samples = state["samples"]
        if not parsed_question["is_relevant"]:
            return {"sql_query": "NOT_RELEVANT", "is_relevant": False}

        schema = self.db_manager.load_schema_description()

        few_shot_examples = get_few_shot_examples(
            self.llm.embed_model, question, num_examples=3
        )

        logger.info(
            f"🚩 Generating SQL query for parsed question:\n{parsed_question}"
            f"\nSamples: {samples}"
            f"\nFew-shot examples: {few_shot_examples}"
        )

        prompt = get_generate_sql_prompt()

        response = self.llm.invoke(
            prompt,
            provider=self.default_provider,
            database_name=os.getenv("DATABASE_NAME"),
            schema=schema,
            sql_examples=few_shot_examples,
            question=question,
            parsed_question=parsed_question,
            samples=samples,
        )

        logger.info(f"🔵 Generated SQL query: \n{response}")

        if response.strip() == "NOT_ENOUGH_INFO":
            return {"sql_query": "NOT_RELEVANT"}
        else:
            return {"sql_query": response}

    def validate_and_fix_sql(self, state: dict) -> dict:
        """Validate and fix the generated SQL query."""
        sql_query = state["sql_query"]

        if sql_query == "NOT_RELEVANT":
            return {"sql_query": "NOT_RELEVANT", "sql_valid": False}

        schema = self.db_manager.load_schema_description()

        prompt = get_validate_and_fix_sql_prompt()

        logger.info("🚩 Validating SQL query")
        output_parser = JsonOutputParser()
        response = self.llm.invoke(
            prompt, provider=self.default_provider, schema=schema, sql_query=sql_query
        )
        result = output_parser.parse(response)
        logger.info(f"🔎 SQL query validation result: {result}")

        if result["valid"] and result["issues"] is None:
            logger.info("🟢 SQL query is valid")
            return {"sql_query": sql_query, "sql_valid": True}
        else:
            return {
                "sql_query": result["corrected_query"],
                "sql_valid": result["valid"],
                "sql_issues": result["issues"],
            }

    def execute_sql(self, state: dict) -> dict:
        """Execute SQL query and return results."""
        query = state["sql_query"]

        if query == "NOT_RELEVANT":
            return {"results": "NOT_RELEVANT"}

        try:
            columns, results = self.db_manager.execute_query(query)
            logger.info(f"🔘 Executed SQL query result: \n{columns}\n{results}")
            return {"cols": columns, "results": results}
        except Exception as e:
            return {"error": str(e)}

    def format_results(self, state: dict) -> dict:
        """Format query results into a human-readable response."""
        question = state["question"]

        # Check if results exist in state
        if "results" not in state:
            return {"answer": "An error occurred: No results available to format."}

        columns = state["cols"]
        results = state["results"]
        sql_query = state["sql_query"]

        if results == "NOT_RELEVANT":
            return {
                "answer": "Sorry, I can only give answers relevant to the database."
            }

        prompt = get_format_results_prompt()

        response = self.llm.invoke(
            prompt,
            provider=self.default_provider,
            question=question,
            cols=columns,
            sql_query=sql_query,
            results=results,
        )
        return {"answer": response}

    def route_based_on_visualization(self, state: dict) -> str:
        """Route the conversation based on the visualization indicator."""
        if state["intent"]["is_visualization"]:
            return "choose_visualization"
        return END

    def default_llm_response(self, state: dict) -> dict:
        """LLM given routing for irrelevnat questions."""
        question = state["question"]
        intent = state["intent"]

        prompt = get_llm_response_prompt()

        response = self.llm.invoke(
            prompt, provider=self.default_provider, question=question, intent=intent
        )
        return {"answer": response}

    def choose_visualization(self, state: dict) -> dict:
        """Choose an appropriate visualization for the data."""
        question = state["question"]
        results = state["results"]
        sql_query = state["sql_query"]

        if results == "NOT_RELEVANT":
            return {
                {
                    "visualization_type": "none",
                    "visualization_reason": "No visualization needed for irrelevant questions.",
                    "visualization_suggestions": "none",
                }
            }

        prompt = get_choose_visualization_prompt()
        output_parser = JsonOutputParser()
        response = self.llm.invoke(
            prompt,
            provider=self.default_provider,
            question=question,
            sql_query=sql_query,
            results=results,
        )
        result = output_parser.parse(response)
        logger.info(f"🚩 Visualization recommendation: \n{result}")

        return {"visualization": result}

    def generate_visualization(self, state: dict) -> dict:
        """Choose an appropriate visualization for the data."""
        question = state["question"]
        columns = state["cols"]
        results = state["results"]
        sql_query = state["sql_query"]
        visualization_type = state["visualization"]["visualization_type"]
        visualization_reason = state["visualization"]["visualization_reason"]
        visualization_suggestions = state["visualization"]["visualization_suggestions"]

        prompt = get_visualization_prompt()

        output_parser = JsonOutputParser()
        response = self.llm.invoke(
            prompt,
            provider=self.default_provider,
            question=question,
            sql_query=sql_query,
            cols=columns,
            results=results,
            visualization=visualization_type,
            visualization_reason=visualization_reason,
            visualization_suggestions=visualization_suggestions,
        )
        parsed_response = output_parser.parse(response)

        return {"visualization_json": parsed_response}
