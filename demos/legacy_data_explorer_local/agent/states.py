from typing import List, Any, Annotated, Dict, Optional
from typing_extensions import TypedDict
import operator

class InputState(TypedDict):
    question: str
    database_name: str
    intent : Dict[str, Any]
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    cols: List[Any]
    results: List[Any]
    visualization: Dict[str, Any]
    samples: Optional[Dict[str, List[Any]]]

class OutputState(TypedDict):
    intent : Dict[str, Any]
    parsed_question: Dict[str, Any]
    unique_nouns: List[str]
    sql_query: str
    sql_valid: bool
    sql_issues: str
    cols: List[Any]
    results: List[Any]
    answer: Annotated[str, operator.add]
    error: str
    visualization: Dict[str, Any]
    visualization_json: Dict[str, Any]