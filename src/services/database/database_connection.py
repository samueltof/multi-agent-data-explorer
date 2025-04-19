import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from src.config.logger import logger

class DatabaseConnection(ABC):
    @abstractmethod
    def connect(self) -> Any:
        pass

    @abstractmethod
    def execute_query(self, query: str) -> Tuple[List[str], List[Any]]:
        logger.info("⎄ Executing query...")
        pass

    @abstractmethod
    def execute_query_df(self, query: str) -> pd.DataFrame:
        logger.info("⎄ Executing query as DataFrame...")
        pass
