import os
import asyncio
from contextlib import asynccontextmanager
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from src.config.logger import logger # Assuming logger is accessible here

@asynccontextmanager
async def tavily_mcp_client_session():
    """Provides an initialized Tavily MCP ClientSession as an asynchronous context manager."""
    logger.info("Attempting to start Tavily MCP server and session...")
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY environment variable not set.")
        # Raising an error here ensures the supervisor knows about the failure.
        raise ValueError("TAVILY_API_KEY must be set to use the Tavily MCP server.")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@0.1.4"],
        env={"TAVILY_API_KEY": tavily_api_key},
        # Consider adding stderr handling if npx errors are not visible
        # stderr=asyncio.subprocess.PIPE  # Example, would need log streaming
    )
    logger.info("Tavily MCP server parameters configured.")

    session_instance = None
    try:
        async with stdio_client(server_params) as (read, write):
            logger.info("Tavily MCP stdio_client started.")
            async with ClientSession(read, write) as mcp_session:
                await mcp_session.initialize()
                logger.info("Tavily MCP ClientSession initialized successfully.")
                session_instance = mcp_session
                yield session_instance
    except ValueError as ve: # Catch specific startup errors like missing API key
        logger.error(f"Failed to initialize Tavily MCP session due to ValueError: {ve}")
        raise # Re-raise to ensure calling code handles it
    except Exception as e:
        logger.error(f"An unexpected error occurred during Tavily MCP session setup: {e}", exc_info=True)
        # Depending on desired robustness, you might raise a custom error or handle differently.
        raise RuntimeError(f"Failed to establish Tavily MCP session: {e}")
    finally:
        if session_instance:
            logger.info("Tavily MCP ClientSession context exited.")
        else:
            logger.warning("Tavily MCP ClientSession context exited without successful initialization.") 