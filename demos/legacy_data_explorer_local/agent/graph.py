from langgraph.graph import StateGraph, START, END
from agents.legacy_data_explorer_local.agent.states import InputState, OutputState
from agents.legacy_data_explorer_local.agent.nodes import Nodes


class Graph:
    def __init__(self):
        self.agent_nodes = Nodes()

    def create_workflow(self) -> StateGraph:
        """Create and configure the workflow graph."""
        workflow = StateGraph(input=InputState, output=OutputState)

        # Add nodes to the graph
        workflow.add_node(
            "parse_question_intent", self.agent_nodes.parse_question_intent
        )
        workflow.add_node(
            "parse_question_prompt", self.agent_nodes.parse_question_prompt
        )
        workflow.add_node("schema_description", self.agent_nodes.schema_description)
        workflow.add_node("get_random_subsample", self.agent_nodes.get_random_subsample)
        workflow.add_node("generate_sql", self.agent_nodes.generate_sql)
        workflow.add_node("validate_and_fix_sql", self.agent_nodes.validate_and_fix_sql)
        workflow.add_node("execute_sql", self.agent_nodes.execute_sql)
        workflow.add_node("format_results", self.agent_nodes.format_results)
        workflow.add_node("choose_visualization", self.agent_nodes.choose_visualization)
        workflow.add_node(
            "generate_visualization",
            self.agent_nodes.generate_visualization,
        )
        workflow.add_node("default_llm_response", self.agent_nodes.default_llm_response)

        # Define edges
        workflow.add_edge(START, "parse_question_intent")
        workflow.add_conditional_edges(
            "parse_question_intent",
            self.agent_nodes.route_based_on_intent,
            ["parse_question_prompt", "schema_description", "default_llm_response"],
        )
        workflow.add_edge("parse_question_prompt", "get_random_subsample")
        workflow.add_edge("get_random_subsample", "generate_sql")
        workflow.add_edge("generate_sql", "validate_and_fix_sql")
        workflow.add_edge("validate_and_fix_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_results")
        # workflow.add_edge("execute_sql", "choose_visualization")
        workflow.add_conditional_edges(
            "execute_sql",
            self.agent_nodes.route_based_on_visualization,
            ["choose_visualization", END],
        )
        workflow.add_edge("choose_visualization", "generate_visualization")
        workflow.add_edge("generate_visualization", END)
        workflow.add_edge("format_results", END)
        workflow.add_edge("schema_description", END)
        workflow.add_edge("default_llm_response", END)

        return workflow

    def returnGraph(self):
        return self.create_workflow().compile()

    def run_agent_nodes(self, question: str) -> dict:
        """Run the SQL agent workflow and return the formatted answer and visualization recommendation."""
        app = self.create_workflow().compile()
        result = app.invoke(input={"question": question})
        return result
