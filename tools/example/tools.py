from langchain_core.tools import tool


@tool
def example_tool(value: str) -> str:
    """Return the supplied value."""
    return value


def get_tools():
    return [example_tool]
