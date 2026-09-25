import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "projects" / "07-agentic-ai" / "agent.py"
spec = importlib.util.spec_from_file_location("agent_demo", PATH)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

def test_echo_tool():
    result = agent.run("echo", "hello")
    assert result.tool == "echo"
    assert result.output == "hello"

def test_unknown_tool_is_blocked():
    try:
        agent.run("not-allowed")
        assert False, "Unknown tool should be rejected"
    except ValueError:
        assert True
