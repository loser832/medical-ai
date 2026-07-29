from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module_tree(filename: str) -> ast.Module:
    source = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
    return ast.parse(source, filename=filename)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


class IntegrationWiringTests(unittest.TestCase):
    def test_all_complex_path_agents_receive_trace_recorder(self):
        tree = _module_tree("md_agent.py")
        process = _function(tree, "process_diff_query")
        parameter_names = [argument.arg for argument in process.args.args]
        self.assertIn("trace_recorder", parameter_names)
        self.assertIn("final_response_instruction", parameter_names)

        agent_calls = [
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Agent"
        ]
        self.assertEqual(len(agent_calls), 3)
        for call in agent_calls:
            self.assertIn("trace_recorder", {keyword.arg for keyword in call.keywords})

    def test_stream_callback_mirrors_events_and_passes_recorder(self):
        tree = _module_tree("multi_agent.py")
        callback = _function(tree, "create_callback")
        self.assertIn("trace_recorder", [argument.arg for argument in callback.args.args])

        wrapper = _function(tree, "process_diff_query_with_callback")
        process_calls = [
            node
            for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "process_diff_query"
        ]
        self.assertEqual(len(process_calls), 1)
        self.assertIn(
            "trace_recorder",
            {keyword.arg for keyword in process_calls[0].keywords},
        )


if __name__ == "__main__":
    unittest.main()
