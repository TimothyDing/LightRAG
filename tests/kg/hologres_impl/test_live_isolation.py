"""Static guards for the isolated Hologres live integration suite."""

import ast
import re
from pathlib import Path


_PACKAGE_DIR = Path(__file__).parents[3] / "lightrag" / "kg" / "hologres"
_LIVE_GLOB = "test_hologres_live*.py"


def _constant_vector_query_pattern():
    function = "approx_cosine_distance" + "("
    literal = re.escape(function)
    return (
        re.compile(literal + r"\s*\$"),
        re.compile(literal + r"\s*ARRAY", re.IGNORECASE),
        re.compile(literal + r"\s*'\{"),
    )


def test_hologres_sql_never_uses_a_constant_only_vector_operand():
    patterns = _constant_vector_query_pattern()
    offenders = [
        path
        for path in _PACKAGE_DIR.rglob("*.py")
        if any(pattern.search(path.read_text()) for pattern in patterns)
    ]

    assert offenders == []


def test_live_modules_import_only_hologres_backends_and_carry_live_markers():
    live_paths = sorted(Path(__file__).parent.glob(_LIVE_GLOB))
    assert len(live_paths) == 7

    for path in live_paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else ""
            assert "postgres" not in (module or "").lower()
            assert not (module or "").startswith("tests.kg.test_")
            if isinstance(node, ast.Import):
                assert all("postgres" not in name.name.lower() for name in node.names)

        marker = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        )
        marker_names = [
            attribute.attr
            for element in marker.value.elts
            if isinstance(element, ast.Attribute)
            for attribute in (element,)
        ]
        assert "integration" in marker_names
        assert "hologres_live" in marker_names
