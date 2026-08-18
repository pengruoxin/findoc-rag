"""Static checks that keep the shipped UI wired to the API it renders.

The workspace is a single hand-written HTML file with no build step, so a typo in
an element id or a CSS class name fails silently in the browser instead of at
import time.  These tests pin the contract between markup, script, and stylesheet.
"""

import re
from pathlib import Path

import pytest

UI_DIR = Path(__file__).resolve().parents[1] / "docs" / "ui"
WORKSPACE = UI_DIR / "workspace-v4.html"


@pytest.fixture(scope="module")
def workspace() -> str:
    return WORKSPACE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script(workspace: str) -> str:
    start = workspace.index("<script>") + len("<script>")
    return workspace[start : workspace.rindex("</script>")]


def test_root_redirect_target_exists() -> None:
    """`GET /` redirects here, so the file must ship under the mounted directory."""
    api_source = (
        Path(__file__).resolve().parents[1] / "src" / "findoc_rag" / "api.py"
    ).read_text(encoding="utf-8")
    targets = re.findall(r'RedirectResponse\("/ui/([^"]+)"\)', api_source)
    assert targets, "api.py no longer redirects to a /ui page"
    for target in targets:
        assert (UI_DIR / target).is_file(), f"/ui/{target} is missing"


def test_every_scripted_element_id_exists(workspace: str, script: str) -> None:
    declared = set(re.findall(r'\bid="([^"]+)"', workspace))
    referenced = set(re.findall(r'\$\("([^"]+)"\)', script))
    assert not referenced - declared, "script reads element ids that the markup never defines"


def test_element_ids_are_unique(workspace: str) -> None:
    declared = re.findall(r'\bid="([^"]+)"', workspace)
    duplicates = {name for name in declared if declared.count(name) > 1}
    assert not duplicates, f"duplicate element ids: {sorted(duplicates)}"


def test_tab_and_panel_ids_pair_up(workspace: str, script: str) -> None:
    names = re.search(r'const TABS = \[([^\]]+)\]', script)
    assert names, "TABS list is missing"
    for name in re.findall(r'"([^"]+)"', names.group(1)):
        assert f'id="tab-{name}"' in workspace
        assert f'id="panel-{name}"' in workspace


def test_generated_class_names_are_styled(workspace: str, script: str) -> None:
    """Guards the class of bug where the script builds `badge-evidenceonly`
    while the stylesheet only defines `badge-evidence-only`."""
    stylesheet = workspace[workspace.index("<style>") : workspace.index("</style>")]
    styled = set(re.findall(r"\.([a-z][a-z0-9-]*)\s*[{,:]", stylesheet))
    used = set(re.findall(r"\bbadge-[a-z0-9-]+", script))
    assert used, "no badge-* class is produced by the script"
    assert not used - styled, f"unstyled badge-* classes: {sorted(used - styled)}"


def test_query_uses_capability_declared_modes(workspace: str, script: str) -> None:
    """Retrieval modes must come from /v1/capabilities, not a hardcoded list, so a
    lexical-only index never offers dense/hybrid options that are certain to fail."""
    assert "cap.modes" in script
    assert "modes.map" in script, "mode options are not derived from the declared modes"
    markup = workspace[: workspace.index("<script>")]
    hardcoded = re.findall(r'<option value="(dense|hybrid|lexical)"', markup)
    assert not hardcoded, f"retrieval modes hardcoded in markup: {sorted(set(hardcoded))}"


def test_endpoints_called_by_ui_are_served_by_the_api(script: str) -> None:
    api_source = (
        Path(__file__).resolve().parents[1] / "src" / "findoc_rag" / "api.py"
    ).read_text(encoding="utf-8")
    called = set(re.findall(r'"(/(?:v1|health)/[a-z:_]+)', script))
    assert called, "the UI calls no API endpoints"
    for path in called:
        assert f'"{path}' in api_source, f"UI calls {path} but api.py does not route it"


def test_dead_prototypes_are_gone() -> None:
    for name in ("workspace-v2.html", "workspace-v3.html", "workspace-wireframe.html"):
        assert not (UI_DIR / name).exists(), f"{name} was superseded and should not return"
