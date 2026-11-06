"""pytest plugin: collect once, keep only the tests listed (one key per line) in the file named by $SELECT_TESTS.
Passing many node ids on the command line makes pytest re-collect each file once per id, which is slower than running everything."""
import os

from collect import nodeid_to_key


def pytest_collection_modifyitems(config, items):
    path = os.environ.get("SELECT_TESTS")
    if not path:
        return
    keep = set(open(path).read().split())
    selected = [it for it in items if nodeid_to_key(it.nodeid.split("tests/")[-1]) in keep]
    config.hook.pytest_deselected(items=[it for it in items if it not in selected])
    items[:] = selected
