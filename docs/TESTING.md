<!-- generated-by: gsd-doc-writer -->
# TESTING.md

## Test framework and setup

The project uses **pytest** as its primary testing framework.

### Prerequisites
To set up the testing environment, ensure all dependencies are installed, including `pytest`:

```bash
pip install -r requirments.txt
```

*(Note: The project uses a file named `requirments.txt` for dependency management.)*

### Setup
The test suite utilizes `pytest` fixtures for database sandboxing. The `tests/conftest.py` file contains these fixtures, providing a temporary SQLite database populated with sample data for testing purposes.

## Running tests

You can run the full test suite using the `pytest` command from the project root:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_db_lookup.py
```

To run tests and see output (even for passing tests), use the `-s` flag:

```bash
pytest -s
```

## Writing new tests

When contributing new tests, please follow these conventions:

- **File Naming**: Test files should be named with the `test_` prefix (e.g., `tests/test_feature_name.py`).
- **Function Naming**: Test functions should also start with `test_` (e.g., `def test_function_behavior():`).
- **Fixtures**: Use existing fixtures from `tests/conftest.py` whenever possible. Key fixtures include:
    - `temp_db`: Provides a path to a temporary SQLite database matching the production schema.
    - `vid_name_present`: Returns a video filename stem that exists in the temporary database.
    - `vid_name_missing`: Returns a video filename stem that does NOT exist in the temporary database.
- **Shared logic**: Any shared setup or global fixtures should be added to `tests/conftest.py`.

## Coverage requirements

No coverage threshold is currently configured for this project. Developers are encouraged to maintain high coverage for new features, but no automated enforcement is in place.

## CI integration

No CI/CD pipeline is currently detected for automated test execution. Tests should be run manually before submitting pull requests.
