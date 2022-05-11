sources = nannyml

.PHONY: test format lint unittest coverage pre-commit clean

test: format lint unittest

BROWSER = python -m webbrowser

format:
	isort $(sources) tests
	black $(sources) tests

lint:
	flake8 $(sources) tests
	mypy $(sources) tests
	doc8 docs

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage

docs: ## generate Sphinx HTML documentation, including API docs
	rm -rf docs/nannyml
	#sphinx-apidoc -o docs/nannyml nannyml tests nannyml/datasets/data
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html


# For some reason make docs doesn't work!
# $ make docs
# make: 'docs' is up to date.
#
# Let's define a small python script to serve docs
define BROWSER_PYSCRIPT
from os import chdir
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Hardcode/Move to appropriate directory
chdir('./docs/_build/html/')
print("http://localhost:8001/")
# Create server object listening the port 8001
server = HTTPServer(server_address=('', 8001), RequestHandlerClass=SimpleHTTPRequestHandler)
# Start the web server
server.serve_forever()
endef
export BROWSER_PYSCRIPT

servedocs: ## generate Sphinx HTML documentation, including API docs
	rm -rf docs/nannyml
	#sphinx-apidoc -o docs/nannyml nannyml tests nannyml/datasets/data
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	python -c "$$BROWSER_PYSCRIPT"

# Use Ctrl+C to stop serving docs.
