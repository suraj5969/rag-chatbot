install_dependencies:
	echo "Installing..."
	uv venv
	uv sync

migrate_db:
	cd backend && uv run python migration.py

start_llama_server:
	docker compose up -d

stop_llama_server:
	docker compose down

setup: install_dependencies migrate_db start_llama_server

start:
	sh start.sh

update:
	uv lock --upgrade
	uv sync

tidy:
	uv run ruff format --exclude=.venv .
	uv run ruff check --exclude=.venv . --fix

test:
	uv run pytest --log-cli-level=DEBUG --capture=tee-sys -v

check-formatting:
	uv run ruff format . --check

clean:
	echo "Cleaning uv environment..."
	rm -rf .venv
	echo "Cleaning all compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	echo "Cleaning the cache..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
