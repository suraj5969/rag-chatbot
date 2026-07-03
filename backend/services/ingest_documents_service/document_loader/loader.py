import concurrent.futures
from pathlib import Path
from typing import Any

from helpers.log import get_logger
from tqdm import tqdm
from unstructured.partition.auto import partition

from services.ingest_documents_service.document import Document

logger = get_logger(__name__)


class DirectoryLoader:
    """Load documents from a directory."""

    def __init__(
        self,
        path: Path,
        glob: str = "**/[!.]*",
        recursive: bool = False,
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
        **partition_kwargs: Any,
    ):
        """Initialize with a path to directory and how to glob over it.

        Args:
            path: Path to directory.
            glob: Glob pattern to use to find files. Defaults to "**/[!.]*"
               (all files except hidden).
            recursive: Whether to recursively search for files. Defaults to False.
            show_progress: Whether to show a progress bar. Defaults to False.
            use_multithreading: Whether to use multithreading. Defaults to False.
            max_concurrency: The maximum number of threads to use. Defaults to 4.
            partition_kwargs: Keyword arguments to pass to unstructured `partition` function.
        """
        self.path = path
        self.glob = glob
        self.recursive = recursive
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.partition_kwargs = partition_kwargs

    def load(self) -> list[Document]:
        """Load documents."""
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not self.path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        docs: list[Document] = []
        items = list(self.path.rglob(self.glob) if self.recursive else self.path.glob(self.glob))

        pbar = None
        if self.show_progress:
            pbar = tqdm(total=len(items))

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                executor.map(lambda item: self.load_file(item, docs, pbar), items)
        else:
            for i in items:
                self.load_file(i, docs, pbar)

        if pbar:
            pbar.close()

        return docs

    # File extensions that can be read directly as plain text — no NLP pipeline needed.
    _PLAIN_TEXT_SUFFIXES = {".md", ".txt", ".rst", ".csv"}

    def load_file(self, doc_path: Path, docs: list[Document], pbar: Any | None) -> None:
        """
        Load document from the specified path.

        For plain-text formats (.md, .txt, etc.) the file is read directly as UTF-8,
        which is orders of magnitude faster than routing through unstructured (which
        triggers an NLTK download on first use and runs a full NLP pipeline).

        For all other formats the unstructured `partition` function is used, which
        auto-detects the file type via libmagic and applies the appropriate partitioner.

        Args:
            doc_path (str): The path to the document.
            docs: List of documents to append to.
            pbar: Progress bar. Defaults to None.

        """
        if doc_path.is_file():
            try:
                logger.debug(f"Processing file: {str(doc_path)}")

                if doc_path.suffix.lower() in self._PLAIN_TEXT_SUFFIXES:
                    # Fast path: read plain-text files directly — avoids unstructured
                    # and the NLTK download it triggers for Markdown.
                    text = doc_path.read_text(encoding="utf-8", errors="replace")
                else:
                    # Slow path: use unstructured for binary/complex formats (PDF, DOCX, …).
                    # The `partition` function detects the file type with libmagic and routes
                    # it to the appropriate partitioning function.
                    elements = partition(filename=str(doc_path), **self.partition_kwargs)
                    text = "\n\n".join([str(el) for el in elements])

                docs.extend([Document(page_content=text, metadata={"source": str(doc_path)})])
            finally:
                if pbar:
                    pbar.update(1)


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    docs_path = root_folder / "docs"
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.md",
        recursive=True,
        use_multithreading=True,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
