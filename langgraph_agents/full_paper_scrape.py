import arxiv
import fitz  # PyMuPDF library, can convert .pdf file to text
import os
import csv
import time
import sqlite3
from tqdm import tqdm


def fetch_and_read_paper(arxiv_id):
    # Setup the arXiv client and search for the specific paper
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        # Get the first result
        paper = next(client.results(search))
    except StopIteration:
        print("Paper not found.")
        return

    # Print Metadata
    print(f"Title: {paper.title}")
    print(f"Authors: {', '.join([author.name for author in paper.authors])}")
    print(f"Published: {paper.published.date()}\n")

    # Download temporary PDF
    pdf_filename = paper.download_pdf()

    # Extract text from the PDF with fitz
    doc = fitz.open(pdf_filename)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()

    doc.close()

    # Delete the PDF
    os.remove(pdf_filename)

    # Preview of the full text
    print(full_text[:1000] + "\n...[TEXT TRUNCATED]...")


def setup_database(db_name="ml_papers.db"):
    # Connect to the database file (create if doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create the table papers
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            published_date TEXT,
            full_text TEXT
        )
    """
    )
    conn.commit()
    return conn


def scrape_to_sqlite(max_papers=100):
    print(f"Setting up SQLite database and searching for {max_papers} papers...")

    # Initialize database
    conn = setup_database()
    cursor = conn.cursor()

    # Setup arXiv client
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.LG",  # This query gets all papers in field of ML
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    # Found paper in ML
    results = client.results(search)

    # tqdm used here to print progess bar
    for paper in tqdm(
        results, total=max_papers, desc="Downloading Papers", unit="paper"
    ):
        # Get individual paper
        arxiv_id = paper.get_short_id()
        title = paper.title
        authors = ", ".join([author.name for author in paper.authors])
        date = str(paper.published.date())

        # Download .pdf file, convert to text, and insert to database
        try:
            pdf_filename = paper.download_pdf()
            doc = fitz.open(pdf_filename)
            full_text = "".join([page.get_text() for page in doc])
            doc.close()
            os.remove(pdf_filename)

            cursor.execute(
                """
                INSERT OR REPLACE INTO papers (arxiv_id, title, authors, published_date, full_text)
                VALUES (?, ?, ?, ?, ?)
            """,
                (arxiv_id, title, authors, date, full_text),
            )

            conn.commit()

        except Exception as e:
            tqdm.write(f"Error on {arxiv_id}: {e}")
            if "pdf_filename" in locals() and os.path.exists(pdf_filename):
                os.remove(pdf_filename)

        # Make sure to make this long enough so API does not break
        time.sleep(0.2233)
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    # Test single paper with Attention is All you Need.
    # fetch_and_read_paper("1706.03762")

    # Test multi-paper scraping
    scrape_to_sqlite(max_papers=50)  # Change max_paper to scrap more papers.
