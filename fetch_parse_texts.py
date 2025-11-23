import requests
import os
import re
import argparse
import concurrent.futures
from pypdf import PdfReader

# --- CONFIGURATION ---
INPUT_URL_FILE = "high_quality_texts.txt"
OUTPUT_TRAINING_FILE = "zen_training_data.txt"
DOWNLOAD_DIR = "pdf_downloads"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def setup_directories():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)


def download_pdf(url):
    """Downloads a PDF from a URL to the local folder."""
    try:
        local_filename = url.split("/")[-1]
        if not local_filename.lower().endswith(".pdf"):
            local_filename = f"doc_{hash(url)}.pdf"

        path = os.path.join(DOWNLOAD_DIR, local_filename)

        if os.path.exists(path):
            print(f"ℹ️  File already exists, skipping download: {local_filename}")
            return path

        print(f"⬇️  Downloading: {local_filename}...")
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        with open(path, "wb") as f:
            f.write(response.content)
        return path
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file using pypdf."""
    text_content = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
        return "\n\n".join(text_content)
    except Exception as e:
        print(f"⚠️  Error reading PDF {pdf_path}: {e}")
        return ""


def clean_and_format_text(raw_text, min_length=50):
    """
    AGGRESSIVE CLEANING: Specifically targeted at Zen PDF artifacts
    (headers, footnotes, page numbers like '359a', inline citations).
    """
    formatted_entries = []

    # Split the massive PDF string into paragraphs/chunks
    chunks = raw_text.split("\n\n")

    for chunk in chunks:
        # --- PHASE 1: ARTIFACT REMOVAL ---

        # 1. Skip Legal/Copyright Junk
        # If the chunk is just copyright info, skip it entirely.
        if "copyright" in chunk.lower() or "all rights reserved" in chunk.lower():
            continue

        # 2. Cut off specific academic sections
        # If "TRANSLATOR'S NOTES" appears, discard everything after it in this chunk
        if "TRANSLATOR'S NOTES" in chunk:
            chunk = chunk.split("TRANSLATOR'S NOTES")[0]

        # 3. Remove Page Headers/Footers (e.g., "152 THE BLUE CLIFF RECORD")
        # Matches lines starting with digits followed by CAPS text
        chunk = re.sub(r"^\d+\s+[A-Z\s]+.*$", "", chunk, flags=re.MULTILINE)

        # 4. Remove 'Side-note' numbers (e.g., "359a The master said")
        chunk = re.sub(r"^\d+[a-z]?\s+", "", chunk, flags=re.MULTILINE)

        # 5. Remove inline footnote markers (e.g., "illuminates, 126 constantly")
        # Looks for 1-3 digits isolated by spaces, ensuring we don't kill 4-digit years (1998)
        chunk = re.sub(r"(?<=[a-zA-Z.,;])\s+\d{1,3}\s+(?=[a-zA-Z])", " ", chunk)

        # 6. Filter out bottom-of-page footnotes (lines starting with "10." or "a.")
        lines = chunk.split("\n")
        clean_lines = []
        for line in lines:
            # Check if line looks like a footnote list item
            if not re.match(r"^\d+\.\s", line) and not re.match(r"^[a-z]\.\s", line):
                clean_lines.append(line)
        chunk = "\n".join(clean_lines)

        # --- PHASE 2: STANDARD CLEANUP ---

        # Fix hyphenated words (e.g. "enlighten-\nment")
        chunk = re.sub(r"(\w+)-\n(\w+)", r"\1\2", chunk)

        # Collapse newlines and multiple spaces
        chunk = chunk.replace("\n", " ")
        chunk = re.sub(r"\s+", " ", chunk).strip()

        # --- PHASE 3: FINAL FILTER ---

        # Only keep chunks with substance
        if len(chunk) >= min_length:
            # Ensure the chunk doesn't start with a stray digit
            if not chunk[0].isdigit():
                entry = f"<|start_of_text|>\n{chunk}\n<|end_of_text|>\n"
                formatted_entries.append(entry)

    return formatted_entries


def process_url(url):
    """Process a single URL: Download -> Extract -> Clean."""
    pdf_path = download_pdf(url)

    if pdf_path:
        raw_text = extract_text_from_pdf(pdf_path)
        entries = clean_and_format_text(raw_text)
        print(
            f"   ✅ Extracted {len(entries)} training chunks from {os.path.basename(pdf_path)}"
        )
        return entries
    return []


def main():
    parser = argparse.ArgumentParser(description="Download and process PDFs from URLs.")
    parser.add_argument(
        "--file",
        default=INPUT_URL_FILE,
        help=f"File containing URLs (default: {INPUT_URL_FILE})",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_TRAINING_FILE,
        help=f"Output file for training data (default: {OUTPUT_TRAINING_FILE})",
    )
    args = parser.parse_args()

    setup_directories()

    urls = []

    if os.path.exists(args.file):
        with open(args.file, "r") as f:
            urls.extend([line.strip() for line in f if line.strip()])
    elif args.file != INPUT_URL_FILE:
        print(f"Error: Input file '{args.file}' not found.")
        return

    if not urls:
        print(f"Please create a file named '{args.file}' with your PDF links.")
        return

    all_training_data = []

    print(f"🚀 Starting processing for {len(urls)} URLs...\n")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_url, urls)

    for result in results:
        all_training_data.extend(result)

    with open(args.output, "w", encoding="utf-8") as out:
        out.write("\n".join(all_training_data))

    print(
        f"\n✨ DONE! Saved {len(all_training_data)} training examples to '{args.output}'."
    )


if __name__ == "__main__":
    main()
