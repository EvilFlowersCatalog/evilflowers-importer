# EvilFlowers WebDAV Book Import

A CLI application for extracting book metadata from WebDAV directories and creating an Excel file with the results.

## Features

- Connect to WebDAV servers with authentication
- List directories containing book data
- Extract book metadata (title, authors, publisher, year, ISBN, DOI, summary) using OpenAI API
- Process text files to extract metadata using advanced NLP techniques
- Generate an Excel file with book metadata and directory paths

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### WebDAV Mode

```bash
python -m evilflowers_webdav_import --webdav-url URL --username USERNAME --password PASSWORD --output OUTPUT_FILE.xlsx
```

#### Arguments

- `--webdav-url`: URL of the WebDAV server
- `--username`: WebDAV username
- `--password`: WebDAV password
- `--output`: Path to the output Excel file
- `--api-key`: (Optional) OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable
- `--verbose`: (Optional) Enable verbose output
- `--help`: Show help message

### Local Mode

You can also process local directories instead of WebDAV:

```bash
python -m evilflowers_webdav_import.process_local --input-dir LOCAL_DIR --output OUTPUT_FILE.xlsx
```

#### Arguments

- `--input-dir`: Path to the directory containing book directories
- `--output`: Path to the output Excel file
- `--api-key`: (Optional) OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable
- `--verbose`: (Optional) Enable verbose output
- `--help`: Show help message

### Extended example

```shell
#!/bin/bash
# Example usage of the EvilFlowers WebDAV Book Import application

# Replace these values with your actual WebDAV server details
WEBDAV_URL="https://your-webdav-server.com/webdav"
USERNAME="your_username"
PASSWORD="your_password"
OUTPUT_FILE="book_metadata.xlsx"
OPENAI_API_KEY="your_openai_api_key"  # Optional: You can also set this as an environment variable

# Run the application in WebDAV mode
python -m evilflowers_webdav_import \
  --webdav-url "$WEBDAV_URL" \
  --username "$USERNAME" \
  --password "$PASSWORD" \
  --output "$OUTPUT_FILE" \
  --api-key "$OPENAI_API_KEY" \
  --verbose

echo "Book metadata has been exported to $OUTPUT_FILE"

# Example for local mode
LOCAL_DIR="/path/to/your/books"
LOCAL_OUTPUT_FILE="local_book_metadata.xlsx"

# Run the application in Local mode
python -m evilflowers_webdav_import.process_local \
  --input-dir "$LOCAL_DIR" \
  --output "$LOCAL_OUTPUT_FILE" \
  --api-key "$OPENAI_API_KEY" \
  --verbose

echo "Local book metadata has been exported to $LOCAL_OUTPUT_FILE"
```

## Output

The application generates an Excel file with the following columns:
- dirname: Path to the book directory
- title: Book title
- authors: List of book authors
- publisher: Book publisher
- year: Publication year
- isbn: Book ISBN
- doi: Book DOI
- summary: Book summary
- cover_image: Path to the book cover image (if available)

## AI Module Architecture

The application uses an object-oriented approach for metadata extraction:

- **AIExtractor**: Main class that orchestrates the extraction process
- **BookProcessor**: Processes books and extracts metadata
- **MetadataExtractor**: Extracts metadata using OpenAI API
- **Summarizer**: Creates summaries of text using recursive summarization
- **TextChunker**: Breaks text into manageable chunks for processing
- **RegexExtractor**: Extracts ISBN and DOI using regular expressions

This modular design makes the code more maintainable and extensible.

## Notes

1. This application requires an OpenAI API key to function. You can provide it using the `--api-key` parameter or by setting the `OPENAI_API_KEY` environment variable.

2. The application uses OpenAI's GPT-4o model to extract metadata from text files. This provides high-quality metadata extraction but requires an internet connection and will incur API usage costs.

3. The AI-extracted metadata may still contain errors. The Excel file is intended to be manually reviewed and corrected before being used for import to an OPDS server.

4. The application uses a recursive summarization technique to create concise summaries of books, regardless of their length.
