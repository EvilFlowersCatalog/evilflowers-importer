# EvilFlowers Book Import

A CLI application for extracting book metadata from local directories and creating output files with the results.

## Features

- List directories containing book data
- Extract book metadata (title, authors, publisher, year, ISBN, DOI, summary) using OpenAI API
- Process text files to extract metadata using advanced NLP techniques
- Generate Parquet and CSV files with book metadata and directory paths

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```bash
# Main usage
python -m evilflowers_webdav_import --input-dir INPUT_DIRECTORY --output-dir OUTPUT_DIRECTORY

# Alternative using process_local module directly
python -m evilflowers_webdav_import.process_local --input-dir INPUT_DIRECTORY --output-dir OUTPUT_DIRECTORY
```

### Arguments

- `--input-dir`: Path to the directory containing book directories
- `--output-dir`: Path to the output directory where index.parquet and index.csv will be created
- `--api-key`: (Optional) OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable
- `--verbose`: (Optional) Enable verbose output with detailed logging
- `--limit`: (Optional) Limit the number of directories to process. Useful for debugging
- `--help`: Show help message

### Extended example

```shell
#!/bin/bash
# Example usage of the EvilFlowers Book Import application

# Set your input and output directories
INPUT_DIR="/path/to/your/books"
OUTPUT_DIR="book_metadata"

# Set OpenAI API key as an environment variable
export OPENAI_API_KEY="your_openai_api_key"

# Run the application
python -m evilflowers_webdav_import \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --verbose

echo "Book metadata has been exported to $OUTPUT_DIR/index.parquet and $OUTPUT_DIR/index.csv"

# Alternative using process_local module directly
ALTERNATIVE_INPUT_DIR="/path/to/other/books"
ALTERNATIVE_OUTPUT_DIR="other_book_metadata"

python -m evilflowers_webdav_import.process_local \
  --input-dir "$ALTERNATIVE_INPUT_DIR" \
  --output-dir "$ALTERNATIVE_OUTPUT_DIR" \
  --verbose

echo "Book metadata has been exported to $ALTERNATIVE_OUTPUT_DIR/index.parquet and $ALTERNATIVE_OUTPUT_DIR/index.csv"
```

## Output

The application generates both Parquet and CSV files in the specified output directory with the following columns:
- dirname: Path to the book directory
- title: Book title
- authors: List of book authors (pipe-separated in the output files)
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

## Troubleshooting

If you encounter issues, try the following:

1. **Detailed Logging**: Use the `--verbose` flag to enable detailed logging, including:
   - Directory listings
   - File processing
   - Metadata extraction

2. **Check Permissions**: Ensure you have read permissions for all directories and files you're trying to process.

3. **Check File Structure**: The application expects a specific directory structure with text files in the Kramerius subdirectory and cover images in the Cover subdirectory.

4. **API Key**: Ensure your OpenAI API key is valid and has sufficient credits.

## Notes

1. This application requires an OpenAI API key to function. You can provide it using the `--api-key` parameter or by setting the `OPENAI_API_KEY` environment variable.

2. The application uses OpenAI's GPT-4o model to extract metadata from text files. This provides high-quality metadata extraction but requires an internet connection and will incur API usage costs.

3. The AI-extracted metadata may still contain errors. The output files are intended to be manually reviewed and corrected before being used for import to an OPDS server.

4. The application uses a recursive summarization technique to create concise summaries of books, regardless of their length.
