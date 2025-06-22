# EvilFlowers Book Import

A CLI application for extracting book metadata from local directories and creating output files with the results.

## Features

- List directories containing book data
- Extract book metadata (title, authors, publisher, year, ISBN, DOI, summary) using AI models (OpenAI API or local models)
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
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --output-dir OUTPUT_DIRECTORY
```

### Arguments

- `--input-dir`: Path to the directory containing book directories
- `--output-dir`: Path to the output directory where index.parquet and index.csv will be created
- `--api-key`: (Optional) OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable
- `--model-type`: (Optional) The type of AI model to use ("openai" or "ollama"). Defaults to "openai"
- `--model-name`: (Optional) The name of the model to use. Defaults to "gpt-4o" for OpenAI and "mistral" for Ollama
- `--verbose`: (Optional) Enable verbose output with detailed logging
- `--workers`: (Optional) Number of worker threads for parallel processing. If not provided, uses the default based on CPU count.
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
python -m evilflowers_importer \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --verbose

echo "Book metadata has been exported to $OUTPUT_DIR/index.parquet and $OUTPUT_DIR/index.csv"
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

The application uses an object-oriented approach for metadata extraction with an abstraction layer for AI models:

### Core Components

- **AIExtractor**: Main class that orchestrates the extraction process
- **AIFacade**: Facade for AI functionality, abstracting away the specific AI model implementation
- **BookProcessor**: Processes books and extracts metadata
- **MetadataExtractor**: Extracts metadata using AI models
- **Summarizer**: Creates summaries of text using recursive summarization
- **TextChunker**: Breaks text into manageable chunks for processing
- **RegexExtractor**: Extracts ISBN and DOI using regular expressions

### AI Model Abstraction

- **AIModelInterface**: Abstract base class for AI models
- **OpenAIModel**: Implementation for OpenAI API
- **OllamaModel**: Implementation for local Ollama models
- **AIModelFactory**: Factory for creating AI model instances

This modular design makes the code more maintainable and extensible, allowing you to use either OpenAI's API or local models like Ollama's Mistral.

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

1. By default, this application uses OpenAI's API and requires an API key. You can provide it using the `--api-key` parameter or by setting the `OPENAI_API_KEY` environment variable.

2. You can now use local models through Ollama instead of OpenAI. This provides a free alternative that works offline but may have lower quality results. To use Ollama:
   - Install Ollama from [ollama.ai](https://ollama.ai/)
   - Pull the Mistral model: `ollama pull mistral`
   - Run the application with `--model-type ollama`

3. The application uses OpenAI's GPT-4o model by default to extract metadata from text files. This provides high-quality metadata extraction but requires an internet connection and will incur API usage costs.

4. The AI-extracted metadata may still contain errors. The output files are intended to be manually reviewed and corrected before being used for import to an OPDS server.

5. The application uses a recursive summarization technique to create concise summaries of books, regardless of their length.

## Using Local Models with Ollama

Ollama is a local model runner that allows you to run AI models on your own machine. The application supports using Ollama models, particularly the Mistral model which works well with Slovak, Czech, and English languages.

### Installing Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Mistral model:
   ```
   ollama pull mistral
   ```

### Running with Ollama

```bash
# Run with Ollama's Mistral model
python -m evilflowers_importer \
  --input-dir INPUT_DIRECTORY \
  --output-dir OUTPUT_DIRECTORY \
  --model-type ollama \
  --model-name mistral

# You can also use other Ollama models
python -m evilflowers_importer \
  --input-dir INPUT_DIRECTORY \
  --output-dir OUTPUT_DIRECTORY \
  --model-type ollama \
  --model-name llama2
```

### Testing AI Models

The application includes a test script to try out different AI models:

```bash
# Test OpenAI model
python -m evilflowers_importer.test_ai_models \
  --model-type openai \
  --prompt "Summarize this book about artificial intelligence"

# Test Ollama model
python -m evilflowers_importer.test_ai_models \
  --model-type ollama \
  --model-name mistral \
  --prompt "Summarize this book about artificial intelligence"

# Test with a specific directory
python -m evilflowers_importer.test_ai_models \
  --model-type ollama \
  --directory "/path/to/book/directory"
```
