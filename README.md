# EvilFlowers Book Import

A TUI (Text User Interface) application for extracting book metadata from local directories and creating output files with the results.

## Features

- Modern TUI with progress bars and log panel using the rich library
- List directories containing book data
- Extract book metadata (title, authors, publisher, year, ISBN, DOI, summary) using AI models (OpenAI API or local models)
- Process text files to extract metadata using advanced NLP techniques
- Generate Parquet and CSV files with book metadata and directory paths
- Track progress and resume processing from where you left off
- Real-time status updates and statistics

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```bash
# Main usage
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json
```

When you run the application, a TUI (Text User Interface) will be displayed with the following components:
- **Status Panel**: Shows the current status of the application and statistics
- **Progress Panel**: Shows progress bars for the current operations
- **Log Panel**: Shows log messages from the application

The TUI provides real-time feedback on the processing of book directories, including:
- Current status and statistics
- Progress bars for overall processing and individual tasks
- Log messages for detailed information

### Arguments

- `--input-dir`: Path to the directory containing book directories
- `--results-file`: Path to the JSON file where progress data will be stored
- `--api-key`: (Optional) OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable
- `--model-type`: (Optional) The type of AI model to use ("openai" or "ollama"). Defaults to "openai"
- `--model-name`: (Optional) The name of the model to use. Defaults to "gpt-4o" for OpenAI and "mistral" for Ollama
- `--strategy`: (Optional) The import strategy to use ("kramerius" or "dummy"). Defaults to "kramerius"
- `--verbose`: (Optional) Enable verbose output with detailed logging
- `--workers`: (Optional) Number of worker threads for parallel processing. If not provided, uses the default based on CPU count.
- `--limit`: (Optional) Limit the number of directories to process. Useful for debugging
- `--ignore-progress`: (Optional) Ignore existing progress and process all directories from scratch
- `--help`: Show help message

### Extended example

```shell
#!/bin/bash
# Example usage of the EvilFlowers Book Import application

# Set your input directory and results file
INPUT_DIR="/path/to/your/books"
RESULTS_FILE="book_metadata/progress.json"

# Set OpenAI API key as an environment variable
export OPENAI_API_KEY="your_openai_api_key"

# Run the application
python -m evilflowers_importer \
  --input-dir "$INPUT_DIR" \
  --results-file "$RESULTS_FILE" \
  --verbose

echo "Book progress data has been exported to $RESULTS_FILE"
```

## Output

The application generates a JSON file at the specified path with progress data. Each entry in the JSON file contains the following fields:
- dirname: Path to the book directory
- title: Book title
- authors: List of book authors (pipe-separated in the output files)
- publisher: Book publisher
- year: Publication year
- isbn: Book ISBN
- doi: Book DOI
- summary: Book summary
- cover_image: Path to the book cover image (if available)

## Application Architecture

The application uses an object-oriented approach for metadata extraction with an abstraction layer for AI models and a modern TUI interface:

### TUI Components

- **TUIApp**: Main TUI application that manages the layout and display
- **LogPanel**: Panel that displays log messages
- **ProgressManager**: Manager for rich progress bars
- **StatusPanel**: Panel that displays status information
- **RichProgressBar**: A tqdm-compatible progress bar that uses rich

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

1. **Check the Log Panel**: The TUI includes a log panel at the bottom of the screen that displays detailed information about the application's operation. This can help identify issues. The log panel shows the most recent logs and automatically rotates to keep the display manageable.

2. **Detailed Logging**: Use the `--verbose` flag to enable more detailed logging in the log panel, including:
   - Directory listings
   - File processing
   - Metadata extraction

3. **Check Permissions**: Ensure you have read permissions for all directories and files you're trying to process.

4. **Check File Structure**: The application expects a specific directory structure with text files in the Kramerius subdirectory and cover images in the Cover subdirectory.

5. **API Key**: Ensure your OpenAI API key is valid and has sufficient credits.

## Notes

1. By default, this application uses OpenAI's API and requires an API key. You can provide it using the `--api-key` parameter or by setting the `OPENAI_API_KEY` environment variable.

2. You can now use local models through Ollama instead of OpenAI. This provides a free alternative that works offline but may have lower quality results. To use Ollama:
   - Install Ollama from [ollama.ai](https://ollama.ai/)
   - Pull the Mistral model: `ollama pull mistral`
   - Run the application with `--model-type ollama`

3. The application uses OpenAI's GPT-4o model by default to extract metadata from text files. This provides high-quality metadata extraction but requires an internet connection and will incur API usage costs.

4. The AI-extracted metadata may still contain errors. The output files are intended to be manually reviewed and corrected before being used for import to an OPDS server.

5. The application uses a recursive summarization technique to create concise summaries of books, regardless of their length.

6. The application tracks progress and saves it to the specified JSON file. If the script is interrupted, you can run it again with the same input directory and results file to continue from where you left off. Use the `--ignore-progress` flag if you want to start from scratch.

## Progress Tracking

The application now includes a progress tracking feature that allows you to resume processing from where you left off if the script is interrupted. This is particularly useful for large collections of books or when using slower AI models.

### How Progress Tracking Works

1. As each book directory is processed, the application saves the extracted metadata to the specified JSON file.
2. If the script is interrupted (e.g., by a power outage, system crash, or manual termination), you can run it again with the same input directory and results file.
3. The application will automatically detect the existing JSON file, load the previously processed directories, and continue with the remaining ones.
4. Each iteration dumps the progress dataframe to the specified file, ensuring that progress is saved continuously.

### Command-Line Options

- Use the `--ignore-progress` flag if you want to start from scratch and reprocess all directories, ignoring any existing progress.

### Example

```bash
# Initial run
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json

# If interrupted, continue from where you left off
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json

# Start from scratch, ignoring previous progress
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json --ignore-progress
```

## Import Strategies

The application supports different strategies for content availability from various sources. These strategies are responsible for listing input directories and processing individual items, providing text streams and required static files to the rest of the application.

### Available Strategies

- **kramerius**: (Default) This strategy is implemented as `FileSystemKramerius` and is designed for the Kramerius directory structure:
  - Directory name contains OPACID
  - Cover images in Cover/ directory with _p.jpg postfix
  - Text files in Kramerius/OPACID_*/*.txt
  - PDF files in PDF/ directory

- **dummy**: A simple boilerplate strategy that doesn't actually access any real content but serves as a template for implementing new strategies. It provides dummy text content and no static files.

### Using Strategies

You can specify which strategy to use with the `--strategy` command-line argument:

```bash
# Use the default Kramerius strategy
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json

# Use the dummy strategy
python -m evilflowers_importer --input-dir INPUT_DIRECTORY --results-file RESULTS_FILE.json --strategy dummy
```

### Implementing New Strategies

To implement a new content strategy:

1. Create a new class that inherits from `ContentStrategy` in the `evilflowers_importer/strategies.py` file
2. Implement the following methods:
   - `list_items`: Lists all items to process in the base directory
   - `process_item`: Processes a single item and returns its metadata
   - `get_text_content`: Returns a dictionary mapping page numbers to text content
   - `get_cover_image_path`: Returns the path to the cover image, or None if not found
   - `get_pdf_path`: Returns the path to the PDF file, or None if not found
3. Add your strategy to the `StrategyFactory.create_strategy` method
4. Update the choices for the `--strategy` argument in `__main__.py`

The strategy interface is designed to be flexible enough to support different data sources like HTTP, PostgreSQL, WebDav, S3, etc. Each strategy is responsible for both finding items to process and processing them, which allows for more flexibility in supporting different data sources.

#### Example: Implementing a WebDav Strategy

Here's a simplified example of how you might implement a WebDav strategy:

```python
class WebDavStrategy(ContentStrategy):
    """Strategy for accessing content from a WebDav server."""

    def __init__(self):
        self.client = None

    def list_items(self, client, base_dir: str) -> List[str]:
        """List all items to process on the WebDav server."""
        # Use WebDav client to list directories
        items = client.list(base_dir)
        return [item for item in items if client.is_dir(item)]

    def process_item(self, client, item: str, progress_bar=None) -> Dict[str, Any]:
        """Process a single item from the WebDav server."""
        # Get text content, cover image path, and PDF path
        text_content = self.get_text_content(client, item, progress_bar)
        cover_path = self.get_cover_image_path(client, item)
        pdf_path = self.get_pdf_path(client, item)

        # Return metadata
        return {
            'text_content': text_content,
            'cover_path': cover_path,
            'pdf_path': pdf_path
        }

    def get_text_content(self, client, item: str, progress_bar=None) -> Dict[int, str]:
        """Get text content from the WebDav server."""
        # Implementation specific to WebDav
        # ...

    def get_cover_image_path(self, client, item: str) -> Optional[str]:
        """Get the path to the cover image on the WebDav server."""
        # Implementation specific to WebDav
        # ...

    def get_pdf_path(self, client, item: str) -> Optional[str]:
        """Get the path to the PDF file on the WebDav server."""
        # Implementation specific to WebDav
        # ...
```

The AI processing is independent of the content strategy, so you only need to focus on providing the content in the expected format.

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
  --results-file RESULTS_FILE.json \
  --model-type ollama \
  --model-name mistral

# You can also use other Ollama models
python -m evilflowers_importer \
  --input-dir INPUT_DIRECTORY \
  --results-file RESULTS_FILE.json \
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
