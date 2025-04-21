3. Enter questions when prompted. Make sure to include both the brand name and model name in your question for best results.

Example questions:
- "What is the battery capacity of the Samsung Galaxy S23?"
- "Does the iPhone 14 Pro Max have NFC?"
- "How many rear cameras does the Google Pixel 7 have?"
- "What is the screen size and refresh rate of the OnePlus 11?"

4. Type 'quit' or 'exit' to end the session.

## How It Works

The system uses a Retrieval-Augmented Generation (RAG) approach:

1. **Data Preprocessing**:
- Loads CSV data and performs type conversion and cleaning
- Creates a simplified base model name for better matching

2. **Context Creation**:
- Generates detailed textual descriptions for each smartphone
- Formats specifications in a way that's conducive to question answering

3. **Query Processing**:
- Extracts brand and model information from user questions
- Uses regular expressions for flexible matching

4. **Retrieval**:
- Identifies relevant smartphones based on the query
- Retrieves and combines context snippets for matched phones

5. **Question Answering**:
- Uses a RoBERTa model fine-tuned on SQuAD2
- Tokenizes the question and context together
- Predicts start and end indices of the answer in the context
- Extracts and returns the answer text

## Limitations

- The system requires both brand name and model name to be present in the question for successful retrieval
- Performance depends on the quality and completeness of the CSV data
- The QA model works best with factual questions about specifications rather than subjective comparisons

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses the [RoBERTa](https://huggingface.co/deepset/roberta-base-squad2) model fine-tuned on SQuAD2 by deepset.
- The RAG approach is inspired by research from Facebook AI and other leading NLP research organizations.
