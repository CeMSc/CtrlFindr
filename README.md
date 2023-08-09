# CtrlFindr

**CtrlFindr** is a natural language processing and **content analysis toolkit** designed to help you analyze and extract insights from text documents. The toolkit is at the moment implemented in a Jupyter Notebook, providing you with an interactive environment to work with the code and visualize the results. The current version works with English text, but it can easily be adapted for other languages.

In the coming months, I plan to provide a version of the code with a User Interface (UI) and an executable file for easier use.

# Features

- Text preprocessing (lowercase conversion, sentence splitting),
- Keyword analysis,
- Co-occurrence analysis,
- Sentiment analysis (using VADER SentimentIntensityAnalyzer from nltk),
- Customizable search strings for automating content analysis of natural language (to be filled in the provided Assessment_framework.xlsx template)
- Export results to TSV files (number of positive findings per document, positive findings per document as booleans, positive findings as percentage of sentences within documents, and sentiment analysis of positive findings)

# Dependencies

    Python 3.6 or higher
    Pandas
    Numpy
    NLTK

# Installation

    Clone the CtrlFindr repository to your local machine:
```
git clone https://github.com/username/CtrlFindr.git
``` 

Navigate to the CtrlFindr directory:
```
cd CtrlFindr
```

Install the required dependencies:
```
pip install -r requirements.txt
```

Fill the search strings and taxonomy in the Assessment_framework.xlsx

Open the CtrlFindr.ipynb file in Jupyter Notebook to start analyzing your text files.

# Usage

- Place the text files you want to analyze in the TXT folder or adjust the file paths in the txt_to_dataframe() function.
- Prepare the Assessment_framework.xlsx file containing the search strings, co-occurrences, document conditionals, and taxonomy as specified in the create_dataframes() function.
- Customize the code to meet your specific requirements (e.g., utilize optional functions).
- Execute the cells in the Jupyter Notebook in the order they appear.
- The results will be saved as TSV files (Output_PosCount.tsv, Output_PosCountBoolean.tsv, Output_PosPercent.tsv, Output_Sentiment.tsv, and Output_Sentences.tsv) in the same directory as the Jupyter Notebook.

- Check the **Jupiter Notebook Example folder** for a sample analysis of two text files.

# License

This project is licensed under the GNU General Public License v3.0. Please read the LICENSE file for more information.

# How to cite

Scartozzi, Cesare M. (2023.) CtrlFindr (Version no.). Available from https://github.com/CeMSc/CtrlFindr.
