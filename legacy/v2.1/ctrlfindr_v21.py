"""CtrlFindr v2.1 analysis functions, extracted verbatim from legacy/v2.1/CtrlFindr.ipynb (cell 3).

Kept only as a reference implementation: the v3 test suite runs it on the committed
v2.1 sentences to prove the golden fixtures are reproducible, and tools/compare_outputs.py
uses it to show what changed between versions. Do not fix bugs here.

Differences from the notebook, all outside the analysis logic:
- NLTK is imported lazily so the module loads without NLTK data downloads;
- the notebook's global settings are module-level variables below;
- `run_analysis_from_sentences` replays the analysis half of the notebook's `main()`.
"""
import os
import re
import time

import pandas as pd
import sqlite3


class _LazyNLTK:
    def __getattr__(self, name):
        import nltk as _nltk
        return getattr(_nltk, name)


nltk = _LazyNLTK()


class _LazyVader:
    _sia = None

    def polarity_scores(self, text):
        if self._sia is None:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._sia = SentimentIntensityAnalyzer()
        return self._sia.polarity_scores(text)


sent_i = _LazyVader()

ASSESSMENT_FRAMEWORK_DIRECTORY = None
SENTIMENT_ANALYSIS = True

## LOAD AND PREPARE DATA FOR ANALYSIS
# IMPORT TXT FILES FROM TEXT_FILES_DIRECTORY
def txt_to_sql(directory, db_path):
    data = {"document": [], "text": []}

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data["document"].append(filename.replace(".txt", ""))
                data["text"].append(content)
    df = pd.DataFrame(data)
    conn = sqlite3.connect(db_path)
    df.to_sql('documents', conn, if_exists='replace', index=False)
    print(df.shape)
    conn.close()
    return df

# IMPORT ASSESSMENT FRAMEWORK
def create_dataframes(ods_file):
    variables = process_dataframe(pd.read_excel(ods_file, sheet_name='variables', engine='odf'))
    set_search_strings = process_dataframe(pd.read_excel(ods_file, sheet_name='search_strings', engine='odf'))
    set_co_occurrences = process_dataframe(pd.read_excel(ods_file, sheet_name='co_occurrences', engine='odf'))
    set_doc_conditionals = process_dataframe(pd.read_excel(ods_file, sheet_name='doc_conditionals', engine='odf'))
    set_keywords = process_dataframe(pd.read_excel(ods_file, sheet_name='taxonomy', engine='odf'), drop_na=False)
    return variables, set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords

# Helper function to process the data from each sheet in the ODS file
def process_dataframe(df, lowercase_columns=True, drop_na=False):
    if lowercase_columns:
        df.columns = [convert_to_lowercase(col) for col in df.columns]
    df = df.map(convert_to_lowercase)
    if 'query' in df.columns:
        df['query'] = df['query'].apply(format_logical_expression)
    if drop_na:
        df = df.dropna()
    return df

# Helper function to format logical expressions in the search strings' query column
def format_logical_expression(expression):
    if not isinstance(expression, str):
        return expression
    expression = re.sub(r"(?<![a-zA-Z])and(?![a-zA-Z])", " and ", expression)
    expression = re.sub(r"(?<![a-zA-Z])or(?![a-zA-Z])", " or ", expression)
    expression = re.sub(r"(?<![a-zA-Z])not(?![a-zA-Z])", " not ", expression)
    expression = re.sub(r"\s+", " ", expression).strip()
    return expression

# Helper function to convert strings to lowercase
def convert_to_lowercase(l):
    if isinstance(l, str):
        return l.lower().strip()
    return l

# ORGANIZE KEYWORDS IN DICTIONARY
def organize_keywords(df):
    cols = df.columns
    key_dict = {}
    for col in cols:
        values = [str(value).strip() for value in df[col].dropna() if str(value).strip()]
        key_dict[col.lower()] = values
    return key_dict

# CLEAN TEXT DATA
def clean_text(df, content):
    df[content] = df[content].apply(lambda text: re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE))
    df[content] = df[content].apply(lambda text: re.sub(r'\.{2,}', '.', text))
    df[content] = df[content].apply(lambda text: re.sub(r'\n(?=[a-z])', ' ', text))

# SPLIT TEXT INTO SENTENCES
def split_sentences(df):
    df["sentences"] = df["text"].apply(nltk.sent_tokenize)
    return df.explode("sentences")

# CLEAN AND DROP SENTENCES
def clean_rows(df):
    df["sentences"] = df["sentences"].str.lower()
    df = df[~df["sentences"].str.match(r'^\d+$')]                   
    df = df[~df["sentences"].str.match(r'^\d+[\W_]+$')]             
    df = df[~df["sentences"].str.match(r'^[\W_]+$')]               
    df.loc[:, "sentences"] = df["sentences"].str.replace('\n', '', regex=False)
    return df

## RUN CONTENT ANALYSIS
# CHECK PRESENCE OF LISTS FROM TAXONOMY IN SENTENCES
def check_groups(df, word_dict):
    for key, words in word_dict.items():
        df[key] = False
        patterns = [r"\b{}\b".format(re.escape(word).replace(r'\*', r'\w*')) for word in words]
        combined_pattern = '|'.join(patterns)
        df[key] = df['sentences'].str.contains(combined_pattern, regex=True)
    return df

# CHECK PRESENCE OF CO-OCCURRENCES IN SENTENCES BASED ON check_groups
def initiate_co_occurrences(df, set_co_occurrences, word_dict):
    set_co_occurrences['name of co-occurrence'] = set_co_occurrences['name of co-occurrence'].astype(object)
    compiled_patterns = {}
    for key, words in word_dict.items():
        patterns = [r"\b%s\b" % word.replace(".", r"\.").replace("*", "\w*") for word in words]
        compiled_patterns[key] = re.compile('|'.join(patterns))
    for index, row in set_co_occurrences.iterrows():
        if pd.isnull(row['name of co-occurrence']):
            set_co_occurrences.at[index, 'name of co-occurrence'] = ''
        key1 = row['first list']
        key2 = row['second list']
        distance = row['distance between lists']
        name = row['name of co-occurrence']
        df[name] = df['sentences'].apply(lambda sentence: find_co_occurrences(sentence, compiled_patterns[key1], compiled_patterns[key2], distance))
    return df

# Helper function to find co-occurrences
def find_co_occurrences(sentence, pattern1, pattern2, distance):
    occurrences1 = [m.start() for m in pattern1.finditer(sentence)]
    occurrences2 = [m.start() for m in pattern2.finditer(sentence)]
    co_occurrences = []
    for occ1 in occurrences1:
        for occ2 in occurrences2:
            start = min(occ1, occ2)
            end = max(occ1, occ2)
            num_words = len(sentence[start:end].split())
            co_occurrences.append(num_words)
    return min(co_occurrences) <= distance if co_occurrences else False

# CHECK DOCUMENT-LEVEL CONDITIONALS
def initiate_document_conditionals(df, set_doc_conditionals):
    for _, row in set_doc_conditionals.iterrows():
        name = row['name of document-level conditional']
        conditional = row['list']
        # Use groupby and transform to create a boolean series directly
        df[name] = df.groupby('document')[conditional].transform(lambda x: x.any())
    return df

# SENTIMENT ANALYSIS BASED ON VADAR
def vadar_sentiment_analysis(text_series):
    # Apply the sentiment analysis to the entire series
    sentiment_scores = text_series.apply(lambda text: sent_i.polarity_scores(text)['compound'])
    return pd.DataFrame({
        'vadar_compound': sentiment_scores,
        'sentiment_positive': sentiment_scores > 0,
        'sentiment_neutral': sentiment_scores == 0,
        'sentiment_neutralpositive': sentiment_scores >= 0,
        'sentiment_negative': sentiment_scores < 0,
        'sentiment_neutralnegative': sentiment_scores <= 0
    })

# RUN QUERIES LISTED IN ASSESSMENT_FRAMEWORK
def run_queries(df, set_search_strings):
    results = df.copy()
    for _, row in set_search_strings.iterrows():
        variable, query = row['query name'], row['query']
        # Convert AND, OR, NOT operations to their Python equivalents
        query = query.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        # Evaluate the query for each row in df
        results[variable] = df.apply(lambda x: evaluate_query(x, query), axis=1)
    return results

# Helper function to evaluate the query
def evaluate_query(row, query):
    for column in row.index:
        query = query.replace(f'"{column}"', str(row[column]))
    try:
        return eval(query)
    except Exception as e:
        print(f"Error evaluating query: {query} - {e}")
        return False
    
# AGGREGATE VARIABLES LISTED IN ASSESSMENT_FRAMEWORK
def define_variables(results, variables):
    for _, row in variables.iterrows():
        condition = row['aggregation query']
        child_vars = condition.split(',')
        child_vars = [var.strip(" ()\"'") for var in child_vars]
        variable_number = str(row['variable number'])
        if '.' not in variable_number:
            variable_number += '.'
        parent_name = f"{variable_number} {row['variable']}"
        results[parent_name] = results[child_vars].any(axis=1)
    return results


## EXPORT RESULTS
def get_v_list(df):
    return [col for col in df.columns if bool(re.match(r'^[0-9]+\.[0-9]*', str(col)))]

def results_document(df, v_list):
    codebook = df.groupby('document')[v_list].sum().astype(int).reset_index()
    return codebook

def results_document_clipped(df, v_list):
    code_bool = df.groupby('document')[v_list].sum().clip(upper=1).reset_index()
    return code_bool

def results_document_percentage_of_sentences(df, v_list):
    grouped = df.groupby('document')
    codebook_count_sent = grouped['sentences'].count().reset_index()
    code_sent_percent = grouped[v_list].sum().reset_index()
    code_sent_percent = code_sent_percent.merge(codebook_count_sent, on='document', how='outer')
    code_sent_percent[v_list] = code_sent_percent[v_list].div(code_sent_percent['sentences'], axis=0).multiply(100)
    code_sent_percent.drop('sentences', axis=1, inplace=True)
    return code_sent_percent

def results_document_sentiment(df, v_list):
    codebook_sentiment = df[['document']].drop_duplicates().set_index('document')
    for var in v_list:
        sentiment = df.loc[df[var], ['document', 'vadar_compound']].groupby('document')['vadar_compound'].mean()
        codebook_sentiment[var] = codebook_sentiment.index.map(sentiment)
    return codebook_sentiment.reset_index()

### SQL FUNCTIONS MANAGEMENT
def load_data_from_sql(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return df

def save_data_to_sql(df, db_path, table_name):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def load_data_from_sql(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    conn.close()
    return df


## CHECK FOR USER-MADE ERRORS IN ASSESSEMENT FRAMEWORK
def preliminary_checks(set_search_strings, set_doc_conditionals, set_co_occurrences, taxonomy):
    if not os.path.exists(ASSESSMENT_FRAMEWORK_DIRECTORY):
        print(f"Error: The Assessment framework file '{ASSESSMENT_FRAMEWORK_DIRECTORY}' not found.")
        return False
    return True

# Check the search strings for errors
def check_search_strings(search_strings, taxonomy, doc_conditionals, co_occurrences,variables):
    required_columns = ['query name', 'query']
    if not all(column in search_strings.columns for column in required_columns):
        print(f"Error: The search_strings tab does not have the required columns. Keep the column headers included in the template.")
        return False
    valid_terms = set(taxonomy.columns) | set(doc_conditionals['name of document-level conditional']) | set(co_occurrences['name of co-occurrence'])
    if search_strings['query name'].duplicated().any():
        print("Error: The 'variable' column in the search_strings tab contains duplicate entries.")
        return False
    for query in search_strings['query']:
        if query.count("(") != query.count(")"):
            print(f"Error: Mismatched parentheses in '{query}'.")
            return False
        strings_in_quotes = re.findall(r'"([^"]*)"', query)
        for string in strings_in_quotes:
            if string not in valid_terms and string not in ['sentiment_positive', 'sentiment_neutral', 'sentiment_neutralpositive', 'sentiment_negative', 'sentiment_neutralnegative']:
                print(f"Error: The string '{string}' inside quotation marks is not found in the provided taxonomy, document-level conditionals, or co-occurrences.")
                return False
        query_simplified = re.sub(r'"[^"]*"', '', query)
        query_simplified = re.sub(r'\b(and|or|not)\b', '', query_simplified, flags=re.IGNORECASE)
        query_simplified = query_simplified.replace("(", "").replace(")", "")
        if query_simplified.strip() != "":
            print(f"Error: Query contains invalid or improperly quoted strings: {query_simplified.strip()}")
            return False
        if '""' in query or re.search(r'"\s+"', query):
            print("Error: Found strings not properly enclosed in quotation marks or missing quotation marks.")
            return False
    required_co_occurrence_columns = ['name of co-occurrence', 'first list', 'distance between lists', 'second list']
    if not all(column in co_occurrences.columns for column in required_co_occurrence_columns):
        print("Error: co_occurrences tab does not have the required columns.")
        return False
    valid_terms_for_co_occurrences = set(taxonomy.columns)
    for index, row in co_occurrences.iterrows():
        if row['first list'] not in valid_terms_for_co_occurrences or row['second list'] not in valid_terms_for_co_occurrences:
            print(f"Error: Check if these strings in co_occurrences '{row['first list']}' or '{row['second list']}' are present in the taxonomy as column headers.")
            return False
    required_doc_conditional_columns = ['name of document-level conditional', 'list']
    if not all(column in doc_conditionals.columns for column in required_doc_conditional_columns):
        print("Error: doc_conditionals tab does not have the required columns.")
        return False
    for index, row in doc_conditionals.iterrows():
        if row['list'] not in valid_terms_for_co_occurrences:
            print(f"Error: The string '{row['list']}' in doc_conditionals is not present in the taxonomy as column headers.")
            return False
    all_variables_in_search_strings = set(search_strings['query name'])
    child_variables = set()
    for query in variables['aggregation query']:
        matches = re.findall(r'"(.*?)"', query)
        child_variables.update(matches)
    missing_variables = child_variables - all_variables_in_search_strings
    if missing_variables:
        print(f"Error: The following queries are not present in the search_strings 'query name' column: {missing_variables}. Note that query names cannot be numerical")
        return False
    return True

## ESTIMATE RUN TIME
def estimate_run_time(sample_df, full_run=False): 
    variables, set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords = create_dataframes(ASSESSMENT_FRAMEWORK_DIRECTORY)
    key_dict = organize_keywords(set_keywords)
    start_time = time.time()
    sample_df = check_groups(sample_df, key_dict)
    sample_df = initiate_co_occurrences(sample_df, set_co_occurrences, key_dict)
    initiate_document_conditionals(sample_df, set_doc_conditionals)
    if SENTIMENT_ANALYSIS:
        sentiment_df = vadar_sentiment_analysis(sample_df['sentences'])
        sample_df = pd.concat([sample_df, sentiment_df], axis=1)
    sample_results = run_queries(sample_df, set_search_strings)
    sample_results = define_variables(sample_results, variables)
    end_time = time.time()
    return end_time - start_time

def run_analysis_from_sentences(df, ods_file, compound=None):
    """Replay the analysis part of the v2.1 ``main()`` on an existing sentence table.

    ``df`` needs ``document`` and ``sentences`` columns (as stored in v2.1's
    results_sentences.tsv). If ``compound`` is given, it replaces VADER scoring so the
    result does not depend on the installed NLTK version. Returns the sentence-level
    frame and the four document-level frames, as v2.1 would export them.
    """
    global ASSESSMENT_FRAMEWORK_DIRECTORY
    ASSESSMENT_FRAMEWORK_DIRECTORY = ods_file
    variables, set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords = create_dataframes(ods_file)
    key_dict = organize_keywords(set_keywords)
    assert check_search_strings(set_search_strings, set_keywords, set_doc_conditionals, set_co_occurrences, variables)
    df = df[["document", "sentences"]].copy()
    df = check_groups(df, key_dict)
    df = initiate_co_occurrences(df, set_co_occurrences, key_dict)
    df = initiate_document_conditionals(df, set_doc_conditionals)
    if SENTIMENT_ANALYSIS:
        if compound is None:
            sentiment_df = vadar_sentiment_analysis(df["sentences"])
        else:
            s = pd.Series(list(compound), index=df.index)
            sentiment_df = pd.DataFrame({
                "vadar_compound": s,
                "sentiment_positive": s > 0,
                "sentiment_neutral": s == 0,
                "sentiment_neutralpositive": s >= 0,
                "sentiment_negative": s < 0,
                "sentiment_neutralnegative": s <= 0,
            })
        df = pd.concat([df, sentiment_df], axis=1)
    df = run_queries(df, set_search_strings)
    df = define_variables(df, variables)
    v_list = get_v_list(df)
    out = {
        "results_document": results_document(df, v_list),
        "results_document_clipped": results_document_clipped(df, v_list),
        "results_document_percentage_of_sentences": results_document_percentage_of_sentences(df, v_list),
    }
    if SENTIMENT_ANALYSIS:
        out["results_document_sentiment"] = results_document_sentiment(df, v_list)
    out["results_sentences"] = df
    return out
