# For additional information or assistance with running this code, please reach out to contact[at]scartozzi[dot]eu.
# This code is released under GNU General Public License v3.0. Feel free to use it as you wish.

import os
import re
import time
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent_i = SentimentIntensityAnalyzer()

# GLOBAL VARIABLES
# specify the directory where the text files are stored:
TEXT_FILES_DIRECTORY = "./Analysis/Text_files"
# specify the directory where the assessment framework is stored:
ASSESSMENT_FRAMEWORK_DIRECTORY = "./Analysis/Input/Assessment_framework.ods"
# specify the directory where the results will be saved:
OUTPUT_DIRECTORY = "./Analysis/Output"
# choose the following settings:
TEXT_CLEANING = True
SENTIMENT_ANALYSIS = True
# choose what to export:
EXPORT_SENTENCE_LEVEL_DATA = True
EXPORT_DOCUMENT_DATA = True
# estimate the time needed to run the code:
PRINT_RUN_TIME = True

# IMPORT TEXT FILES
def txt_to_dataframe():
    file_path = []
    for file in os.listdir(TEXT_FILES_DIRECTORY):
        file_path.append(os.path.join(TEXT_FILES_DIRECTORY, file))
    file_name = re.compile('\\\\(.*)\.txt')
    data = {}
    for file in file_path:
        key = file_name.search(file)
        with open(file, "r", encoding='Latin-1') as read_file:
            if key is not None:
                data[key[1]] = [read_file.read()]
    df = pd.DataFrame(data).T.reset_index().rename(columns = {'index':'document', 0:'text'})
    codebook = df[['document']].copy()
    codebook_sentiment = codebook
    df.head(3)
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
    df = df.applymap(convert_to_lowercase)
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

# CLEAN TEXT
def clean_text(df, content):
    df[content] = df[content].apply(lambda text: re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE))
    df[content] = df[content].apply(lambda text: re.sub(r'\.{2,}', '.', text))
    df[content] = df[content].apply(lambda text: re.sub(r'\n(?=[a-z])', ' ', text))

# SPLIT TEXT INTO SENTENCES
def split_sentences(df): 
    df["sentences"] = df["text"].apply(nltk.sent_tokenize)
    df["sentences"] = df["sentences"].apply(lambda sentences: [sentence.lower() for sentence in sentences])
    return df.explode("sentences")

# LABEL SENTENCES BASED ON KEYWORDS IN TAXONOMY
def check_groups(row, word_dict, *args): 
    for key, words in word_dict.items():
        for word in words:
            pattern = r"\b%s\b" % word.replace(".", r"\.").replace("*", "\w*")
            matches = re.findall(pattern, row['sentences'])
            if len(matches) > 0:
                row[key] = True
                break
        else:
            row[key] = False
    return row

# LABEL SENTENCES BASED ON CO-OCCURRENCES
def initiate_co_occurrences(df, set_co_occurrences, word_dict):
    # Convert the 'name' column to the 'object' data type
    set_co_occurrences['name of co-occurrence'] = set_co_occurrences['name of co-occurrence'].astype(object)
    for index, row in set_co_occurrences.iterrows():
        # Replace NaN values in the 'name' column with an empty string
        if pd.isnull(row['name of co-occurrence']):
            row['name of co-occurrence'] = ''
        key1 = row['first list']
        key2 = row['second list']
        distance = row['distance between lists']
        name = row['name of co-occurrence']
        df = df.apply(find_co_occurrences, key1=key1, key2=key2, distance=distance, name=name, word_dict=word_dict, axis=1)
    return df

# helper function to find co-occurrences
def find_co_occurrences(row, key1, key2, word_dict, distance, name):
    words1 = word_dict[key1]
    words2 = word_dict[key2]
    words1 = [str(word) for word in words1]
    words2 = [str(word) for word in words2]
    patterns1 = [r"\b%s\b" % word.replace(".", r"\.").replace("*", "\w*") for word in words1]
    patterns2 = [r"\b%s\b" % word.replace(".", r"\.").replace("*", "\w*") for word in words2]
    occurrences1 = []
    occurrences2 = []
    for pattern in patterns1:
        occurrences1 += [m.start() for m in re.finditer(pattern, row['sentences'])]
    for pattern in patterns2:
        occurrences2 += [m.start() for m in re.finditer(pattern, row['sentences'])]
    co_occurrences = []
    for occ1 in occurrences1:
        for occ2 in occurrences2:
            start = min(occ1, occ2)
            end = max(occ1, occ2)
            num_words = len(row['sentences'][start:end].split())
            co_occurrences.append(num_words)
    if co_occurrences:
        if min(co_occurrences) <= distance:
            row[name] = True 
        else:
            row[name] = False
    else:
        row[name] = False
    return row

# INITIATE DOCUMENT CONDITIONALS    
def initiate_document_conditionals(df, set_doc_conditionals):
    set_doc_conditionals.apply(lambda row: find_document_conditionals(df, row['name of document-level conditional'], row['list']), axis=1)

# Helper function to find document conditionals
def find_document_conditionals(df, name, conditional): 
    df[name] = df['document'].map(df.groupby('document').apply(lambda x: x[conditional].eq(1).any()))

# SENTIMENT ANALYSIS
def vadar_sentiment_analysis(text):
        return sent_i.polarity_scores(text)['compound']

# RUN QUERIES
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

# AGGREGATE VARIABLES
def define_variables(results, variables):
    for _, row in variables.iterrows():
        condition, threshold_str = row['aggregation query'].split('>')
        threshold = int(threshold_str.strip())
        child_vars = []
        for var in condition.split('+'):
            var_name = var.strip(" ()\"'")
            if var_name in results.columns:
                child_vars.append(var_name)
        variable_number = str(row['variable number'])
        if '.' not in variable_number:
            variable_number += '.'
        parent_name = f"{variable_number} {row['variable']}"
        count_true = sum([results[var] for var in child_vars])
        results[parent_name] = count_true > threshold
    return results

# EXPORT RESULTS
def results_document(df):
    v_list = [col for col in df.columns if bool(re.match('^[0-9]+\.[0-9]*', str(col)))]
    codebook = df.groupby(['document'])[v_list].sum().astype(int).reset_index()
    return codebook

def results_document_clipped(df):
    v_list = [col for col in df.columns if bool(re.match('^[0-9]+\.[0-9]*', str(col)))]
    code_bool = df.groupby(['document'])[v_list].sum().reset_index()
    code_bool[v_list] = code_bool[v_list].clip(upper=1)
    return code_bool

def results_document_percentage_of_sentences(df):
    v_list = [col for col in df.columns if bool(re.match('^[0-9]+\.[0-9]*', str(col)))]
    codebook_count_sent = df.groupby(['document'])['sentences'].count().reset_index()
    code_sent_percent = df.groupby(['document'])[v_list].sum().reset_index()
    code_sent_percent = code_sent_percent.merge(codebook_count_sent, on='document', how='outer')
    code_sent_percent[v_list] = code_sent_percent[v_list].div(code_sent_percent['sentences'], axis=0).multiply(100)
    code_sent_percent.drop('sentences', axis=1, inplace=True)
    return code_sent_percent

def results_document_sentiment(df):
    v_list = [col for col in df.columns if bool(re.match('^[0-9]+\.[0-9]*', str(col)))]
    codebook_sentiment = df[['document']].drop_duplicates()
    for var in v_list:
        sentiment = df[df[var]].groupby('document', as_index=False)['vadar_compound'].mean()
        codebook_sentiment[var] = codebook_sentiment['document'].map(sentiment.set_index('document')['vadar_compound'])
    return codebook_sentiment.groupby('document').mean().reset_index()


# CHECK FOR USER-MADE ERRORS IN ASSESSEMENT FRAMEWORK
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
            if string not in valid_terms:
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

def estimate_run_time(df, full_run=False):
    start_time = time.time()
    sample_df = df.sample(n=50)
    variables, set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords = create_dataframes(ASSESSMENT_FRAMEWORK_DIRECTORY)
    key_dict = organize_keywords(set_keywords)
    sample_df = sample_df.apply(check_groups, axis=1, args=(key_dict,))
    sample_df = initiate_co_occurrences(sample_df, set_co_occurrences, key_dict)
    initiate_document_conditionals(sample_df, set_doc_conditionals)
    if SENTIMENT_ANALYSIS:
        sample_df['vadar_compound'] = sample_df['sentences'].apply(vadar_sentiment_analysis) 
    sample_results = run_queries(sample_df, set_search_strings)
    sample_results = define_variables(sample_results, variables)
    end_time = time.time()
    return end_time - start_time

def main():  
    # Load, clean, and organize data
    df = txt_to_dataframe()
    variables, set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords = create_dataframes(ASSESSMENT_FRAMEWORK_DIRECTORY)
    # check for errors in the assessment framework
    if not preliminary_checks(set_search_strings, set_co_occurrences, set_doc_conditionals, set_keywords):
        print("Preliminary checks failed. Exiting...")
        return  
    if not check_search_strings(set_search_strings, set_keywords, set_doc_conditionals, set_co_occurrences, variables):
        print("Search strings check failed.")
        return
    # Clean text and split into sentences
    if TEXT_CLEANING:
        clean_text(df, 'text')
    df = split_sentences(df)
    df = df.drop('text', axis=1)
    if PRINT_RUN_TIME:
        total_sentences = len(df)
        if total_sentences < 50:
            print(f"The dataset has only {total_sentences} sentences, which is insufficient to accurately estimate runtime.")
        else:
            time_for_50 = estimate_run_time(df, full_run=False)
            estimated_total_time = (time_for_50 / 50) * total_sentences
            print(f"Estimated run time for {total_sentences} sentences: {estimated_total_time} seconds (excluding time to export results).")
    key_dict = organize_keywords(set_keywords)
    # Analyze text based on the taxonomy
    df = df.apply(check_groups, axis=1, args=(key_dict,))
    df = initiate_co_occurrences(df, set_co_occurrences, key_dict)
    initiate_document_conditionals(df, set_doc_conditionals)
    if SENTIMENT_ANALYSIS:
        df['vadar_compound'] = df['sentences'].apply(vadar_sentiment_analysis) 
    # Run queries and aggregate variables
    results = run_queries(df, set_search_strings)
    results = define_variables(results, variables)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    if EXPORT_DOCUMENT_DATA:
        codebook = results_document(results)
        codebook.to_csv(f'{OUTPUT_DIRECTORY}/results_document.tsv', sep='\t', index=False)
        code_bool = results_document_clipped(results)
        code_bool.to_csv(f'{OUTPUT_DIRECTORY}/results_document_clipped.tsv', sep='\t', index=False)    
        code_sent_percent = results_document_percentage_of_sentences(results)
        code_sent_percent.to_csv(f'{OUTPUT_DIRECTORY}/results_document_percentage_of_sentences.tsv', sep='\t', index=False)
        if SENTIMENT_ANALYSIS:
            codebook_sentiment = results_document_sentiment(results)
            codebook_sentiment.to_csv(f'{OUTPUT_DIRECTORY}/results_document_sentiment.tsv', sep='\t', index=False)
    if EXPORT_SENTENCE_LEVEL_DATA:
        results.to_csv(f'{OUTPUT_DIRECTORY}/results_sentences.tsv', sep='\t', index=False)
    return results

if __name__ == "__main__":
    main_results = main()