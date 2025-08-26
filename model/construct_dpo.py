#!/usr/bin/env python3
import argparse
import json
import sqlite3
import re
import openai
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def execute_sql_on_database(sql_query: str, db_path: str) -> List[Tuple]:

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"SQL execution error: {e}")
        return []

def generate_sql_with_knowledge(question: str, schema: str, knowledge: str, client) -> str:

    prompt = f"""Given the following database schema and question, generate a SQL query. Use the provided knowledge to help generate accurate SQL.

    Database Schema:
    {schema}

    Question: {question}

    Knowledge: {knowledge}

    Generate SQL:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate accurate SQL queries based on the given schema, question, and knowledge."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return ""

def check_sql_contribution_with_gpt(question: str, knowledge: str, ground_truth_sql: str, client) -> bool:

    few_shot_examples = [

        {
            "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
            "SQL": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"
        },
        {
            "evidence": "most popular movie refers to MAX(movie_popularity); when it was released refers to movie_release_year; director for the movie refers to director_name;",
            "SQL": "SELECT movie_title, movie_release_year, director_name FROM movies ORDER BY movie_popularity DESC LIMIT 1"
        },

        {
            "evidence": "longest movie title refers to MAX(LENGTH(movie_title)); when it was released refers to movie_release_year;",
            "SQL": "SELECT movie_title, movie_release_year FROM movies ORDER BY LENGTH(movie_popularity) DESC LIMIT 1"
        },
        {
            "evidence": "movie with the most rating refers to MAX(SUM(rating_score));",
            "SQL": "SELECT movie_title FROM movies GROUP BY movie_title ORDER BY COUNT(movie_title) DESC LIMIT 1"
        }
    ]

    few_shot_text = ""
    for i, example in enumerate(few_shot_examples):
        few_shot_text += f"Example {i+1}:\n"
        few_shot_text += f"Evidence: {example['evidence']}\n"
        few_shot_text += f"SQL: {example['SQL']}\n"
        if i < 2:  
            few_shot_text += f"Contribution: [CON]True[CON]\n\n"
        else:  
            few_shot_text += f"Contribution: [CON]False[CON]\n\n"

    prompt = f"""{few_shot_text}Now evaluate this case:

    Evidence: {knowledge}
    SQL: {ground_truth_sql}

    Does the evidence contribute to the SQL generation? Answer with [CON]True[CON] or [CON]False[CON]:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Determine if the evidence contributes to SQL generation. Answer only with [CON]True[CON] or [CON]False[CON]."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip()
        
        import re
        match = re.search(r'\[CON\](True|False)\[CON\]', result)
        if match:
            return match.group(1).lower() == "true"
        else:
            print(f"Could not extract True/False from response: {result}")
            return False
            
    except Exception as e:
        print(f"Error checking SQL contribution: {e}")
        return False

def find_matching_knowledge(question: str, schema: str, generated_knowledge_map: Dict[str, str], index: int) -> str:

    knowledge_list = list(generated_knowledge_map.values())
    if index < len(knowledge_list):
        return knowledge_list[index]
    return ""

def generate_knowledge_with_sft_model(question: str, schema: str, sft_model_path: str, generated_knowledge_file: str, index: int) -> str:
    try:
        generated_knowledge_map = {}
        with open(generated_knowledge_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)

                    generated_knowledge_map[data.get('label', '')] = data.get('predict', '')
        
        return find_matching_knowledge(question, schema, generated_knowledge_map, index)
        
    except Exception as e:
        print(f"Error loading generated knowledge: {e}")
        return ""

def create_dpo_pairs(sft_data: List[Dict], db_root_path: str, sft_model_path: str, openai_api_key: str, generated_knowledge_file: str) -> List[Dict]:

    dpo_pairs = []

    client = openai.OpenAI(api_key=openai_api_key)
    
    generated_knowledge_list = []
    try:
        with open(generated_knowledge_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    generated_knowledge_list.append(data.get('predict', ''))
    except Exception as e:
        print(f"Error loading generated knowledge: {e}")
        return dpo_pairs
    
    for i, item in enumerate(tqdm(sft_data, desc="Creating DPO pairs")):
        question = item.get('instruction', '')
        schema = item.get('input', '')
        gold_knowledge = item.get('output', '')
        db_id = item.get('db_id', '')
        gold_sql = item.get('sql', '')
        
        if i < len(generated_knowledge_list):
            generated_knowledge = generated_knowledge_list[i]
        else:
            continue
        
        if not generated_knowledge:
            continue
        
        db_path = f"{db_root_path}/{db_id}/{db_id}.sqlite"
        

        gold_sql_generated = generate_sql_with_knowledge(question, schema, gold_knowledge, client)
        gold_sql_result = execute_sql_on_database(gold_sql_generated, db_path)
        
        generated_sql = generate_sql_with_knowledge(question, schema, generated_knowledge, client)
        generated_sql_result = execute_sql_on_database(generated_sql, db_path)
        
        db_feedback = (set(gold_sql_result) == set(generated_sql_result))
        
        gold_contribution = check_sql_contribution_with_gpt(question, gold_knowledge, gold_sql, client)
        generated_contribution = check_sql_contribution_with_gpt(question, generated_knowledge, gold_sql, client)
        
        if not db_feedback:

            dpo_pair = {
                "instruction": question,
                "input": schema,
                "output": [gold_knowledge, generated_knowledge]
            }
            dpo_pairs.append(dpo_pair)
        
        if gold_contribution and not generated_contribution:

            dpo_pair = {
                "instruction": question,
                "input": schema,
                "output": [gold_knowledge, generated_knowledge]
            }
            dpo_pairs.append(dpo_pair)
    
    return dpo_pairs

def load_sft_data(sft_file: str) -> List[Dict]:

    with open(sft_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dpo_data(dpo_pairs: List[Dict], output_file: str):

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Construct DPO data based on database execution and SQL contribution feedback')
    parser.add_argument('--sft_file', type=str, required=True, help='Path to SFT data file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output DPO data file')
    parser.add_argument('--db_root_path', type=str, required=True, help='Root path to databases')
    parser.add_argument('--sft_model_path', type=str, required=True, help='Path to SFT model')
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--generated_knowledge_file', type=str, required=True, help='Path to generated knowledge file from SFT model')
    
    args = parser.parse_args()
    
    print("Loading SFT data...")
    sft_data = load_sft_data(args.sft_file)
    
    print("Creating DPO pairs...")
    dpo_pairs = create_dpo_pairs(sft_data, args.db_root_path, args.sft_model_path, args.openai_api_key, args.generated_knowledge_file)
    
    print(f"Saving DPO data to {args.output_file}...")
    save_dpo_data(dpo_pairs, args.output_file)
    
    print(f"Successfully created {len(dpo_pairs)} DPO pairs")

if __name__ == "__main__":
    main()
