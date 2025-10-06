import re
import requests
import csv 
import json
# Define the regex pattern
pattern = r'^\s*(vn)\.train\s*\(\s*question'
sec_pattern = r'question="(.*?)",\s*sql="(.*?)"'

def get_question_sql_pairs(python_script_path='./dev/poc_vanna.py')->list[dict]:
    question_sql_pairs_list = []
    with open(python_script_path, 'r') as f:
        lines = f.readlines()
    matched_lines = [line.replace("vn.train","") for line in lines if re.match(pattern, line)]
    print(len(matched_lines))
    for i in matched_lines:
        match = re.search(sec_pattern, i)
        if match:
            question = match.group(1)
            sql = match.group(2)
            question_sql_pairs_list.append({"question":question , "sql":sql})

    return question_sql_pairs_list


token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImUyNmYyZDkxLWQ2NWEtNDRjYS1hZWQxLThmODE5NmUwNDk0NyJ9.uAXT9kSomM3GzoVTE1bXcuWBVrriS7bGBrT0y4iQaEI"
def chat_with_model(token,users_question,model='vanna-sql-query'):
    url = 'http://3.29.105.219:3000/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model,
      "mode":"test_eval",
      "messages": [
        {
          "role": "user",
          "content": f"{users_question}"
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    response = response.json().get("choices")

    if len(response) != 1:
        raise VauleError
    response_message = (response[0].get("message").get("content")).replace("\n"," ")
    return response_message


number_of_statements_to_test = 1
def run_test(question_sql_pairs)-> list[dict]:
    index = 0
    for question_sql_pair in question_sql_pairs:
        if index < number_of_statements_to_test:
            print(index)
            # get the question
            question = question_sql_pair.get("question")
            model_answer_to_question = chat_with_model(token,question,model="vanna-sql-query-hr")
            # print("-----"*10)
            # print(model_answer_to_question)
            # # structure the answer as a dictionary
            # {"answer":model_answer_to_question}
            # update the old dictionary with the new key-value pair

            match = re.search(r"Here's the SQL query I generated for your question:\s+```sql\s+(.*?)```", model_answer_to_question, re.DOTALL)
            if match:
                extracted_sql = match.group(1).strip()
            else:
                extracted_sql = str(model_answer_to_question)
            question_sql_pair["models_answer"] = extracted_sql
            question_sql_pairs[index] = question_sql_pair
            index += 1
        else:
            break
    return question_sql_pairs  


def normalizer(sql):
    # remove the aliases

    # remove the schemas
    # remote limits
    normalized_sql = ' '.join(sql.strip().lower().split()).replace(";","") 
    limit_pattern = r'\s+LIMIT\s+\d+\s*'
    normalized_sql = re.sub(pattern, ' ', normalized_sql, flags=re.IGNORECASE)
    # remove line breaks to that the whole statment is on a single line.
    normalized_sql = normalized_sql.replace(r"\n"," ")
    return 

def compare(question_sql_pairs):
    index = 0
    # print("$$$$$$$$")    
    # print(question_sql_pairs)

    for question_sql_pair in question_sql_pairs:
        if index < number_of_statements_to_test:

            sql1 = question_sql_pair.get("sql")
            sql2 = question_sql_pair.get("models_answer")
            sql1_normalized = normalizer(sql1)
            sql2_normalized = normalizer(sql2)
            question_sql_pair["is_exact"] = sql1_normalized == sql2_normalized
            question_sql_pairs[index] = question_sql_pair
            index += 1
        else:
            break
    return question_sql_pairs 
# print("=>"*15+" question_sql_pairs")
question_sql_pairs = get_question_sql_pairs(python_script_path="./dev/poc_vanna_hr.py")
# print(json.dumps(question_sql_pairs,indent = 3))
# print("=>"*15+" question_sql_answer")
question_sql_answer = run_test(question_sql_pairs)
# print(json.dumps(question_sql_answer,indent=3))
# print("=>"*15 + " question_sql_answer_is_exact")
question_sql_answer_is_exact = compare(question_sql_answer)
# print(json.dumps(question_sql_answer_is_exact,indent=3))

# Define the output CSV file
filename = "./tests/models/hr/hr_models_accuracy.csv"

# Write to the CSV file
with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
    # Create a CSV writer using the fieldnames from the first dict
    fieldnames = question_sql_answer_is_exact[0].keys()
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Write column headers
    writer.writerows(question_sql_answer_is_exact)  # Write data rows


