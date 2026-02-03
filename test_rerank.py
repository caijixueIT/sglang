import requests
import json
import math

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

def make_input(instruction, queries, docs):
    query_text = query_template.format(prefix=prefix, instruction=instruction, query=query)
    doc_text = document_template.format(doc=doc, suffix=suffix)
    input_content = query_text + doc_text
    return input_content


def get_yes_score_from_logits(logits_list):
    """
    Get the 'yes' score from logits list (score is already computed by server).
    
    logits_list: [{"token": "yes", "token_id": 9693, "logit": xxx, "score": xxx}, 
                  {"token": "no", "token_id": 2201, "logit": xxx, "score": xxx}]
    """
    if logits_list is None:
        return None
    
    for item in logits_list:
        if item["token"] == "yes":
            return item["score"]
    
    return None



instruction = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "I want to eat an apple",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


for query, doc in zip(queries, documents):
    input_content = make_input(instruction, queries, doc)
    # Request with return_logits=True to get raw logits
    data = {
        "model": "/pfs-verdent/libaoguo/models/Qwen_Qwen3-Reranker-8B",
        "prompt": input_content,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "logprobs": 2,
        "return_logits": True  # Enable raw logits return
    }

    response = requests.post(
        url="http://45.78.195.171:8998/v1/completions",
        headers={"Content-Type": "application/json"},
        json=data
    )
    result = response.json()
    
    # Extract logits from response
    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        logits = choice.get("logits", None)
        
        if logits is not None:
            yes_score = get_yes_score_from_logits(logits)
            print(f"Relevance Score (yes): {yes_score:.4f}")
            print(f"Logits with scores: {json.dumps(logits, indent=2)}")
        else:
            print("No logits returned. Make sure return_logits=True is set.")
    
    # Also print generated text (should only be 'yes' or 'no' now due to mask)
    print(f"Generated text: {result.get('choices', [{}])[0].get('text', 'N/A')}")
    print(f"Full response: {json.dumps(result, indent=2)}")


