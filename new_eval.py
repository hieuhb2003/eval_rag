
import re
import json
import time
import logging
import traceback
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import backoff

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("direct_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Function to safely parse JSON with multiple attempts and cleaning
def safe_json_parse(json_str, max_attempts=3):
    """
    Safely parse JSON string with multiple cleaning attempts
    
    Args:
        json_str: JSON string to parse
        max_attempts: Maximum number of attempts to parse
    
    Returns:
        Parsed JSON object or None if all attempts fail
    """
    attempts = 0
    
    # Define cleaning steps to try in order
    cleaning_steps = [
        lambda s: s,  # Try original string first
        lambda s: s.replace("```json", "").replace("```", "").strip(),  # Remove code blocks
        lambda s: s.encode('utf-8').decode('unicode_escape'),  # Handle unicode
        lambda s: s.replace("\u2019", "'").replace("\u201c", "\"").replace("\u201d", "\"").replace("\u00e2\u0080\u0099", "'"),  # Fix quotes
        lambda s: re.sub(r',(\s*[\]}])', r'\1', s),  # Fix trailing commas
        lambda s: re.sub(r'\\([^\\"])', r'\1', s),  # Fix extra backslashes
        lambda s: s if s.strip().startswith("{") else "{" + s.strip() + "}",  # Add missing braces if needed
    ]
    
    # Try progressive combinations of cleaning steps
    for i in range(len(cleaning_steps)):
        for j in range(i, len(cleaning_steps)):
            if attempts >= max_attempts:
                return None
                
            try:
                # Apply sequence of cleaning functions
                cleaned = json_str
                for k in range(i, j+1):
                    cleaned = cleaning_steps[k](cleaned)
                
                # Try to parse the JSON
                result = json.loads(cleaned, strict=False)
                logger.info(f"JSON parsed successfully after {attempts+1} attempts")
                return result
            except json.JSONDecodeError:
                attempts += 1
                continue
    
    # If we get here, all attempts failed
    return None

# Backoff handler for API calls
@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    on_backoff=lambda details: logger.warning(f"API call failed. Retrying in {details['wait']:.1f} seconds...")
)
def make_api_call(client, sys_prompt, prompt):
    """Make API call with exponential backoff retry"""
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
    )

def extract_answers(result_data, file_path):
    """Extract answers from result data with robust handling of different formats."""
    try:
        if "result" in result_data and isinstance(result_data["result"], list):
            # Single result object with a "result" list
            return [item.get("answer", "") for item in result_data["result"]]
        elif isinstance(result_data, list):
            if all("result" in item for item in result_data):
                # List of objects with "result" key
                return [item.get("result", "") for item in result_data]
            elif all("answer" in item for item in result_data):
                # List of objects with "answer" key
                return [item.get("answer", "") for item in result_data]
            else:
                # Mixed format or other structure
                answers = []
                for item in result_data:
                    if "answer" in item:
                        answers.append(item["answer"])
                    elif "result" in item:
                        answers.append(item["result"])
                    else:
                        # If we can't find the expected fields, use the first string value we find
                        for k, v in item.items():
                            if isinstance(v, str) and len(v) > 10:
                                answers.append(v)
                                break
                        else:
                            answers.append("")  # Fallback to empty string if nothing found
                return answers
        else:
            logger.error(f"Completely unknown format in {file_path}")
            raise ValueError(f"Unknown format in {file_path}")
    except Exception as e:
        logger.error(f"Error extracting answers from {file_path}: {str(e)}")
        raise

def direct_eval(query_file, result1_file, result2_file, ground_truth_path, output_file, limit=None):
    """
    Perform direct evaluation comparing two sets of answers
    
    Args:
        query_file: Path to file containing queries
        result1_file: Path to file containing first set of answers
        result2_file: Path to file containing second set of answers
        ground_truth_path: Path to file containing ground truth data
        output_file: Path to save evaluation results
        limit: Optional limit on number of queries to evaluate
    """
    client = OpenAI(api_key=os.getenv('API_KEY'))

    try:
        with open(query_file, "r") as f:
            queries = json.load(f)
        
        if limit:
            queries = queries[:limit]
        logger.info(f"Loaded {len(queries)} queries")

        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        with open(result1_file, "r") as f:
            result1_data = json.load(f)
        
        with open(result2_file, "r") as f:
            result2_data = json.load(f)
        
        answers1 = extract_answers(result1_data, result1_file)
        answers2 = extract_answers(result2_data, result2_file)
        
        # Ensure we have same number of items
        min_length = min(len(queries), len(answers1), len(answers2))
        if min_length < len(queries):
            logger.warning(f"Limiting evaluation to {min_length} items due to mismatched data lengths")
            queries = queries[:min_length]
            answers1 = answers1[:min_length]
            answers2 = answers2[:min_length]
    
    except Exception as e:
        logger.error(f"Error loading data files: {str(e)}")
        raise

    all_results = []
    sys_prompt = """
    ---Role---
    You are an expert tasked with evaluating two answers to the same question based on four criteria: **Accuracy**, **Comprehensiveness**, **Diversity**, and **Empowerment**.
    You must produce valid JSON output.
    """

    for i, (query, answer1, answer2) in enumerate(tqdm(zip(queries, answers1, answers2), total=len(queries))):
        # Skip empty answers
        if not answer1 or not answer2:
            logger.warning(f"Skipping query {i+1} due to empty answer")
            all_results.append({
                "query_id": i + 1,
                "query": query,
                "error": "Empty answer"
            })
            continue
            
        context = "\n\n".join(ground_truth.get(query, ["No context available"]))
        prompt = f"""
        You will evaluate two answers to the same question based on four criteria: **Accuracy**, **Comprehensiveness**, **Diversity**, and **Empowerment**.

        - **Accuracy**: How factually correct is the answer based on the provided context and ground truth? Does it avoid hallucinations or factual errors?
        - **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
        - **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
        - **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

        For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these four categories, with Accuracy being the most important criterion.

        Here is the question:
        {query}

        Here is the context which supported to answer the question:
        {context}

        Here are the two answers:

        **Answer 1:**
        {answer1}

        **Answer 2:**
        {answer2}

        Evaluate both answers using the four criteria listed above and provide detailed explanations for each criterion.

        Output your evaluation in the following JSON format and NOTHING ELSE:

        {{
            "Accuracy": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here, specifically noting any factual errors or hallucinations in either answer]"
            }},
            "Comprehensiveness": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Winner": {{
                "Winner": "[Answer 1 or Answer 2]",
                "Explanation": "[Summarize why this answer is the overall winner based on the four criteria, with emphasis on accuracy]"
            }}
        }}
        """
        
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Make API call with backoff retry
                response = make_api_call(client, sys_prompt, prompt)
                content = response.choices[0].message.content

                # Try to parse the JSON response with multiple cleaning attempts
                eval_data = safe_json_parse(content)
                
                if eval_data:
                    # Validate required fields
                    required_keys = ["Accuracy", "Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
                    if all(key in eval_data for key in required_keys) and all("Winner" in eval_data[key] for key in required_keys):
                        result = {
                            "query_id": i + 1,
                            "query": query,
                            "evaluation": eval_data
                        }
                        
                        all_results.append(result)
                        winner = eval_data["Overall Winner"]["Winner"]
                        logger.info(f"Query {i+1}: Overall winner - {winner}")
                        success = True
                    else:
                        # Missing required fields
                        missing = [key for key in required_keys if key not in eval_data or "Winner" not in eval_data[key]]
                        logger.warning(f"Query {i+1}, Retry {retry_count+1}: Missing required fields in JSON: {missing}")
                        retry_count += 1
                else:
                    # JSON parsing failed completely
                    logger.warning(f"Query {i+1}, Retry {retry_count+1}: Could not parse JSON response")
                    retry_count += 1
                    
            except Exception as e:
                # Handle general exceptions
                logger.error(f"Error processing query {i+1}, Retry {retry_count+1}: {str(e)}")
                traceback.print_exc()
                retry_count += 1
                time.sleep(2)  # Wait before retrying
        
        # If all retries failed, add error entry
        if not success:
            all_results.append({
                "query_id": i + 1,
                "query": query,
                "raw_response": content if 'content' in locals() else "No response",
                "error": "Max retries exceeded"
            })
        
        # Save intermediate results every 10 queries
        if (i + 1) % 10 == 0:
            with open(output_file + ".partial", "w") as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"Saved intermediate results after {i+1} queries")

        # Add a small delay between queries to avoid rate limits
        time.sleep(1)

    # Save final results
    try:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"Evaluation results saved to {output_file}")
        
        # Generate summary
        summary_file = output_file.replace(".json", "_summary.json")
        summarize_direct_evaluation(all_results, summary_file)
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        # Save to backup file in case main save fails
        with open(output_file + ".backup", "w") as f:
            json.dump(all_results, f, indent=4)
    
    return all_results

def summarize_direct_evaluation(evaluation_results, summary_file):
    """
    Summarize evaluation results from direct evaluations
    
    Args:
        evaluation_results: List of evaluation results
        summary_file: File to save the summary
    """
    logger.info("Generating evaluation summary...")
    
    summary = {
        "total_comparisons": len(evaluation_results),
        "successful_evaluations": 0,
        "failed_evaluations": 0,
        "overall_winners": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
        "criteria_winners": {
            "Accuracy": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
            "Comprehensiveness": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
            "Diversity": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0},
            "Empowerment": {"Answer 1": 0, "Answer 2": 0, "Unknown": 0}
        }
    }
    
    for result in evaluation_results:
        if "evaluation" in result and isinstance(result["evaluation"], dict):
            summary["successful_evaluations"] += 1
            eval_data = result["evaluation"]

            if "Overall Winner" in eval_data and "Winner" in eval_data["Overall Winner"]:
                winner = eval_data["Overall Winner"]["Winner"]
                if winner in summary["overall_winners"]:
                    summary["overall_winners"][winner] += 1
                else:
                    summary["overall_winners"]["Unknown"] += 1
            else:
                summary["overall_winners"]["Unknown"] += 1
            
            # Count criteria winners
            for criterion in ["Accuracy", "Comprehensiveness", "Diversity", "Empowerment"]:
                if criterion in eval_data and "Winner" in eval_data[criterion]:
                    winner = eval_data[criterion]["Winner"]
                    if winner in summary["criteria_winners"][criterion]:
                        summary["criteria_winners"][criterion][winner] += 1
                    else:
                        summary["criteria_winners"][criterion]["Unknown"] += 1
                else:
                    summary["criteria_winners"][criterion]["Unknown"] += 1
        else:
            summary["failed_evaluations"] += 1
    
    # Calculate percentages
    valid_total = summary["overall_winners"]["Answer 1"] + summary["overall_winners"]["Answer 2"]
    if valid_total > 0:
        summary["percentages"] = {
            "Answer 1": round(summary["overall_winners"]["Answer 1"] / valid_total * 100, 2),
            "Answer 2": round(summary["overall_winners"]["Answer 2"] / valid_total * 100, 2)
        }
    
    # Add per-criterion percentages
    summary["criteria_percentages"] = {}
    for criterion in ["Accuracy", "Comprehensiveness", "Diversity", "Empowerment"]:
        valid_criterion_total = summary["criteria_winners"][criterion]["Answer 1"] + summary["criteria_winners"][criterion]["Answer 2"]
        if valid_criterion_total > 0:
            summary["criteria_percentages"][criterion] = {
                "Answer 1": round(summary["criteria_winners"][criterion]["Answer 1"] / valid_criterion_total * 100, 2),
                "Answer 2": round(summary["criteria_winners"][criterion]["Answer 2"] / valid_criterion_total * 100, 2)
            }
    
    # Save summary
    try:
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Evaluation summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving summary: {str(e)}")
    
    # Print quick results to console
    logger.info("\n--- EVALUATION SUMMARY ---")
    logger.info(f"Total comparisons: {summary['total_comparisons']}")
    logger.info(f"Successful evaluations: {summary['successful_evaluations']}")
    logger.info(f"Failed evaluations: {summary['failed_evaluations']}")
    
    if valid_total > 0:
        logger.info(f"Answer 1 wins: {summary['overall_winners']['Answer 1']} ({summary['percentages']['Answer 1']}%)")
        logger.info(f"Answer 2 wins: {summary['overall_winners']['Answer 2']} ({summary['percentages']['Answer 2']}%)")
    
    logger.info("--- Per-criterion Results ---")
    for criterion in ["Accuracy", "Comprehensiveness", "Diversity", "Empowerment"]:
        if criterion in summary.get("criteria_percentages", {}):
            logger.info(f"{criterion}: Answer 1: {summary['criteria_percentages'][criterion]['Answer 1']}%, Answer 2: {summary['criteria_percentages'][criterion]['Answer 2']}%")
    
    return summary

if __name__ == "__main__":
    query_file = "/home/hungpv/projects/TN/data/data_musique/query_en.json"
    result1_file = "/home/hungpv/projects/TN/LIGHTRAG/eval/results/answer_LightRAG Hybrid.json"  
    result2_file = "/home/hungpv/projects/TN/LIGHTRAG/eval/results/answer_naive.json"       
    ground_truth_path = "/home/hungpv/projects/TN/data/data_musique/text/truth_en.json"
    output_file = "/home/hungpv/projects/TN/LIGHTRAG/eval/results/direct_evaluation_results.json"
    
    # Run direct evaluation with limit (optional, remove or set to None for all queries)
    direct_eval(
        query_file=query_file,
        result1_file=result1_file,
        result2_file=result2_file,
        ground_truth_path=ground_truth_path,
        output_file=output_file,
        limit=10  # Set to a number or None
    )