import json
import os
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import openai
# from langdetect import detect

# Load environment variables from .env file
load_dotenv()

class LLMQueryProcessor:
    """
    A class that processes queries and their relevant chunks using an LLM to generate answers.
    Supports both OpenAI and Anthropic (Claude) models.
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 base_url=None):
        """
        Initialize the LLM Query Processor.
        
        Args:
            model_provider: The LLM provider ("openai" or "anthropic")
            model_name: The specific model to use (e.g., "gpt-4" or "claude-3-opus-20240229")
            api_key: API key for the model provider
        """
        
        # Set default models based on provider
        if model_name is None:
            self.model_name = "gpt-4-turbo"

        else:
            self.model_name = model_name
        
        if api_key is None:
            self.api_key = os.getenv("API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide it directly.")
        else:
            self.api_key = api_key

        if base_url:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=os.getenv("API_KEY"))


    def create_prompt(self, query: str, chunks: List[str], lang) -> str:
        """
        Create a prompt for the LLM based on the query and relevant chunks.
        
        Args:
            query: The user's query
            chunks: List of relevant text chunks/context
            
        Returns:
            A formatted prompt string
        """
        chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
        
        prompt = f"""You are a helpful assistant that provides accurate, comprehensive answers based on the provided context.

QUERY: {query}

RELEVANT CONTEXT:
{chunks_text}

Instructions:
1. Answer the query based ONLY on the information provided in the relevant context.
2. If the context doesn't contain sufficient information to answer the query, acknowledge this limitation.
3. Structure your answer in a clear, well-organized manner using markdown formatting.
4. Include section headers where appropriate.
5. Do not include any disclaimers or notes about the source of your information.

STRICTLY RULE: Respond in {lang}.

ANSWER:
"""
        return prompt

    def generate_answer(self, query: str, chunks: List[str], lang) -> str:
        """
        Generate an answer for a query using the LLM.
        
        Args:
            query: The user's query
            chunks: List of relevant text chunks/context
            
        Returns:
            The generated answer as a string
        """
        prompt = self.create_prompt(query, chunks, lang)
        
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides comprehensive answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content, prompt

        

    def process_queries(self, model_name, query_data: List[Dict[str, Any]], lang,top_k,output_file=None) -> List[Dict[str, Any]]:
        """
        Process a list of queries and their relevant chunks to generate answers.
        
        Args:
            query_data: A list of dictionaries containing queries and their relevant chunks
                Format: [{"query": "...", "chunks": ["chunk1", "chunk2", ...]}]
            
        Returns:
            A list of dictionaries with queries and their answers
                Format: [{"query": "...", "answer": "..."}]
        """
        results = []
        for i, item in enumerate(query_data):
            query = item["query"]
            chunks = item.get("chunks", [])
            if len(chunks) >= top_k:
                chunks=chunks[:top_k]
            
            try:
                answer,prompt = self.generate_answer(query, chunks, lang)
                result = {
                    "query": query,
                    "answer": answer,
                    "prompt": prompt,
                    "model": model_name,
                }
                results.append(result)
                print(f"Processed query: {query[:50]}...")
            except Exception as e:
                print(f"Error processing query '{query[:50]}...': {str(e)}")
                result = {
                    "query": query,
                    "answer": f"Error generating answer: {str(e)}",
                    "prompt": prompt,
                    "model": model_name
                }
                results.append(result)
            if output_file:
                self.save_results(results, output_file)
            else:
                self.save_results(results)
            if i == 10:
                break
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str = "answers.json") -> None:
        """
        Save the results to a JSON file.
        
        Args:
            results: List of dictionaries with queries and their answers
            output_file: Path to the output file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"Results saved to {output_file}")


# # Example usage
# if __name__ == "__main__":
#     # Sample input data
#     sample_input = [
#         {
#             "query": "How does Dickens portray the concept of redemption through the character arc of Ebenezer Scrooge?",
#             "chunks": [
#                 "In 'A Christmas Carol,' Ebenezer Scrooge begins as a miserly, cold-hearted businessman who despises Christmas and all forms of charity. His famous 'Bah! Humbug!' response to holiday greetings exemplifies his cynical attitude.",
#                 "Following visitations from the ghost of his former business partner Jacob Marley and the Spirits of Christmas Past, Present, and Yet to Come, Scrooge confronts his past decisions, current isolation, and potential legacy.",
#                 "The Ghost of Christmas Past reveals Scrooge's childhood loneliness, his lost love Belle, and happier times working for Fezziwig, exposing the roots of his bitterness.",
#                 "The Ghost of Christmas Present shows Scrooge the joy in his nephew Fred's home and the struggles of his clerk Bob Cratchit's family, particularly through Tiny Tim's illness.",
#                 "The Ghost of Christmas Yet to Come presents a grim future where Scrooge dies unmourned and Tiny Tim passes away, shocking Scrooge into profound regret.",
#                 "Scrooge awakens on Christmas morning transformed, embracing generosity and kindness. He sends a turkey to the Cratchit family, gives charitable donations, attends his nephew's dinner, and becomes 'as good a friend, as good a master, and as good a man, as the good old city knew.'"
#             ]
#         }
#     ]
    
#     # Initialize the processor
#     processor = LLMQueryProcessor(
#         model_provider="anthropic",
#         model_name="claude-3-opus-20240229"
#         # API key will be loaded from environment variable ANTHROPIC_API_KEY
#     )
#     result_path = "/home/hungpv/projects/train_embedding/nanographrag/results_musique/bge_naive_query_en_doc_vi.json"
#     mapping_query_path = "/home/hungpv/projects/TN/data/data_musique/dev_queries_en.json"
#     mapping_docs_path = "/home/hungpv/projects/TN/data/data_musique/filter_corpus_vi.json"
#     with open(result_path,'r') as f:
#         data_eval = json.load(f)
#     with open(mapping_query_path,'r') as f:
#         mapping_query=json.load(f)
#     with open(mapping_docs_path,'r') as f:
#         mapping_docs = json.load(f)
#     # Process the queries
#     eval_set = []
#     for model,sample in data_eval.items():
#         query_data = []
#         for qid,docs in sample.items():
#             q = mapping_query[qid]
#             chunks = []
#             for doc in docs:
#                 if doc in mapping_docs:
#                     d = mapping_docs[doc]
#                 else:
#                     if isinstance(doc, list):
#                         d = doc[0]
#                 chunks.append(d)
#             item = {
#                 "query": q,
#                 "chunks": d
#             }
#             query_data.append(item)
#         eval_set.append([model, query_data])
#     for item in eval_set:
#         model_name = item[0]
#         query_data = item[1]
#         output_dir = f"./answer_{model_name}.json"
#         results = processor.process_queries(model_name=model_name,
#                                             query_data=query_data,
#                                             output_file=output_dir)
#         break
    
#     # Save the results
#     # processor.save_results(results, "dickens_answers.json")
    
#     # Print the results
#     # print(json.dumps(results, indent=4))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process evaluation queries with an LLM.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the evaluation result JSON file.")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query mapping JSON file.")
    parser.add_argument("--docs_path", type=str, required=True, help="Path to the document mapping JSON file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model name to use.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save output answers.")
    parser.add_argument("--only_first_model", action="store_true", help="Only process the first model in eval set.")
    parser.add_argument("--language", type=str, default="English")
    parser.add_argument("--top_k", type=int, default="English")
    args = parser.parse_args()

    # Khởi tạo processor
    processor = LLMQueryProcessor(
        model_name=args.model_name
    )

    with open(args.result_path, 'r') as f:
        data_eval = json.load(f)
    with open(args.query_path, 'r') as f:
        mapping_query = json.load(f)
    with open(args.docs_path, 'r') as f:
        mapping_docs = json.load(f)

    # Tạo eval set
    eval_set = []
    for model, sample in data_eval.items():
        query_data = []
        sorted_sample = sorted(sample.items(), key=lambda x: int(x[0].split('_')[1]))
        for qid, docs in sorted_sample:
            q = mapping_query[qid]
            chunks = []
            for doc in docs:
                if isinstance(doc, list):
                    d = doc[0]
                elif doc in mapping_docs:
                    d = mapping_docs[doc]
                else:
                    d = ""
                chunks.append(d)
            query_data.append({
                "query": q,
                "chunks": chunks
            })
        eval_set.append([model, query_data])
        if args.only_first_model:
            break

    # Xử lý
    for model_name, query_data in eval_set:
        output_file = f"{args.output_dir}/answer_{model_name}.json"
        processor.process_queries(
            model_name=model_name,
            query_data=query_data,
            output_file=output_file,
            lang=args.language,
            top_k=args.top_k
        )

if __name__ == "__main__":
    main()
