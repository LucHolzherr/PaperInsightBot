import argparse
import logging
import os
from omegaconf import OmegaConf
from dotenv import load_dotenv

from src.AuthorSummarizer.author_summarizer import AuthorSummarizer, AuthorSummary

# TODO: 
# - Change to google scholar, semanticScholar is missing many papers
# - Search result verification if correct person
# - provide web-search urls that were used

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # read api keys from .env
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # read config with omegaconf and parse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    is_user_input = config.is_user_input
    top_K_papers = config.num_papers_considered
    citation_threshold = config.author_citations_threshold
    num_web_results = config.num_web_results
    is_load_precomputed_results = config.is_load_precomputed_results
    llm_model_cfg = config.llm

    author_summarizer = AuthorSummarizer(
        cfg=config,
        is_user_input=is_user_input, 
        top_K_papers=top_K_papers,
        citation_threshold=citation_threshold,
        num_web_results=num_web_results,
        llm_model_cfg=llm_model_cfg,
        is_load_precomputed_results=is_load_precomputed_results)
    
    success = False
    while not success:
        if not is_user_input:
            paper_name =  config.paper_name
        else:
            paper_name = input("Enter the paper name: ")

        output: AuthorSummary = author_summarizer.create_author_summary_from_paper(paper_name)
        success = output.success

        if not success:
            print(f"Fetching Author information failed.")
            if is_user_input:
                print(f"Input paper name again, check for typos.")
            else:
                # if paper name is from config, the program just finished here.
                print(f"check config 'paper_name' field for typos.")
                break
    
