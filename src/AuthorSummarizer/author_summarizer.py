from typing import List, Dict
import logging
import os
import glob
from omegaconf import OmegaConf
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path

from src.util import (
    filter_top_K_papers,
    sort_authors_citations,
    filter_authors_with_few_citations,
    save_json_write,
    save_json_load,
    save_file_load,
    sanitize_folder_name,
    check_files_with_pattern_exist
)
from src.scholar_api.semantic_scholar import SemanticScholarApi
from src.WebSearch.tavily_search import TavilySearch
from src.LLMProcessor.llm_processor import LLMProcessor

@dataclass
class AuthorSummary:
    summary: str
    success: bool


class AuthorSummarizer:
    def __init__(self, 
                 cfg,
                 is_user_input: bool,
                 top_K_papers: int,
                 citation_threshold: int,
                 num_web_results: int,
                 llm_model_cfg: dict,
                 is_load_precomputed_results: bool=False):
         # read api keys from .env
        load_dotenv()
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.cfg = cfg
        self.is_user_input = is_user_input
        self.top_K_papers = top_K_papers
        self.citation_threshold = citation_threshold
        self.num_web_results = num_web_results
        self.is_load_precomputed_results = is_load_precomputed_results

        # scholar API
        self.scholar = SemanticScholarApi()
        # module creating summaries using a large language model
        self.llm_processor = LLMProcessor(
            model=llm_model_cfg.model,
            temperature=llm_model_cfg.temperature,
            save_summaries=True,
        )
        # web search module
        self.web_search = TavilySearch(api_key=tavily_api_key)

    def create_author_summary_from_paper(self, paper_name: str) -> bool:
        # set output directory where intermediate results are saved to
        output_dir = f"output/{paper_name}"
        output_dir = sanitize_folder_name(output_dir)
        intermediate_out_dir = f"{output_dir}/intermediate_results"
        self.llm_processor.set_output_directory(output_dir)

        # in debugging mode, large parts are skipped adn the intermediate results are directly loaded.
        skip_scholar_extraction = self.is_load_precomputed_results and Path(f"{output_dir}/scholar.json").exists()
        removed_authors = []
        if skip_scholar_extraction:
            authors_data_loaded = save_json_load(f'{output_dir}/scholar.json')
            if authors_data_loaded is not None:
                authors_data = authors_data_loaded
                author_names = [data["name"] for data in authors_data]
            else:
                raise FileNotFoundError(f"Loading file {output_dir}/scholar.json failed.")
            
        # extract information about the papers authors from scholar and filter relevant data
        else:
            # load for the paper the available information from scholar (http request)
            authors_data, paper_data, is_success = self.scholar.extract_information(paper_name)
            if not is_success:
                return AuthorSummary(summary=None, success=False)

            author_names_before = [data["name"] for data in authors_data]
            # filter out authors with few citations, sort them according to citation count and keep only most relevant papers.
            authors_data = self._filter_scholar_data(authors_data, paper_data)
            author_names = [data["name"] for data in authors_data]
            removed_authors = list(set(author_names_before).difference(set(author_names)))
            
            # summarize the abstracts of each authors most relevant papers using a LLM
            for auth_data in authors_data:
                for paper in auth_data['papers']:
                    paper["abstract_summary"] = self.llm_processor.summarize_abstract(abstract=paper["abstract"])
                # save the authors information
                save_json_write(authors_data, f'{intermediate_out_dir}/scholar.json')
       
        # summarize the scholar information using a LLM
        skip_scholar_summary = self.is_load_precomputed_results and Path(f"{output_dir}/scholar_summary.txt").exists()
        if skip_scholar_summary:
            scholar_summary = save_file_load(f"{output_dir}/scholar_summary.txt")
            if scholar_summary is None:
                raise FileNotFoundError(f"Loading file {output_dir}/scholar_summary.txt failed.")
        else:
            scholar_summary = self.llm_processor.summarize_scholar_information(
                paper_name=paper_name,
                authors_information=authors_data
            )

        # perform a web-search about each author
        author_seach_results = {}
        skip_web_search = self.is_load_precomputed_results and check_files_with_pattern_exist(output_dir, "search_result_*.txt")
        if skip_web_search:
            # for each author, load the search result summary
            for auth_name in author_names:
                file_path = f'{output_dir}/search_result_{auth_name}.txt'
                search_result = save_file_load(file_path)
                if search_result is None:
                    raise FileNotFoundError(f"Loading file {file_path} failed.")
                author_seach_results[auth_name] = search_result
        else:
            for auth_name in author_names:
                search_result = self.web_search.search(search_query=auth_name, max_results=self.num_web_results)
                author_seach_results[auth_name] = search_result

        # summarize the web search result about each author using an llm
        skip_search_summary = self.is_load_precomputed_results and check_files_with_pattern_exist(output_dir, "sr_summary_*.txt")
        search_summary = {}
        if skip_search_summary:
            for auth_name in author_names:
                file_path = f'{output_dir}/sr_summary_{auth_name}.txt'
                summary = save_file_load(file_path)
                if summary is None:
                    raise FileNotFoundError(f"Loading file {file_path} failed.")
                search_summary[auth_name] = summary
        else:
            for auth_name, search_result in author_seach_results.items():
                summary = self.llm_processor.summarize_author_web_search(author_name=auth_name, search_text=search_result)
                search_summary[auth_name] = summary
        
        # append each authors summary to one text
        search_results_summary = ""
        for auth_name, summary in search_summary.items():
            search_results_summary += f"{auth_name}:\n{summary}\n"
            
        # merge web search results and scholar result to a final summary using an llm
        final_summary = self.llm_processor.create_final_summary(
            paper_name=paper_name,
            scholar_summary=scholar_summary,
            web_summary=search_results_summary,
        )

        if len(removed_authors) > 0:
            final_summary += f"\n\n The authors: {removed_authors} are not well known."

        html_summary = self.llm_processor.create_html_output(final_summary)
        print(final_summary)

        return AuthorSummary(summary=final_summary, success=True)

    
    def _filter_scholar_data(self, authors_data: dict, paper_data: dict):
        for auth_data in authors_data:
            auth_data["papers"] = filter_top_K_papers(auth_data["papers"], k=self.top_K_papers)

        authors_data = sort_authors_citations(authors_data)
        # filter out authors with too low number of citations
        authors_data = filter_authors_with_few_citations(
            authors_data, num_citations=self.citation_threshold, paper_citations=paper_data['citationCount']
        )
        return authors_data
    
    