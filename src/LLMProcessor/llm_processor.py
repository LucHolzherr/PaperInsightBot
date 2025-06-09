from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from pathlib import Path

import logging
from src.util import create_string_of_author_data, save_file_write


class LLMProcessor:
    def __init__(
            self, 
            model: str = "gpt-4", 
            temperature=0.3, 
            save_summaries: bool=True, 
            save_path: str='output'):
        # define large language model used
        self._llm = ChatOpenAI(model=model, temperature=temperature)
        self.save_summaries = save_summaries
        self.output_dir = save_path
        # create path if it does not exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # define the prompts for the different use-cases
        # prompt for paper-abstract summary creation
        abstract_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant who writes concise academic summaries.",
                ),
                (
                    "user",
                    "Summarize the following abstract in a maximum 3 short sentences or keywords, focus mainly on the research direction, avoid long formulations. :\n\n{abstract_text}",
                ),
            ]
        )
        self._abstract_chain = abstract_prompt | self._llm

        # prompt and chain for summarizing the entire author scholar profile
        scholar_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant who writes concise academic summaries.",
                ),
                (
                    "user",
                    """
             Summarize for the following authors their academic impact in a short paragraph.
             They coauthored the paper {paper_name}.
             If they coauthored a impactful paper, do not listen it for every author, but state that they coauthored a paper.
             It should be clear from your summary which authors have other impactful papers and which only have the common one.
             {authors_info}
             """,
                ),
            ]
        )
        self._scholar_chain = scholar_prompt | self._llm

        # web search prompt
        web_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant who writes concise academic summaries.",
                ),
                (
                    "user",
                    "Summarize the public research profile and affiliations of {author_name} based on this search:\n\n{search_text}",
                ),
            ]
        )
        self._web_chain = web_prompt | self._llm

        # final summary prompt
        final_summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert research assistant who writes concise academic summaries.",
                ),
                (
                    "user",
                    """
             From the google scholar summary and the web search summary below about these researchers, create a final summary of each author. 
             Highlight their affiliations, the focus of their research, important papers or projects they worked on, specifically mention their citations count. 
             If they coauthored a paper other than {paper_name}, mention this.
             Google scholar summary:
             {scholar_summary}

             Web search summary:
             {web_summary}
             """,
                ),
            ]
        )
        self._final_summary_chain = final_summary_prompt | self._llm

    def set_output_directory(self, output_dir: str):
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_final_summary(
        self, paper_name: str, scholar_summary: str, web_summary: str
    ):
        final_summary = self._inference(
            self._final_summary_chain,
            prompt_input_dict={
                "paper_name": paper_name,
                "scholar_summary": scholar_summary,
                "web_summary": web_summary,
            }
        )

        if self.save_summaries:
            save_file_write(f'{self.output_dir}/final_summary.txt', data=final_summary)
        
        return final_summary

    def summarize_author_web_search(self, author_name: str, search_text: str):
        
        author_web_summary = self._inference(
            self._web_chain, 
            prompt_input_dict={"author_name":author_name, "search_text":search_text}
        )

        if self.save_summaries:
            save_file_write(f'{self.output_dir}/{author_name}_websummary.txt', data=author_web_summary)
           
        return author_web_summary
    
    # def dummy_summarize_author_web_search(self, author_name: str, *args, **kwargs):
    #     file_path = f"{self.output_dir}/sr_summary_{author_name}.txt"
    #     try:
    #         with open(file_path, "r") as f:
    #             summary = f.read()
    #     except:
    #         logging.info(f"Could not read file {file_path}")
    #         return None

    #     return summary
    
    def summarize_abstract(self, abstract: str):
        return self._inference(self._abstract_chain, prompt_input_dict={"abstract_text":abstract})

    def summarize_scholar_information(self, paper_name: str, authors_information: list):
        # create a llm readable text from the authors_information dictionary
        data_str = ""
        author_texts = []
        for auth_data in authors_information:
            auth_str: str = create_string_of_author_data(auth_data)
            author_texts.append(auth_str)
            data_str += auth_str + "\n"

        scholar_summary = self._inference(
            self._scholar_chain, prompt_input_dict={"paper_name": paper_name, "authors_info": data_str}
        )

        if self.save_summaries:
            save_file_write(f'{self.output_dir}/scholar_summary.txt', data=scholar_summary)

        return scholar_summary

    # def dummy_summarize_scholar_information(self, *args, **kwargs):
    #     file_path = f"{self.output_dir}/scholar_summary.txt"
    #     try:
    #         with open(file_path, "r") as f:
    #             scholar_summary = f.read()
    #     except:
    #         logging.info(f"Could not load precomputed scholar summary from {file_path}")
    #         return None
        
    #     return scholar_summary

    @staticmethod
    def _inference(chain, prompt_input_dict):
        response = chain.invoke(prompt_input_dict)
        return response.content
