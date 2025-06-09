from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path

from src.util import create_string_of_author_data, file_write


class LLMProcessor:
    """
    A processor class that leverages a large language model (LLM) to generate structured summaries 
    and outputs from academic data sources such as abstracts, Google Scholar profiles, and web search results.

    It supports generating concise academic summaries, formatting them in HTML, and saving them to disk.
    """
    def __init__(
            self, 
            model: str = "gpt-4", 
            temperature=0.3, 
            save_summaries: bool=True, 
            save_path: str='output'):
        """
        Initialize the LLMProcessor with a specific language model and configuration.

        Args:
            model (str): The name of the OpenAI model to use (e.g., "gpt-4").
            temperature (float): Sampling temperature for the LLM (controls randomness).
            save_summaries (bool): Whether to save intermediate and final summaries to disk.
            save_path (str): Directory path where results should be saved.
        """
        # define large language model used
        self._llm = ChatOpenAI(model=model, temperature=temperature)
        self.save_summaries = save_summaries
        self.output_dir = save_path
        self.intermediate_out_dir = f"{save_path}/intermediate_results"
        # create path if it does not exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_out_dir).mkdir(parents=True, exist_ok=True)
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
             Do not highlight the paper {paper_name}, as the user already knows that the authors have coauthored this paper.
             Google scholar summary:
             {scholar_summary}

             Web search summary:
             {web_summary}
             """,
                ),
            ]
        )
        self._final_summary_chain = final_summary_prompt | self._llm

        # prompt to format the final summary to a nice looking HTML
        display_prompt = ChatPromptTemplate.from_messages(
            [
                 (
                    "system",
                    "You are an expert at formatting academic content for clean HTML display.",
                ),
                (
                    "user",
                    """
                    Format the following author summary into clean, semantic HTML for use on a webpage.

                    - Use <h2> for each author name.
                    - Display the text content for each author unchanged below the author name.

                    Author summary:

                    {author_summary}
                    """,
                ),
            ]
        )
        self._display_chain = display_prompt | self._llm

    def set_output_directory(self, output_dir: str):
        """
        Set a new output directory and ensure subdirectories are created.

        Args:
            output_dir (str): Path to the new output directory.
        """
        self.output_dir = output_dir
        self.intermediate_out_dir = f"{output_dir}/intermediate_results"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_out_dir).mkdir(parents=True, exist_ok=True)
    
    def create_final_summary(
        self, paper_name: str, scholar_summary: str, web_summary: str
    ) -> str:
        """
        Create a final summary of all authors by combining the Scholar and web search summaries.

        Args:
            paper_name (str): Title of the paper the authors co-authored.
            scholar_summary (str): Text summary based on authors' Scholar profiles.
            web_summary (str): Text summary based on web search data.

        Returns:
            str: Combined final summary of authors, created by the LLM
        """
        final_summary = self._inference(
            self._final_summary_chain,
            prompt_input_dict={
                "paper_name": paper_name,
                "scholar_summary": scholar_summary,
                "web_summary": web_summary,
            }
        )

        if self.save_summaries:
            file_write(f'{self.output_dir}/final_summary.txt', data=final_summary)
        
        return final_summary

    def summarize_author_web_search(self, author_name: str, search_text: str) -> str:
        """
        Generate a summary of an author's profile and affiliations based on a web search.

        Args:
            author_name (str): Name of the author.
            search_text (str): Raw web search result text.

        Returns:
            str: LLM-generated summary of the author's public profile.
        """
        author_web_summary = self._inference(
            self._web_chain, 
            prompt_input_dict={"author_name":author_name, "search_text":search_text}
        )

        if self.save_summaries:
            file_write(f'{self.intermediate_out_dir}/{author_name}_websummary.txt', data=author_web_summary)
           
        return author_web_summary
    
    def summarize_abstracst(self, abstract: str) -> str:
        """
        Generate a short summary of an academic paper abstract.

        Args:
            abstract (str): The abstract text of the paper.

        Returns:
            str: Concise summary or keywords describing the research direction.
        """
        return self._inference(self._abstract_chain, prompt_input_dict={"abstract_text":abstract})

    def summarize_scholar_information(self, paper_name: str, authors_information: list):
        """
        Summarize authors' research profiles based on structured Scholar information.

        Args:
            paper_name (str): Title of the paper all authors co-authored.
            authors_information (list): List of dictionaries for each author.

        Returns:
            str: LLM-generated summary of each author's research impact.
        """
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
            file_write(f'{self.intermediate_out_dir}/scholar_summary.txt', data=scholar_summary)

        return scholar_summary
    
    def create_html_output(self, final_summary: str) -> str:
        """
        Format a plain-text author summary into styled HTML and save to disk.

        Args:
            final_summary (str): Final plain-text author summary.

        Returns:
            str: HTML-formatted version of the summary.
        """
        html_text = self._inference(self._display_chain, prompt_input_dict={'author_summary': final_summary})
        file_write(file_path=f"{self.output_dir}/final_summary.html", data=html_text)
        return html_text

    @staticmethod
    def _inference(chain, prompt_input_dict):
        response = chain.invoke(prompt_input_dict)
        return response.content
