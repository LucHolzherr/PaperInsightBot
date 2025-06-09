from tavily import TavilyClient
from typing import List

class TavilySearch:
    """
    A wrapper class for performing web searches using the Tavily API.

    This class provides methods to perform basic searches and aggregate results
    for multiple author names.
    """
    def __init__(self, api_key: str):
        """
        Initialize the TavilySearch instance with an API key.

        Args:
            api_key (str): The Tavily API key used to authenticate requests.
        """
        self.tavily_client = TavilyClient(api_key=api_key)
    
    def search(self, search_query: str, max_results=5) -> str:
        """
        Perform a basic web search for input query.

        Args:
            search_query (str): The query string to search for.
            max_results (int, optional): Maximum number of results to retrieve. Defaults to 5.

        Returns:
            str: A newline-separated string containing the titles and contents of the top search results.
        """
        search_result = self.tavily_client.search(query=search_query, search_depth="basic", max_results=max_results)
        results = search_result['results']
        result_str = "\n".join([f"{res["title"]}: {res['content']}" for res in results])
        return result_str

    
    def search_authors(self, author_names: List[str], max_results=5) -> str:
        """
        Perform web searches for a list of author names and return results as one string.

        Args:
            author_names (List[str]): A list of author names to search for.
            max_results (int, optional): Maximum number of results per author. Defaults to 5.

        Returns:
            str: A concatenated string of search summaries for all given authors.
        """
        result_txt = ""
        for auth_name in author_names:
            result = self.search(search_query=auth_name, max_results=max_results)
            result_txt += f"{result}\n"
           
        return result_txt
    