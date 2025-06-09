from tavily import TavilyClient
from typing import List
import logging

class TavilySearch:
    def __init__(self, api_key: str):
        self.tavily_client = TavilyClient(api_key=api_key)
    
    def search(self, search_query: str, max_results=5):
        search_result = self.tavily_client.search(query=search_query, search_depth="basic", max_results=max_results)
        results = search_result['results']
        result_str = "\n".join([f"{res["title"]}: {res['content']}" for res in results])
        return result_str

    
    def search_authors(self, author_names: List[str], max_results=5):
        result_txt = ""
        for auth_name in author_names:
            result = self.search(search_query=auth_name, max_results=max_results)
            result_txt += f"{result}\n"
           
        return result_txt
    