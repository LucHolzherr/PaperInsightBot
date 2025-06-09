import requests
import logging
from typing import Tuple, List

class SemanticScholarApi:
    """
    A wrapper class for accessing the Semantic Scholar API to retrieve academic paper and author information.

    Provides methods to search for a paper by title and extract author metadata including citation counts,
    h-index, affiliations, and paper history.
    """
    def __init__(self):
        self.base_paper_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.base_author_url = "https://api.semanticscholar.org/graph/v1/author"


    def extract_information(self, paper_name: str) -> Tuple[List[dict], dict, bool]:
        """
        Search for a paper by title and retrieve associated author profiles.

        Args:
            paper_name (str): The title of the academic paper to search.

        Returns:
            tuple: A 3-element tuple containing:
                - authors_data (list): List of dictionaries with author metadata or None if paper not found.
                - paper_data (dict or None): Dictionary of paper metadata, or None if not found.
                - success (bool): True if the operation succeeded, False otherwise.
        """
        try:
            paper_data = self._search_paper(title_query=paper_name)
        except requests.exceptions.RequestException as e:
            print(f"Semantic Scholar Request for the paper failed: {e}")
            return None, None, False
        
        authors = paper_data["authors"]
        authors_data = []
        for author in authors:
            auth_data = self._get_author_publications(author["authorId"])
            if not auth_data:
                logging.warning(f"Could not extract info for {author["name"]}")
                continue

            authors_data.append(auth_data)
        
        return authors_data, paper_data, True

    def _search_paper(self, title_query) -> dict:
        """
        Search Semantic Scholar for a paper by title and return structured metadata.

        Args:
            title_query (str): The title of the paper to search for.

        Returns:
            dict or None: Paper metadata or None if paper not found.
        """
        params = {
            "query": title_query,
            "limit": 1,
            "fields": "title,abstract,year,authors,citationCount,url,externalIds"
        }
        
        response = requests.get(self.base_paper_url, params=params)
        response.raise_for_status() # Raise an error for bad status codes (4xx, 5xx)
        data = response.json()
        
        if data.get("data"):
            paper = data["data"][0]
            return {
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount"),
                "authors": paper.get("authors"),
                "url": paper.get("url"),
                "externalIds": paper.get("externalIds")
            }
        else:
            return None
    
    def _get_author_publications(self, author_id: int) -> dict:
        """
        Retrieve author metadata and a list of their publications from Semantic Scholar.

        Args:
            author_id (int): Semantic Scholar's unique ID for the author.

        Returns:
            dict or None: Author information or None on failure.
        """
        url = f"{self.base_author_url}/{author_id}"
        params = {
            "fields": "name,affiliations,citationCount,hIndex,papers.title,papers.year,papers.citationCount,papers.abstract"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Author request failed with status code: {response.status_code}")
            return None

        data = response.json()
        if "name" not in data:
            return None
        
        author_info = {
            "name": data["name"],
            "affiliations": data.get("affiliations", ""),
            "citation_count": data.get("citationCount", 0),
            "h_index": data.get("hIndex", 0),
            "papers": []
        }

        for paper in data.get("papers", []):
            author_info["papers"].append({
                "title": paper.get("title"),
                "year": paper.get("year"),
                "citations": paper.get("citationCount"),
                "abstract": paper.get("abstract")
            })
        
        return author_info