import logging
import json
import re
import os
import glob

def filter_top_K_papers(papers, k: int):
    top_k_papers = sorted(papers, key=lambda x: x["citations"], reverse=True)[0:k]
    return top_k_papers

def sort_authors_citations(authors_data: list):
    return sorted(authors_data, key=lambda x: x["citation_count"], reverse=True)

def filter_authors_with_few_citations(authors_data: list, num_citations: int, paper_citations: int):
    authors_data_new = []
    for auth_data in authors_data:
        citations = auth_data["citation_count"]
        if citations - paper_citations >= num_citations:
            authors_data_new.append(auth_data)
        else:
            logging.warning(f"Author {auth_data['name']} was filtered out because of low number of citations ({citations})")
    
    return authors_data_new

def sanitize_folder_name(name):
    # Remove characters that are invalid in folder names
    sanitized = re.sub(r'[<>:"|?*\n\r\t]', '', name)
    # Strip trailing dots and spaces
    sanitized = sanitized.strip().rstrip('.')
    return sanitized

def create_string_of_author_data(author_data: dict):
    paper_txt = "\n".join(
        [
            f"paper title: {paper['title']}, year: {paper['year']}, number citations: {paper['citations']}. Summary of abstract: {paper["abstract_summary"]}"
            for paper in author_data["papers"]
        ]
    )

    text = (
        f"Author: {author_data['name']}\n"
        f"Total Citations: {author_data['citation_count']}\n"
        f"h-index> {author_data['h_index']}\n"
        f"This are their papers with most citations:\n {paper_txt}\n"
    )
    return text


def json_write(data: dict, out_path: str):
    out_path = sanitize_folder_name(out_path)
    try:
        with open(out_path, 'w', encoding="utf-8") as file:
            json_data = json.dumps(data, indent=4)
            file.write(json_data)
    except:
        logging.error(f"Saving json to {out_path} failed.")

def json_load(json_path: str):
    try:
        with open(json_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
        return data
    except:
        logging.error(f"Loading json {json_path} failed")
        return None
    
def file_load(file_path: str):
    file_path = sanitize_folder_name(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            return data
    except UnicodeDecodeError:
        try:
            # Fallback to system default or latin1 (ISO-8859-1), which always succeeds but may misinterpret characters
            with open(file_path, "r", encoding="latin-1") as f:
                data = f.read()
                return data
        except Exception as e:
            logging.error(f"Could not load file from {file_path}, {e}")
    except Exception as e:
        logging.error(f"Could not load file from {file_path}, {e}")
        return None

def file_write(file_path: str, data: str):
    file_path = sanitize_folder_name(file_path)
    try:
        with open(file_path, "w") as f:
            f.write(data)
    except:
        logging.error(f"Could not save file {file_path}.")
    
def check_files_with_pattern_exist(folder_path: str, pattern: str):
    """ Method checks whether files with naming pattern exist in the folder.
    """
    total_pattern = os.path.join(folder_path, pattern)
    matching_files = glob.glob(total_pattern)
    return len(matching_files) > 0