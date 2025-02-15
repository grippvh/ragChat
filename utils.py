import re

def normalize_numbers(query: str) -> str:
    """
    Normalizes numeric formats in the query.

    1. Converts commas used as decimal separators (e.g., "3,14") into periods ("3.14").
    2. Removes thousand separators (e.g., converts "5.000" to "5000") by finding numbers
       that match the pattern for thousand-separated digits.
    """
    # replace commas between digits with a period (decimal separator normalization)
    query = re.sub(r'(?<=\d),(?=\d)', '.', query)

    # remove thousand separators: look for numbers like "1.234" or "12.345.678"
    def remove_thousands(match):
        # Remove all dots (thousand separators) from the matched string.
        return match.group(0).replace('.', '')

    # pattern matches numbers with one to three digits, then at least one group of a dot
    # followed by exactly three digits, and word boundaries to avoid matching parts of a larger string.
    query = re.sub(r'\b\d{1,3}(?:\.\d{3})+\b', remove_thousands, query)

    return query

def _build_context(results):
    return "\n\n---\n\n".join([doc.page_content for doc, _ in results])