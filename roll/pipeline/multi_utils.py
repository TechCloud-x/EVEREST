import re
import ast

def parse_points_text_from_content(content):
    """
    """

    answer_pattern = r"<answer>(.*?)</answer>"

    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        points_text = answer_match.group(1)
        return points_text.strip()
    else:

        return ""
