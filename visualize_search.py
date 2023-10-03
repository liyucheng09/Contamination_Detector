from difflib import SequenceMatcher
import json

def compare_strings(query: str, match: str):
    result = []
    s = SequenceMatcher(None, query, match)    
    for opcode, a0, a1, b0, b1 in s.get_opcodes():
        if opcode == 'equal':
            result.append(f'<span class="diff">{query[a0:a1]}</span>')
        else:
            result.append(query[a0:a1])
    result.append('</div>')
    return ''.join(result)

if __name__ == '__main__':

    num_samples = 5
    benchmark = 'mmlu'

    html_string = """
<html>
<head>
<title>Contamination Report for """ + benchmark + """. </title>
<style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #000;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    .highlight {
        background-color: yellow;
    }
    .title {
        font-size: 15px;
        margin-bottom: 10px;
    }
    .datasetname {
        font-size: 60px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .diff {
        background: #cee1f4;
    }
</style>
</head>
<body>
<div class = "datasetname">""" + benchmark.upper() + "</div>"

    with open(f'reports/{benchmark}.json') as f:
        report = json.load(f)
    
    for results in report['matches'][:num_samples]:
        table_string = f"""
    <div class="title"><b>Evaluation sample:</b> {results[0]['query']}</div>
    <table>
        <tr>
            <th>Page Name</th>
            <th>Overlapping</th>
            <th>Match Ratio</th>
            <th>URL</th>
        </tr>
        """
        for result in results:
            query = result['query']
            match_string = result['match_string']
            score = result['score']
            name = result['name']
            url = result['url']
            # snippet = result['snippet'].replace('<b>', '').replace('</b>', '')
            snippet = result['snippet']

            # snippet = compare_strings(snippet, query)
            rwo_string = f"""
        <tr>
            <td>{name}</td>
            <td>{snippet}</td>
            <td>{score:.3f}</td>
            <td><a href="{url}">Link</a></td>
        </tr>
        """
            table_string += rwo_string
        table_string += """
    </table>
    <br>
    """
        html_string += table_string

    html_string += """
</body>
</html>
"""

    with open(f'reports/{benchmark}.html', 'w') as f:
        f.write(html_string)
