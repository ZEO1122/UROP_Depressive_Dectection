"""Guard the notebook's default execution boundary without launching training."""
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / 'experiments/hique/v3_input_study/pipeline.ipynb'


def code_cells():
    return [c for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code']


def test_notebook_cells_compile_and_default_to_read_only():
    cells = code_cells()
    for cell in cells:
        compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
    settings = next(''.join(c['source']) for c in cells if 'settings' in c['metadata']['tags'])
    # Execute the literal mode assignments only, without loading local datasets.
    import ast
    assignments = [node for node in ast.parse(settings).body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ('RUN_PIPELINE', 'RUN_TEST')
                           for t in node.targets)]
    values = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<defaults>', 'exec'), values)
    assert values['RUN_PIPELINE'] is False
    assert values['RUN_TEST'] is False


def test_saved_test_toggle_does_not_change_semantic_recipe():
    import ast
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(c for c in code_cells() if 'inventory' in c['metadata']['tags'])
    # Evaluate the notebook's actual code_sources assignment against saved documents.
    assignment = next(node for node in ast.walk(ast.parse(''.join(cell['source'])))
                      if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'code_sources' for t in node.targets))
    expression = compile(ast.Expression(assignment.value), '<recipe>', 'eval')
    before = eval(expression, {'saved': notebook})
    settings = next(c for c in notebook['cells'] if 'settings' in c.get('metadata', {}).get('tags', []))
    settings['source'] = ''.join(settings['source']).replace('RUN_TEST = False', 'RUN_TEST = True')
    assert eval(expression, {'saved': notebook}) == before
    other = next(c for c in notebook['cells'] if 'text' in c.get('metadata', {}).get('tags', []))
    other['source'] = ''.join(other['source']) + '\n# changed feature recipe'
    assert eval(expression, {'saved': notebook}) != before
