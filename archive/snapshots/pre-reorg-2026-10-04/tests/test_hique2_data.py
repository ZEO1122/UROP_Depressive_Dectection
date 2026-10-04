from experiments.hique2_data import build_record


def test_followup_chain_parent_change_and_last_occurrence():
    questions = [{'type': 'Primary' if i < 66 else 'Follow-up'} for i in range(85)]
    mapping = {'job?': {'slot': 4}, 'more?': {'slot': 66}, 'why?': {'slot': 67}, 'family?': {'slot': 8}}
    rows = []
    for index, question in enumerate(['job?', 'more?', 'why?', 'family?', 'more?']):
        rows.extend([{'speaker': 'Ellie', 'text': question, 'start': index*4, 'end': index*4+1},
                     {'speaker': 'Participant', 'text': f'answer{index}', 'start': index*4+1, 'end': index*4+3}])
    r = build_record(301, 'train', rows, mapping, questions)
    assert r['position_ids'][66] == 8
    assert r['position_ids'][67] == 4
    assert r['position_ids'][10] == 10
    assert r['audit']['cross_parent_collisions'] == 1
    assert next(s for s in r['slots'] if s['slot'] == 66)['text'] == 'answer4'
    assert not r['events'][1]['retained']
    assert r['events'][2]['followup_depth'] == 2


def test_orphan_followup_and_missing_answer():
    questions = [{'type': 'Follow-up'} for _ in range(85)]
    rows = [{'speaker': 'Ellie', 'text': 'why?', 'start': 0, 'end': 1},
            {'speaker': 'Participant', 'text': 'yes', 'start': 1, 'end': 2},
            {'speaker': 'Ellie', 'text': 'why?', 'start': 3, 'end': 4}]
    r = build_record(301, 'train', rows, {'why?': {'slot': 67}}, questions)
    assert r['position_ids'][67] == 67
    assert r['slots'][0]['orphan_fallback']
    assert r['events'][-1]['reason'] == 'no_following_response'
