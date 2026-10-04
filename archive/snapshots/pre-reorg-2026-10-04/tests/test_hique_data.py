import numpy as np

from experiments.hique_data import build_slots, is_question, length_batches, normalize, subtract_intervals


def row(speaker, start, stop, text):
    return {'speaker': speaker, 'start_time': str(start), 'stop_time': str(stop), 'value': text}


def test_overlap_subtraction_never_keeps_interviewer_interval():
    assert subtract_intervals([1, 10], [[0, 2], [4, 6], [9, 12]]) == [[2, 4], [6, 9]]
    assert subtract_intervals([2, 3], [[1, 4]]) == []


def test_slot_assignment_preserves_backchannel_continuation_and_scrub_mask():
    mapping = {'how do you sleep': {'slot': 2, 'method': 'exact'},
               'mhm': {'slot': None, 'method': 'non_question'},
               'where are you from': {'slot': 4, 'method': 'exact'}}
    rows = [row('Ellie', 0, 2, 'how do you sleep'), row('Participant', 1, 4, 'a response'),
            row('Ellie', 4, 5, 'mhm'), row('Participant', 5, 7, 'continuation'),
            row('Participant', 8, 9, 'scrubbed_entry'), row('Ellie', 10, 11, 'where are you from'),
            row('Participant', 12, 14, 'another response')]
    result = build_slots(1, 'train', rows, mapping)
    assert result['slots'][0]['text'] == 'a response continuation'
    assert result['slots'][0]['spans'] == [[2, 4], [5, 7]]
    assert result['question_presence'][2] and result['question_presence'][4]
    assert result['mapping_counts']['scrubbed_response_rows'] == 1
    assert 'label' not in result


def test_question_normalization_and_nonquestion_boundary():
    assert normalize("q2 (what’s nice about l_a)") == 'whats nice about la'
    assert is_question('okay how are you', {})
    assert not is_question('however', {})
    assert not is_question('mhm', {})


def test_sorted_batching_visits_each_index_and_preserves_restoration():
    lengths = [512, 10, 50, 9, 100, 200, 10]
    batches = length_batches(lengths, token_budget=512, max_batch=3)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    restored = np.zeros(len(lengths), dtype=int)
    for batch in batches:
        assert len(batch) <= 3
        assert max(lengths[i] for i in batch) * len(batch) <= 512
        for i in batch:
            restored[i] = lengths[i]
    assert restored.tolist() == lengths
