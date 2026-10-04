from experiments.hique2_asr import tag_and_merge


def test_literal_question_and_merge():
    segments = [dict(start=0, end=1, text='Hello?'),
                dict(start=1, end=2, text=' Really?'),
                dict(start=2, end=3, text=' Yes.'),
                dict(start=3, end=4, text='Question? ')]
    tagged, rows = tag_and_merge(segments)
    assert [s['speaker'] for s in tagged] == ['Ellie', 'Ellie', 'Participant', 'Participant']
    assert rows == [dict(start=0, end=2, text='Hello?  Really?', speaker='Ellie'),
                    dict(start=2, end=4, text=' Yes. Question? ', speaker='Participant')]
    assert 'speaker' not in segments[0]


def test_empty_and_distinct():
    assert tag_and_merge([]) == ([], [])
    tagged, rows = tag_and_merge([dict(start=3, end=4, text='No'), dict(start=8, end=9, text='Why?')])
    assert len(rows) == 2
    assert rows[1]['start'] == 8


def test_official_roster_case_and_exclusion(tmp_path):
    from experiments.hique2_asr import official_ids
    roster = [pid for pid in range(300, 493) if pid not in {342, 394, 398, 460}]
    for split, ids, header in [('train', roster[:107], 'Participant_ID'),
                               ('dev', roster[107:142], 'Participant_ID'),
                               ('test', roster[142:], 'participant_ID')]:
        (tmp_path / f'{split}_split_Depression_AVEC2017.csv').write_text(
            header + ',Gender\n' + ''.join(f'{pid},0\n' for pid in ids))
    actual = official_ids(tmp_path)
    assert len(actual) == 188
    assert 440 not in actual
    assert 300 in actual and 451 in actual and 458 in actual and 480 in actual
