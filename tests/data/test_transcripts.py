import math
import pytest
from urop.data.transcripts import build_record, select_record, union, normalize, clean_answer, review_packet

Q=[{'slot':i,'type':'Follow-up' if i==1 else 'Primary'} for i in range(85)]
M={'question a':{'status':'mapped','slot':0},'follow':{'status':'mapped','slot':1},'yeah':{'status':'ack','slot':None}}

def row(speaker,text,start=0,end=1):
    return {'speaker':speaker,'value':text,'start_time':str(start),'stop_time':str(end)}

def build(rows):
    return build_record(303,'train',rows,100,M,Q)

def test_unknown_boundary_ack_and_source_coverage():
    r=build([row('Ellie','question a'),row('Participant','one',1,2),row('Ellie','yeah'),row('Participant','two',2,3),row('Ellie','unknown'),row('Participant','unknown answer',4,5),row('Ellie','follow'),row('Participant','three',6,7)])
    assert r['events'][0]['answer_text']=='one two'
    assert r['events'][1]['answer_text']=='unknown answer'
    assert r['events'][2]['parent_unknown']
    assert select_record(r,'all')['slots'][0]['text']=='one two'
    assert sum(r['audit'].values())==8
    assert all('classification' in x for x in r['source_rows'])

def test_chains_and_repeated_first_last_all():
    r=build([row('Ellie','question a'),row('Participant','first',1,3),row('Ellie','follow'),row('Participant','a',4,5),row('Ellie','follow'),row('Participant','b',6,7),row('Ellie','question a'),row('Participant','last',8,9)])
    assert r['events'][2]['followup_depth']==2
    assert r['events'][2]['preceding_question_event_id']==r['events'][1]['event_id']
    assert r['events'][2]['root_primary_event_id']==r['events'][0]['event_id']
    assert select_record(r,'first')['slots'][0]['text']=='first'
    assert select_record(r,'last')['slots'][0]['text']=='last'
    assert select_record(r,'all')['slots'][0]['text']=='first last'
    assert select_record(r,'all')['slots'][0]['spans']==[[1.,3.],[8.,9.]]

def test_invalid_clipped_scrubbed_and_prequestion():
    r=build([row('Participant','before'),row('Ellie','question a'),row('Participant','xxx'),row('Participant','word xxx preserved',-2,3),row('Participant','bad',math.nan,3),row('Participant','tail',99,105),row('Participant','scrubbed_entry')])
    assert r['events'][0]['answer_text']=='word preserved tail'
    assert r['events'][0]['answer_intervals']==[[0.,3.],[99.,100.]]
    assert r['source_rows'][0]['classification']=='prequestion'
    assert r['source_rows'][4]['classification']=='excluded_invalid_time'
    assert clean_answer('xxx hello scrubbed_entry')=='hello'

def test_normalization_and_union():
    assert normalize('how_do_you_like_la (how do you like l_a)')=='how do you like la'
    assert normalize('yeah3 (yeah)')=='yeah'
    assert normalize("don't")==normalize('dont')
    assert union([[2,4],[1,3],[7,8]])==[[1,4],[7,8]]

def test_review_unique_train_only():
    records=[]
    for pid in range(300,304):
        record=build([row('Ellie','question a'),row('Participant','x')]*60)
        record['split']='dev' if pid==303 else 'train'
        for event in record['events']:
            event['event_id']=f'{pid}:{event["question_row_id"]}'
        records.append(record)
    packet=review_packet(records)
    ids=[p['event']['event_id'] for p in packet['packets']]
    assert len(ids)==len(set(ids))
    assert all(not x.startswith('303:') for x in ids)


def test_reject_nonfinite_union_and_invalid_policy():
    with pytest.raises(ValueError): union([[0,math.nan]])
    with pytest.raises(ValueError): build_record(303,'train',[],math.inf,M,Q)
    with pytest.raises(ValueError): select_record(build([]),'typo')
    assert clean_answer('[scrubbed_entry]')==''
