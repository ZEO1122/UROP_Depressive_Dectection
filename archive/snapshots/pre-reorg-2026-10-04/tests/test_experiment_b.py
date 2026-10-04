import io
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.features import span_mask, visual_values, baseline_rows, union_spans, extract_participant, VISUAL, FEATURES
from experiments.run_b import matrices

class FeatureTests(unittest.TestCase):
    def test_spans_exclude_interviewer_and_half_open_boundaries(self):
        self.assertEqual(span_mask(np.array([0,1,2,3,4,5]),[[1,2],[4,5]]).tolist(),[False,True,False,False,True,False])
    def test_visual_rejects_failed_low_confidence_sentinel_and_future(self):
        f=pd.DataFrame({'timestamp':[0,1,2,3,4,5],'confidence':[1,.7,1,1,1,1],'success':[1,1,0,1,1,1],'AU04_r':[1,2,3,-100,5,6]})
        self.assertEqual(visual_values(f,['AU04_r'],[[0,5]]).ravel().tolist(),[1,5])
    def test_reference_uses_only_earlier_rapport(self):
        rows=[dict(start=0,stop=10,domain='none',question_category='rapport'),dict(start=12,stop=15,domain='sleep',question_category='sleep'),dict(start=16,stop=60,domain='none',question_category='rapport'),dict(start=1,stop=8,domain='none',question_category='rapport',scrubbed=True)]
        rows.append(dict(start=2,stop=7,domain='none',question_category='rapport',quality={'overlap':True}))
        self.assertEqual(baseline_rows(rows),rows[:1])
        self.assertEqual(baseline_rows([rows[0]]),[])
    def test_overlapping_spans_do_not_double_reference_duration(self):
        self.assertEqual(union_spans([[0,10],[5,15],[20,25]]),[[0.,15.],[20.,25.]])
    def test_all_comparisons_same_rows_and_delta(self):
        rows=[dict(word_count=5,duration=2,position=.5,question_category='sleep')]
        feat=[{'features':dict.fromkeys(FEATURES,3),'baseline':dict.fromkeys(FEATURES,2)}]
        x=matrices(rows,feat)
        self.assertTrue(all(v.shape[0]==1 for v in x.values()))
        np.testing.assert_equal(x['delta'][0,4:],1)
    def test_video_end_never_extrapolated_and_scrubbed_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'402_P.zip';wav_bytes=io.BytesIO()
            with wave.open(wav_bytes,'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(1000);w.writeframes((np.sin(np.arange(40000)) * 1000).astype('<i2').tobytes())
            with zipfile.ZipFile(path,'w') as z:
                z.writestr('402_AUDIO.wav',wav_bytes.getvalue())
                z.writestr('402_COVAREP.csv','100,1\n'*4000)
                for suffix,cols in VISUAL.items():
                    frame=pd.DataFrame({'timestamp':np.arange(600)/30,'confidence':1,'success':1,**dict.fromkeys(cols,1)})
                    z.writestr('402_'+suffix,frame.to_csv(index=False))
            rows=[dict(qa_id='b',start=0,stop=30,participant_spans=[[0,30]],domain='none',question_category='rapport'),dict(qa_id='late',start=31,stop=35,participant_spans=[[31,35]],domain='sleep',question_category='sleep'),dict(qa_id='scrub',start=2,stop=4,participant_spans=[[2,4]],domain='interest',question_category='interest',scrubbed=True)]
            rows.extend([dict(qa_id='partial_visual',start=19,stop=24,participant_spans=[[19,24]],domain='sleep',question_category='sleep'),dict(qa_id='partial_audio',start=39,stop=44,participant_spans=[[39,44]],domain='sleep',question_category='sleep')])
            got=extract_participant(path,402,rows)
            self.assertFalse(got[0]['visual_ok']);self.assertTrue(got[0]['audio_ok'])
            self.assertFalse(got[1]['audio_ok']);self.assertFalse(got[1]['visual_ok'])
            self.assertFalse(got[2]['visual_ok']);self.assertLess(got[2]['visual_coverage_fractions']['CLNF_AUs.txt'],.8)
            self.assertFalse(got[3]['audio_ok']);self.assertLess(got[3]['audio_coverage_fraction'],.8)
if __name__=='__main__':unittest.main()
