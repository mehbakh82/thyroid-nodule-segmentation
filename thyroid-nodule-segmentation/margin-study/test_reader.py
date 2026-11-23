import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from score_tracings import compare, load_return, reader_mask, summarize


class TracingReturnTests(unittest.TestCase):
    def test_decoders_on_reference_and_frozen_grid(self):
        raw=np.zeros((16,20),dtype='uint8')
        raw[2:14,3:17]=180
        raw[3:13,4:16]=255
        rd=raw>127
        r=compare(raw,rd,rd)
        self.assertEqual(r['native_agreement']['above127']['dice'],1)
        self.assertLess(r['native_agreement']['only255']['relative_area_error'],0)
        self.assertEqual(r['evaluation_grid']['reader']['region_dice'],1)
        self.assertLess(r['evaluation_grid']['reader']['shrunk_region_dice'],1)
        r.update(status='traced',stratum='small')
        totals=summarize([r,dict(status='untraceable')])
        self.assertEqual(totals['dispositions'],{'traced':1,'untraceable':1})
        self.assertEqual(totals['mean_native_dice']['above127'],1)

    def test_bad_encoding_or_native_shape(self):
        with TemporaryDirectory() as tmp:
            f=Path(tmp)/'mask.png'
            a=np.zeros((10,12),dtype='uint8');a[2:7,3:8]=255
            Image.fromarray(a).save(f)
            self.assertEqual(reader_mask(f,(12,10)).sum(),25)
            with self.assertRaises(ValueError):reader_mask(f,(10,12))
            a[5,5]=128;Image.fromarray(a).save(f)
            with self.assertRaises(ValueError):reader_mask(f,(12,10))

    def test_missing_mask_and_unexplained_case(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def sheet(case,status):
                with (folder/'observations.csv').open('w') as stream:
                    writer=csv.writer(stream)
                    writer.writerow(['case','status','margin_visibility','measurement_marks'])
                    writer.writerow([case,status,'',''])
            sheet('case-01','traced')
            with self.assertRaises(ValueError):load_return(folder,['case-01'])
            sheet('case-01','target_unclear')
            rows,masks=load_return(folder,['case-01'])
            self.assertEqual(summarize(rows)['dispositions'],{'target_unclear':1})
            self.assertFalse(masks)
            sheet('case-02','pending')
            with self.assertRaises(ValueError):load_return(folder,['case-01'])
