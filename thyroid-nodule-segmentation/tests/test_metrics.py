import unittest
import numpy as np
from evaluate import measure,summarize


class Scoring(unittest.TestCase):
    def test_perfect_empty_and_shift(self):
        truth=np.zeros((8,8),bool);truth[2:5,2:5]=True
        same=measure(truth,truth)
        self.assertEqual([same[m] for m in ['dice','iou','surface_dice','hd95']],[1,1,1,0])
        miss=measure(np.zeros_like(truth),truth)
        self.assertTrue(miss['empty']);self.assertTrue(miss['miss'])
        self.assertEqual(miss['hd95'],np.hypot(8,8))
        shifted=measure(np.roll(truth,1,axis=1),truth)
        self.assertAlmostEqual(shifted['dice'],2/3)
        self.assertEqual(shifted['fp'],3);self.assertEqual(shifted['fn'],3)
        self.assertEqual(shifted['surface_dice'],1)
        rows=[dict(gray=miss,edge=same,rescue=True,harm=False,delta=1.)]
        self.assertEqual(summarize(rows)['conditional_image_interval'],[1,1])
        self.assertEqual(summarize(rows)['rescue'],1)
