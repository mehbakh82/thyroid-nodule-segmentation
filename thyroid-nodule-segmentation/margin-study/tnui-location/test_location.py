import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pilot import automatic_center,centered_box,localized_predictions,native_prediction
from evaluate import measure
from border_check import scan_bounds


class LocationGeometry(unittest.TestCase):
    def test_dark_interior_is_kept_and_isolated_text_is_ignored(self):
        image=np.zeros((80,80),dtype='uint8')
        image[2,20:60]=255
        image[11:65,5:75]=90
        image[21:30,:]=0
        self.assertEqual(scan_bounds(image),(11,65))
        self.assertEqual(scan_bounds(image*0),(0,80))

    def test_same_size_even_at_native_edges(self):
        for xy in ((0,0),(18,12),(36,24)):
            l,t,r,b=centered_box((37,25),xy)
            self.assertEqual((r-l,b-t),(12,12))
            self.assertTrue(0<=l<r<=37 and 0<=t<b<=25)

    def test_largest_component_and_fallback(self):
        p=np.zeros((13,21),bool)
        self.assertEqual(automatic_center(p),((10.,6.),True))
        p[2:5,5:8]=True;p[10,20]=True
        self.assertEqual(automatic_center(p),((6.,3.),False))

    def test_projection_and_clipping(self):
        im=Image.new('L',(25,17))
        mask=np.zeros((17,25),bool);mask[1:4,1:4]=True
        with patch('pilot.predict',return_value=torch.ones(1,1,8,8)):
            pred=native_prediction(None,im,(4,3,12,11))
            self.assertEqual(pred.shape,mask.shape)
            self.assertTrue(pred[3:11,4:12].all())
            self.assertEqual(pred.sum(),64)
            preds,cover,_=localized_predictions(None,im,mask)
            self.assertEqual(cover['oracle'],1.)
            np.testing.assert_array_equal(preds['clipped_whole'],preds['oracle'])
            self.assertGreater(measure(preds['center'],mask)['fn'],0)


if __name__=='__main__':unittest.main()
