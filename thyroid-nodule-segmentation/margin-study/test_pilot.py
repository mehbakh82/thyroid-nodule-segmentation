"""Known-answer checks for edit support, abstention and stable-pixel accounting."""
import unittest

import numpy as np

from pilot import apply_heads,classical,describe,evaluate,signed_band


class MarginPilotTests(unittest.TestCase):
    def test_far_field_is_immutable(self):
        y,x=np.indices((48,48))
        original=(x-24)**2+(y-24)**2<=9**2
        prob=original.astype('f4')*.8+.1
        image=(original*200).astype('uint8')
        band,features=describe(image,prob,np.zeros_like(prob))
        self.assertEqual(features.shape,(band.sum(),10))
        self.assertFalse(band[24,24]);self.assertFalse(band[0,0])
        for vote in (0.,1.):
            head=np.full(band.sum(),vote)
            for candidate in apply_heads(original,band,head,head).values():
                np.testing.assert_array_equal(candidate[~band],original[~band])
        out=classical(image,original,band)
        np.testing.assert_array_equal(out[~band],original[~band])

    def test_disagreement_abstains_and_thresholds(self):
        mask=np.array([[False,True,False,True,False,True]])
        band=np.ones_like(mask)
        a=np.array([.7,.3,.99,.01,.65,.35])
        b=np.array([.8,.29,.40,.70,.78,.20])
        out=apply_heads(mask,band,a,b)['selective']
        np.testing.assert_array_equal(out,[[True,False,False,True,False,True]])

    def test_empty_and_full_masks_cannot_be_edited(self):
        for value in (True,False):
            base=np.full((20,20),value)
            band,_=signed_band(base)
            self.assertFalse(band.any())
            for result in apply_heads(base,band,np.array([]),np.array([])).values():
                np.testing.assert_array_equal(result,base)

    def test_stable_damage_is_distinct_from_ambiguous_edits(self):
        base=np.zeros((10,10),bool);base[2:5,2:5]=True
        primary=base.copy(); primary[5,3]=True; primary[3,5]=True
        alternate=base.copy(); alternate[5,3]=True
        new=base.copy();new[5,3]=True;new[3,5]=True;new[2,3]=False
        s=evaluate(new,base,[primary,alternate])
        self.assertEqual([s[k] for k in ('fixed','damaged','ambiguous_edits','edits')],[1,1,1,3])


if __name__=='__main__':unittest.main()
