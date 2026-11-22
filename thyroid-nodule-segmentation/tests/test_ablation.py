import unittest
import numpy as np
from ablation_report import contrasts


class FactorialDesign(unittest.TestCase):
    def test_interaction_and_marginal_effects(self):
        # The first image has additive inputs. On the second, the joint gain is suppressed.
        scores=np.array([[.50,.60,.65,.75],[.60,.65,.65,.55]])
        c=contrasts(scores)
        np.testing.assert_allclose(c['interaction'],[0,-.15],atol=1e-12)
        np.testing.assert_allclose(c['contrast_main_effect'],[.10,-.025])
        np.testing.assert_allclose(c['gradient_main_effect'],[.15,-.025])
        np.testing.assert_allclose(c['combined_minus_gray'],[.25,-.05])
