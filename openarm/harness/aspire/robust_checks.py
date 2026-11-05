"""Lower-half contact invariants; no real hardware."""
import unittest
from .robust import lower_half


def contacts(*fractions):
    return {'contacts':[{'geom':'bottle_body','body_fraction':v} for v in fractions]}


class ContactRegionChecks(unittest.TestCase):
    def test_mean_below_half_does_not_accept_an_upper_contact(self):
        self.assertFalse(lower_half(contacts(.3,.6)))

    def test_lower_body_boundary(self):
        self.assertTrue(lower_half(contacts(.3,.5)))
        self.assertFalse(lower_half(contacts(.3,.500001)))

    def test_missing_nonfinite_and_outside_geometry_rejected(self):
        for c in [{},contacts(float('nan')),contacts(-.01),contacts(1.1),
                  {'contacts':[{'geom':'bottle_neck','body_fraction':.4}]}]:
            self.assertFalse(lower_half(c))


if __name__=='__main__':unittest.main()
