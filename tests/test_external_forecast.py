"""External cohort eligibility and safe raw-photometry parsing."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from external_forecast import eligible,valid_rows


class ExternalForecastTests(unittest.TestCase):
    def test_invalid_detection_values_are_excluded(self):
        good=dict(mjd=60700,fid=1,magpsf=18.,sigmapsf=.1)
        items=[good,dict(good,sigmapsf=-1),dict(good,fid=4),dict(good,magpsf=None),dict(good,mjd=float('nan'))]
        np.testing.assert_array_equal(valid_rows(dict(detections=items)),[[60700,1,18,.1]])

    def test_maturity_and_full_curve_quality_are_required(self):
        start=pd.Timestamp('2025-01-01').to_julian_date()-2400000.5
        rows=np.column_stack([start+np.arange(8)*10,np.tile([1,2],4),np.full(8,18.),np.full(8,.1)])
        self.assertEqual(eligible(rows,start+90),'eligible')
        self.assertEqual(eligible(rows,start+89.99),'less_than_90_days_since_first_detection')
        self.assertEqual(eligible(rows[:7],start+100),'fails_original_quality')
        rows[:,1]=1
        self.assertEqual(eligible(rows,start+100),'fails_original_quality')


if __name__=='__main__':unittest.main()
