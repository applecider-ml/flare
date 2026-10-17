"""Boundary tests for the supervised research comparison."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import improve_supervised as api


class SupervisedTests(unittest.TestCase):
    def events(self):
        a=np.zeros((12,15));a[:,0]=np.arange(12)*2.;a[:,2]=np.arange(12)%2
        a[:,3]=2+np.arange(12)*.03;a[:,4]=.02
        return a

    def test_future_flux_and_labels_cannot_change_recency(self):
        a=self.events();before=api.recency(a,10)
        a[a[:,0]>10,3]+=100;a[:,8:]=999
        np.testing.assert_array_equal(before,api.recency(a,10))

    def test_anchor_uses_first_at_tied_latest_time(self):
        a=self.events()[:6];duplicate=a[-1].copy();duplicate[3]+=2
        expected=23.9-2.5*a[-1,3]-(23.9-2.5*duplicate[3])
        np.testing.assert_allclose(api.recency(np.vstack([a,duplicate]),10)[1,0],expected)

    def test_views_preserve_original_and_never_create_external_values(self):
        cols=sum(api.BLOCKS,[]);x=pd.DataFrame(np.ones((10,len(cols))),columns=cols)
        x['lc_feature']=np.arange(10);x.loc[2,cols[0]]=np.nan
        out=api.views(x,17)
        pd.testing.assert_frame_equal(out.iloc[:10],x)
        for i in range(4):
            v=out.iloc[i*10:(i+1)*10].reset_index(drop=True)
            np.testing.assert_array_equal(v.lc_feature,x.lc_feature)
            self.assertTrue(np.isnan(v.loc[2,cols[0]]))

    def test_each_object_has_equal_total_training_weight(self):
        ids=np.array([10,20,20,20,30,30]);w=api.object_weights(ids)
        np.testing.assert_allclose([w[ids==i].sum() for i in [10,20,30]],[2,2,2])

    def test_real_targets_respect_horizon_band_and_available_anchor(self):
        x=pd.DataFrame({'feature':[7.]});last=np.array([[1.,np.nan]])
        points=[np.array([[9,0,1],[10,0,1],[11,0,2],[12,1,4],[91,0,5]])]
        table,target,ids,bands=api.real_long(x,points,last,[0],10)
        self.assertEqual(table.fc_day.tolist(),[11.])
        np.testing.assert_array_equal(target,[2.]);np.testing.assert_array_equal(ids,[0])

    def test_grid_layout_and_recency_alignment(self):
        x=pd.DataFrame({'feature':[0.,1.]});t=np.arange(84).reshape(2,2,21)
        table,truth,ids,bands=api.grid_long(x,t,np.array([1,0]),20)
        np.testing.assert_array_equal(truth[:2],t[[1,0],0,5])
        s=np.arange(16).reshape(2,2,4)
        result=api.add_recency(table,ids,bands,s)
        np.testing.assert_array_equal(result.fc_last_mag,s[ids,bands,0])


if __name__=='__main__':unittest.main()
