from belief import Label, label_config_sets
from cog_belief import CostBelief
import numpy as np

def test_label_config_sets():
    writer_configs = np.array([[1,2]])
    reader_configs = np.array([[1,2]])
    assert label_config_sets(writer_configs, reader_configs) == Label.SPECIFIC

    writer_configs = np.array([[1,2], [3,4]])
    reader_configs = np.array([[1,2]])
    assert label_config_sets(writer_configs, reader_configs) == Label.COARSE

    writer_configs = np.array([[1,2]])
    reader_configs = np.array([])
    assert label_config_sets(writer_configs, reader_configs) == Label.UNRESOLVABLE

    writer_configs = np.array([[1,2]])
    reader_configs = np.array([[3,4]])
    assert label_config_sets(writer_configs, reader_configs) == Label.ERROR

def test_cog_belief():
    prior = np.array([0.01453737, 0.01377215, 0.01377215, 0.00845222, 0.01377215,
       0.00845222, 0.00515188, 0.00758852, 0.01377215, 0.00515188,
       0.00185154, 0.0038144 , 0.00680205, 0.00664499, 0.0038144 ,
       0.00417203, 0.01377215, 0.00350171, 0.00185154, 0.00570146,
       0.00845222, 0.00758852, 0.00475793, 0.00556271, 0.00845222,
       0.00664499, 0.00475793, 0.00417203, 0.00758852, 0.00625805,
       0.00486737, 0.00402087, 0.01377215, 0.00680205, 0.00845222,
       0.00758852, 0.00515188, 0.00570146, 0.00664499, 0.00625805,
       0.00020136, 0.00287086, 0.0038144 , 0.00417203, 0.00287086,
       0.0034767 , 0.0034767 , 0.00344646, 0.00185154, 0.0038144 ,
       0.00475793, 0.09248011, 0.0038144 , 0.00417203, 0.00417203,
       0.08731041, 0.0038144 , 0.0034767 , 0.00208602, 0.04365521,
       0.00417203, 0.00344646, 0.00229764, 0.03861807, 0.01377215,
       0.00515188, 0.00350171, 0.00475793, 0.00680205, 0.00570146,
       0.00475793, 0.00486737, 0.00350171, 0.0038144 , 0.00192733,
       0.00208602, 0.00475793, 0.00486737, 0.00278136, 0.00287205,
       0.00845222, 0.00664499, 0.00570146, 0.00486737, 0.00758852,
       0.00625805, 0.00556271, 0.00459528, 0.00570146, 0.00486737,
       0.0034767 , 0.00287205, 0.00625805, 0.00459528, 0.00402087,
       0.00254066, 0.00350171, 0.00475793, 0.00570146, 0.00556271,
       0.0038144 , 0.00417203, 0.00486737, 0.00459528, 0.0009838 ,
       0.00208602, 0.00278136, 0.00287205, 0.00208602, 0.00229764,
       0.00229764, 0.00203253, 0.0038144 , 0.0034767 , 0.0034767 ,
       0.06548281, 0.00417203, 0.00344646, 0.00287205, 0.04827258,
       0.00278136, 0.00229764, 0.00114882, 0.01930903, 0.00344646,
       0.00203253, 0.0015244 , 0.        ])
    context = np.array([[-0.565     ,  0.18      , -0.33333333, -0.02666667], 
       [ 0.475     ,  0.345     , -0.66666667,  0.10666667], 
       [-0.575     , -0.355     ,  0.        , -0.78666667], 
       [-0.46      , -0.805     , -0.66666667,  0.90666667], 
       [-0.13      , -0.33      , -1.        ,  0.28      ], 
       [ 0.42      ,  0.15      ,  0.        ,  0.48      ], 
       [ 0.24      , -0.195     ,  0.66666667, -0.89333333]])

    belief = CostBelief(
        7, context,
        absolute = True,
        num_size_buckets = 5,
        num_color_buckets = 5,
        use_diameter = False,
        use_contiguity = False,
    )

    EdHs = belief.compute_EdHs(prior)
    cs, hs = belief.viz_belief(EdHs, 8)
    print(cs)

    utt = np.array([1,1,0,0,0,0,0])
    print(belief.spatial_deny(utt, context))
    utt = np.array([1,0,1,0,0,0,0])
    print(belief.spatial_deny(utt, context))
    utt = np.array([1,0,0,1,0,0,0])
    print(belief.spatial_deny(utt, context))
    utt = np.array([1,0,0,0,1,0,0])
    print(belief.spatial_deny(utt, context))

    print("4dot")
    utt = np.array([0,1,1,0,0,1,1])
    print(belief.spatial_deny(utt, context))


if __name__ == "__main__":
    test_label_config_sets()
    test_cog_belief()
