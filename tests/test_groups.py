from relgauge.binarycompositegaugespectrumaudit import _close_group, _perm_order

def test_s3_from_overlapping_transpositions():
    # labels 0,1,2; generators (0 1), (0 2)
    s01=(1,0,2)
    s02=(2,1,0)
    G,trunc=_close_group([s01,s02], max_group_order=100)
    assert not trunc
    assert len(G)==6
    assert sorted(_perm_order(g) for g in G)==[1,2,2,2,3,3]
