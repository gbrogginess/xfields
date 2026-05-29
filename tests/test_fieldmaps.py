# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xfields as xf


def test_trilinear_interpolated_fieldmap_repr():
    fieldmap = xf.TriLinearInterpolatedFieldMap(
        x_range=(-1, 1),
        y_range=(-2, 2),
        z_range=(-3, 3),
        nx=2,
        ny=2,
        nz=2,
    )

    repr_fieldmap = repr(fieldmap)

    assert fieldmap.x_min == -1
    assert fieldmap.y_min == -2
    assert fieldmap.z_min == -3
    assert "x_min=-1" in repr_fieldmap
    assert "y_min=-2" in repr_fieldmap
    assert "z_min=-3" in repr_fieldmap
