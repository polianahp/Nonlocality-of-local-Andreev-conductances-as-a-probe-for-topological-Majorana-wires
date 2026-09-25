import numpy as np
import xarray as xr
dl_ref = xr.load_dataset("Tests/Microsoft_Comparison/verification_data/reference/ds_left_ref_0.nc")
dl_usr = xr.load_dataset("Tests/Microsoft_Comparison/verification_data/user/ds_left_user_0.nc")

zbp_ref = xr.load_dataset("Tests/Microsoft_Comparison/verification_data/reference/zbp_ds_ref_0.nc")
zbp_usr = xr.load_dataset("Tests/Microsoft_Comparison/verification_data/user/zbp_ds_user_0.nc")

print("ZBP diff sum:", np.sum(zbp_ref['zbp'].values ^ zbp_usr['zbp'].values))
