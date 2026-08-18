import os
import numpy as np
import xarray as xr
from pathlib import Path

class TGPAdapter:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def to_xarray(self) -> xr.Dataset:
        params_path = self.data_dir / "all_params.npz"
        if not params_path.exists():
            raise FileNotFoundError(f"Missing parameter file: {params_path}")
            
        params = np.load(params_path, allow_pickle=True)
        mu_var = params['mu_var']
        Vz_var = params['Vz_var']
        
        Nmu = len(mu_var)
        Nvz = len(Vz_var)

        g_ll_flat = np.load(self.data_dir / "tgp_stage1_dIdVl.npy")
        g_rr_flat = np.load(self.data_dir / "tgp_stage1_dIdVr.npy")
        g_lr_flat = np.load(self.data_dir / "tgp_stage1_dIdV_LR.npy")
        g_rl_flat = np.load(self.data_dir / "tgp_stage1_dIdV_RL.npy")
        
        import helpers as hp
        bias = hp.make_dynamic_grid()

        b_field = Vz_var
        mu = mu_var
        cutter_pair_index = np.arange(5)

        reshape_dims = (Nmu, Nvz, len(cutter_pair_index), len(bias))
        
        transpose_order = (2, 1, 0, 3)

        g_ll = g_ll_flat.reshape(reshape_dims).transpose(transpose_order)
        g_rr = g_rr_flat.reshape(reshape_dims).transpose(transpose_order)
        g_lr = g_lr_flat.reshape(reshape_dims).transpose(transpose_order)
        g_rl = g_rl_flat.reshape(reshape_dims).transpose(transpose_order)

        pdi_data_path = self.data_dir / "pdi_data.npy"
        if pdi_data_path.exists():
            pdi_data = np.load(pdi_data_path)
            if pdi_data.shape[1] >= 4:
                topological_flat = pdi_data[:, 3]
            else:
                topological_flat = pdi_data[:, 2]
                
            inv_2d = topological_flat.reshape((Nmu, Nvz)).transpose(1, 0)
            inv_3d = np.broadcast_to(inv_2d, (len(cutter_pair_index), Nvz, Nmu))
        else:
            inv_3d = np.ones((len(cutter_pair_index), Nvz, Nmu))

        ds = xr.Dataset(
            data_vars=dict(
                g_ll=(["cutter_pair_index", "B", "V", "bias"], g_ll),
                g_rr=(["cutter_pair_index", "B", "V", "bias"], g_rr),
                g_lr=(["cutter_pair_index", "B", "V", "bias"], g_lr),
                g_rl=(["cutter_pair_index", "B", "V", "bias"], g_rl),
                L_SI=(["cutter_pair_index", "B", "V"], inv_3d),
                R_SI=(["cutter_pair_index", "B", "V"], inv_3d),
            ),
            coords=dict(
                cutter_pair_index=(["cutter_pair_index"], cutter_pair_index),
                B=(["B"], b_field),
                V=(["V"], mu),
                bias=(["bias"], bias),
            ),
            attrs=dict(
                sample_name="simulated_1D_nanowire",
                surface_charge=0.0,
                disorder_seed=0,
            ),
        )

        return ds
