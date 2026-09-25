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
        
        import src.helpers as hp
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
            inv_3d = np.broadcast_to(inv_2d, (len(cutter_pair_index), Nvz, Nmu)).astype(np.int64)
        else:
            inv_3d = np.ones((len(cutter_pair_index), Nvz, Nmu), dtype=np.int64)

        disorder_seed = 0
        import re
        m = re.search(r"disorder_realization_(\d+)", self.data_dir.name)
        if m:
            disorder_seed = int(m.group(1))

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
                disorder_seed=disorder_seed,
                geometry_seed=0,
            ),
        )

        return ds

    def to_netcdf(self, filename: str) -> None:
        """
        Exports the dataset to a .nc file that perfectly matches the Microsoft 
        simulated data structure.
        """
        ds = self.to_xarray()
        ds = ds.transpose("cutter_pair_index", "V", "B", "bias", missing_dims="ignore")
        
        n_cutters = ds.sizes.get("cutter_pair_index", 5)
        
        barrier_path = self.data_dir / "tgp_barrier_arr.npy"
        if barrier_path.exists():
            v_cutters = np.load(barrier_path).astype(float)
        else:
            v_cutters = np.zeros(n_cutters)
            
        ds.coords["V_leftcutter"] = (["cutter_pair_index"], v_cutters)
        ds.coords["V_rightcutter"] = (["cutter_pair_index"], v_cutters)

        
        ds.to_netcdf(filename, engine="h5netcdf")
