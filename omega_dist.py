import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def sun_earth_corotating_to_geo_eclip_batch_full(states_corotating, earth_states):
    """
    Converts (6,N) SECR states to (6,N) geocentric ECLIPJ2000.
    """
    _, N = states_corotating.shape

    # Earth heliocentric in ECLIPJ2000 (used for rotation & Omega)
    h_r_E = earth_states[:3, :].T  # (N,3)
    h_v_E = earth_states[3:, :].T  # (N,3)

    # SECR‐frame L1 states
    E_r_o_prime = states_corotating[:3, :].T  # (N,3)
    E_v_o_prime = states_corotating[3:, :].T  # (N,3)

    # Rotation angle from Earth→Sun vector (negated for co‑rot.→inertial)
    angles = np.arctan2(-h_r_E[:, 1], -h_r_E[:, 0])  # (N,)

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # build (N,3,3) rotation matrices about Z
    R = np.zeros((N, 3, 3))
    R[:, 0, 0], R[:, 0, 1] = cos_a, -sin_a
    R[:, 1, 0], R[:, 1, 1] = sin_a, cos_a
    R[:, 2, 2] = 1.0

    # rotate positions
    geo_r_o = np.einsum('nij,nj->ni', R, E_r_o_prime)  # (N,3)

    # compute Earth’s spin rate around Sun for Coriolis
    omega_mag = np.linalg.norm(np.cross(h_r_E, h_v_E), axis=1) / (np.linalg.norm(h_r_E, axis=1) ** 2)
    h_omega = np.zeros((N, 3));
    h_omega[:, 2] = omega_mag

    # rotate that into SECR, add Coriolis, rotate back
    E_omega = np.einsum('nij,nj->ni', R, h_omega)  # (N,3)
    v_cori = np.cross(E_omega, E_r_o_prime)  # (N,3)
    v_rel = E_v_o_prime + v_cori  # (N,3)
    geo_v_o = np.einsum('nij,nj->ni', R, v_rel)  # (N,3)

    out = np.zeros_like(states_corotating)
    out[:3, :] = geo_r_o.T
    out[3:, :] = geo_v_o.T
    return out


population_dir = '/media/aeromec/Seagate Desktop Drive/minimoon_files_oorb'
population_file = 'minimoon_master_with_L1_geo_omega.csv'
population_file_path = population_dir + '/' + population_file
master_data = pd.read_csv(population_file_path, sep=' ')

# Prepare empty arrays to collect new L1‐geo fields
N = master_data.shape[0]
L1_geo = np.zeros((N, 6))  # columns: x, y, z, vx, vy, vz
rel_geo = np.zeros((N, 6))  # for the first 100: dX,dY,dZ, dVx,dVy,dVz
sph = np.zeros((N, 4))  # ra,dec, ra_dot, dec_dot
omega = np.zeros(N)  # apparent motion ω
master_data["omega_arcsecph_minV_earth"] = np.nan

# ----------------------------------------------------------------------
# Loop over each minimoon, pull Earth‐heliocentric & fill SECR L1 state
# ----------------------------------------------------------------------
# for i, row in tqdm(master_data.iterrows(), total=N, desc="L1→geocentric"):
for count, (i, row) in enumerate(
        tqdm(master_data.iterrows(), total=N, desc="Test Earth→ω")):

    # if count >= 1000:
    #     break
    oid = row['Object id']
    idx = row['Min_Earth_V_index']
    if np.isnan(idx):
        continue
    idx = int(idx)

    csv_path = os.path.join(population_dir, f"{oid}.csv")
    if not os.path.isfile(csv_path):
        continue

    # Read the per‐object CSV; must include Earth(Helio) & asteroid geo/vgeo
    df = pd.read_csv(csv_path, sep=' ')
    if idx < 0 or idx + 1 >= len(df):
        continue

    # 3a) Extract Earth's heliocentric at that row
    h_r = df.iloc[idx][["Earth x (Helio)", "Earth y (Helio)", "Earth z (Helio)"]].astype(float).values
    h_v = df.iloc[idx][["Earth vx (Helio)", "Earth vy (Helio)", "Earth vz (Helio)"]].astype(float).values
    earth_states = np.hstack([h_r, h_v]).reshape(6, 1)  # (6,1) array

    # 3b) Build SECR‐frame L1 state: (+0.01,0,0) AU, zero vel
    d_L1 = 0.00  # AU
    secr_L1 = np.zeros((6, 1))
    secr_L1[0, 0] = + d_L1

    # 3c) Convert to geocentric ECLIPJ2000
    l1_geo6 = sun_earth_corotating_to_geo_eclip_batch_full(secr_L1, earth_states)  # (6,1)
    # L1_geo[i,:] = l1_geo6.flatten()
    L1_geo[count, :] = l1_geo6.flatten()

    # ────────────────────────────────────────────────────────────────────
    # 3d) NEW: read asteroid's geocentric state at that same idx
    ast_r = df.iloc[idx][["Geo x", "Geo y", "Geo z"]].astype(float).values
    ast_v = df.iloc[idx][["Geo vx", "Geo vy", "Geo vz"]].astype(float).values

    # 3e) compute relative
    dr = ast_r - L1_geo[count, :3]
    dv = ast_v - L1_geo[count, 3:]

    rel_geo[count, :3] = dr
    rel_geo[count, 3:] = dv
    # ────────────────────────────────────────────────────────────────────

    # ——————————————————————————————————————————
    # 4) Spherical + time‐derivatives
    x, y, z = dr
    vx, vy, vz = dv
    rho = np.hypot(x, y)  # √(x²+y²)
    r2 = x * x + y * y + z * z  # r²

    # RA, DEC
    ra = np.arctan2(y, x)  # radians
    dec = np.arctan2(z, rho)  # radians

    # dRA = (x*vy - y*vx)/(x²+y²)
    ra_dot = (x * vy - y * vx) / (x * x + y * y)

    # dDEC = [ vz*ρ  -  z*(x*vx+y*vy)/ρ ] / r²
    dec_dot = (vz * rho - z * (x * vx + y * vy) / rho) / r2

    sph[count] = [ra, dec, ra_dot, dec_dot]

    # 5) apparent motion ω = √((dRA·cos dec)² + dec_dot²)  [rad/day]
    omega[count] = np.hypot(ra_dot * np.cos(dec), dec_dot)

    master_data.at[i, "omega_arcsecph_minV_earth"] = omega[count] * (206265.0 / 24.0)

# ----------------------------------------------------------------------
# 4) Attach results to master_data & save
# ----------------------------------------------------------------------
# master_data[['L1_geo_x', 'L1_geo_y', 'L1_geo_z',
#              'L1_geo_vx', 'L1_geo_vy', 'L1_geo_vz']] = L1_geo

outpath = os.path.join(population_dir, 'minimoon_master_with_L1_geo_omega_w_earth.csv')
master_data.to_csv(outpath, sep=' ', index=False)
print("Saved augmented master to:", outpath)

# ----------------------------------------------------------------------
# Print the first 10 results for inspection
# ----------------------------------------------------------------------
rel_df = pd.DataFrame(
    rel_geo[:count + 1],
    columns=["dX", "dY", "dZ", "dVx", "dVy", "dVz"]
)
sph_df = pd.DataFrame(
    sph[:count + 1],
    columns=["ra_rad", "dec_rad", "ra_dot_radpd", "dec_dot_radpd"]
)
om_df = pd.DataFrame(
    omega[:count + 1],
    columns=["omega_radpd"]
)
om_df["omega_arcsecph_minV_earth"] = om_df["omega_radpd"] * 206265.0 / 24.0

print("\nFirst 10 relative states:")
print(rel_df.iloc[:10])
print("\nFirst 10 spherical rates:")
print(sph_df.iloc[:10])
print("\nFirst 10 ω (arcsec/hour):")
print(om_df.iloc[:10])
