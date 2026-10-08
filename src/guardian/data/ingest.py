"""Data ingestion module for NASA PCoE battery aging dataset.

Recursively locates .mat files in data/raw/ and parses charge/discharge cycles
into standardized telemetry DataFrames according to Section 5.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.io

from guardian.utils.logging import setup_logger

logger = setup_logger("guardian.data.ingest")


def find_mat_files(raw_dir: str | Path = "data/raw", target_batteries: Optional[List[str]] = None) -> List[Path]:
    """Recursively search raw_dir for NASA PCoE battery .mat files.

    Args:
        raw_dir: Root directory to search recursively.
        target_batteries: Optional list of battery IDs (e.g. ['B0005', 'B0006']).
                         If None, returns all found .mat files.

    Returns:
        List of Path objects for discovered .mat files.
    """
    path = Path(raw_dir)
    if not path.exists():
        return []

    discovered = sorted(list(path.rglob("*.mat")))
    if not target_batteries:
        return discovered

    target_set = {b.upper() for b in target_batteries}
    filtered: List[Path] = []
    seen: set[str] = set()
    for f in discovered:
        stem = f.stem.upper()
        if stem in target_set and stem not in seen:
            filtered.append(f)
            seen.add(stem)

    return filtered


def _parse_cycle_timestamp(c_time_arr: np.ndarray, default_year: int = 2008) -> datetime:
    """Parse cycle base timestamp from MATLAB date vector [Y, M, D, H, M, S]."""
    try:
        vec = c_time_arr.flatten()
        if len(vec) >= 6:
            year = int(vec[0])
            month = int(vec[1])
            day = int(vec[2])
            hour = int(vec[3])
            minute = int(vec[4])
            second = int(vec[5])
            # Bound check valid dates
            if 1990 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime(default_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def ingest_battery_mat(mat_path: str | Path, default_ambient: float = 24.0) -> pd.DataFrame:
    """Parse a single NASA battery .mat file into raw tabular format.

    Args:
        mat_path: Path to the .mat file.
        default_ambient: Default ambient temperature in °C.

    Returns:
        DataFrame containing cycle records for the battery.
    """
    path = Path(mat_path)
    mat = scipy.io.loadmat(str(path))
    # Identify battery struct key (usually file stem, e.g. 'B0005')
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No valid data struct found in {mat_path}")
    root_key = keys[0]
    battery_id = path.stem.upper()

    cycles_arr = mat[root_key][0, 0]["cycle"]
    num_cycles = cycles_arr.shape[1]
    logger.info(f"Ingesting {battery_id} ({num_cycles} raw cycles)...")

    # Step 1: Pre-extract capacities to interpolate across cycles
    cycle_capacities: Dict[int, float] = {}
    for idx in range(num_cycles):
        cycle_obj = cycles_arr[0, idx]
        c_type = str(cycle_obj["type"][0]).lower()
        if c_type == "discharge":
            data_obj = cycle_obj["data"]
            if "Capacity" in data_obj.dtype.names:
                cap_val = data_obj["Capacity"][0, 0].flatten()
                if len(cap_val) > 0 and float(cap_val[0]) > 0:
                    cycle_capacities[idx + 1] = float(cap_val[0])

    # Initial battery capacity (first discharge capacity or default 2.0 Ah)
    first_cap = next(iter(cycle_capacities.values()), 2.0)
    nominal_capacity = first_cap if first_cap > 0 else 2.0

    # Step 2: Extract telemetry rows
    records: List[Dict[str, Any]] = []

    last_known_cap = nominal_capacity

    for idx in range(num_cycles):
        cycle_num = idx + 1
        cycle_obj = cycles_arr[0, idx]
        c_type = str(cycle_obj["type"][0]).lower()

        # Only process charge and discharge cycles (skip impedance)
        if c_type not in ("charge", "discharge"):
            continue

        try:
            ambient_temp = float(cycle_obj["ambient_temperature"][0, 0])
        except Exception:
            ambient_temp = default_ambient

        base_dt = _parse_cycle_timestamp(cycle_obj["time"][0])
        data_obj = cycle_obj["data"]
        fields = data_obj.dtype.names

        if "Voltage_measured" not in fields or "Current_measured" not in fields or "Time" not in fields:
            continue

        v_arr = data_obj["Voltage_measured"][0, 0].flatten()
        i_arr = data_obj["Current_measured"][0, 0].flatten()
        t_arr = data_obj["Time"][0, 0].flatten()

        if "Temperature_measured" in fields:
            temp_arr = data_obj["Temperature_measured"][0, 0].flatten()
        else:
            temp_arr = np.full_like(v_arr, ambient_temp)

        n_steps = min(len(v_arr), len(i_arr), len(t_arr), len(temp_arr))
        if n_steps == 0:
            continue

        v_arr = v_arr[:n_steps].astype(float)
        i_arr = i_arr[:n_steps].astype(float)
        t_arr = t_arr[:n_steps].astype(float)
        temp_arr = temp_arr[:n_steps].astype(float)

        # Enforce signed convention: discharge negative, charge positive
        if c_type == "discharge":
            i_arr = -np.abs(i_arr)
        else:
            i_arr = np.abs(i_arr)

        # Per-cycle capacity & SOH
        if cycle_num in cycle_capacities:
            cap_cycle = cycle_capacities[cycle_num]
            last_known_cap = cap_cycle
        else:
            cap_cycle = last_known_cap

        soh_val = min(100.0, max(0.0, (cap_cycle / nominal_capacity) * 100.0))

        # Coulomb counting SOC
        dt_arr = np.diff(t_arr, prepend=t_arr[0])
        dt_arr[dt_arr < 0] = 0.0
        d_ah = (np.abs(i_arr) * dt_arr) / 3600.0
        cumu_ah = np.cumsum(d_ah)

        effective_cap = cap_cycle if cap_cycle > 0.1 else nominal_capacity
        if c_type == "discharge":
            # Discharge starts at 100% and drops
            soc_arr = 100.0 * (1.0 - (cumu_ah / effective_cap))
        else:
            # Charge starts near 0% and rises
            soc_arr = 100.0 * (cumu_ah / effective_cap)
        soc_arr = np.clip(soc_arr, 0.0, 100.0)

        for step in range(n_steps):
            t_s = float(t_arr[step])
            step_dt = base_dt + timedelta(seconds=t_s)
            records.append({
                "battery_id": battery_id,
                "cycle": cycle_num,
                "mode": c_type,
                "t_s": round(t_s, 3),
                "timestamp": step_dt.isoformat(),
                "voltage_v": round(float(v_arr[step]), 4),
                "current_a": round(float(i_arr[step]), 4),
                "temp_c": round(float(temp_arr[step]), 4),
                "ambient_c": round(float(ambient_temp), 2),
                "soc_pct": round(float(soc_arr[step]), 2),
                "capacity_ah": round(float(cap_cycle), 4),
                "soh_pct": round(float(soh_val), 2),
                "label": "normal",
                "fault_type": None,
            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} telemetry steps for {battery_id}.")
    return df


def ingest_all(
    raw_dir: str | Path = "data/raw",
    target_batteries: Optional[List[str]] = None,
    default_ambient: float = 24.0,
) -> pd.DataFrame:
    """Ingest all target battery .mat files and concatenate into a single DataFrame.

    Args:
        raw_dir: Directory containing NASA raw data.
        target_batteries: Optional battery list (default ['B0005', 'B0006', 'B0007', 'B0018']).
        default_ambient: Ambient temperature fallback in °C.

    Returns:
        Consolidated DataFrame of all telemetry.
    """
    targets = target_batteries or ["B0005", "B0006", "B0007", "B0018"]
    mat_files = find_mat_files(raw_dir, target_batteries=targets)

    if not mat_files:
        raise FileNotFoundError(f"No target .mat files for {targets} found in {raw_dir}")

    dfs = []
    for mat_path in mat_files:
        df_bat = ingest_battery_mat(mat_path, default_ambient=default_ambient)
        if not df_bat.empty:
            dfs.append(df_bat)

    if not dfs:
        raise ValueError("No data extracted from discovered .mat files.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined)} rows across {combined['battery_id'].nunique()} batteries.")
    return combined
