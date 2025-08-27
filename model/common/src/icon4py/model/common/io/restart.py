
# (Removed duplicate code and class definition. The correct RestartManager class is defined below.)
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
from typing import Optional

import datetime
import logging

# flake8: noqa
log = logging.getLogger(__name__)

RESTART_DIR = os.path.join(
    os.environ.get("ICON4PY_OUTPUT_DIR", "runxxx_undefined_output"), "restart"
)

if not os.path.isdir(RESTART_DIR):
    os.makedirs(RESTART_DIR)


class RestartManager:
    """
    Handles reading and writing of restart files for simulation restarts.
    Maintains two alternating files for backup in case of interruption.
    """

    def __init__(self, base_filename: str = "restart"):
        self.restart_dir = RESTART_DIR
        self.base_filename = base_filename
        self.filepaths = [
            os.path.join(self.restart_dir, f"{base_filename}_0.pkl"),
            os.path.join(self.restart_dir, f"{base_filename}_1.pkl"),
        ]
        self._restart_data = None
        # TODO: Implement file locking for multi-process safety if needed in the future

    def restore_from_restart(self, prognostic_states, diagnostic_state_nh, logger=None):
        """
        Restore state variables from the restart file into the provided state objects.
        Returns True if restoration was successful, False otherwise.
        """
        if self._restart_data is None:
            self._restart_data = self._read_restart()
        restart_data = self._restart_data
        if restart_data is None:
            if logger:
                logger.info("No restart data found or file missing.")
            return False
        missing = []
        # Restore prognostic_states.current and .next
        for state_name, state_obj in [("current", prognostic_states.current), ("next", prognostic_states.next)]:
            for var in ["vn", "w", "rho", "exner", "theta_v"]:
                key = f"prognostic_states.{state_name}.{var}"
                if key in restart_data:
                    setattr(state_obj, var, restart_data[key])
                    if logger:
                        logger.info(f"Restored {key} from restart file.")
                else:
                    missing.append(key)
        # Restore diagnostic_state_nh variables, handling nested attributes
        diag_restore = [
            ("perturbed_exner_at_cells_on_model_levels", "diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels"),
            ("vertical_wind_advective_tendency.predictor", "diagnostic_state_nh.vertical_wind_advective_tendency.predictor"),
        ]
        for attr_path, key in diag_restore:
            if key in restart_data:
                attrs = attr_path.split(".")
                obj = diagnostic_state_nh
                for a in attrs[:-1]:
                    obj = getattr(obj, a)
                setattr(obj, attrs[-1], restart_data[key])
                if logger:
                    logger.info(f"Restored {key} from restart file.")
            else:
                missing.append(key)
        if missing and logger:
            logger.warning(f"Missing variables in restart file: {missing}")
        elif logger:
            logger.info("All prognostic and diagnostic state variables successfully restored from restart file.")
        return restart_data is not None

    def write_restart(self, state: dict, last_written: Optional[int] = None) -> int:
        """
        Writes the simulation state to a restart file, alternating between two files.
        Adds a timestamp to the state dict and writes a metadata file for fast lookup.
        Args:
            state: The simulation state to serialize (dict).
            last_written: Index (0 or 1) of the last written file. If None, will pick based on file existence.
        Returns:
            The index (0 or 1) of the file just written.
        """

        # Add timestamp to state
        state = dict(state)  # Copy to avoid mutating caller's dict
        state["restart_timestamp"] = datetime.datetime.now().isoformat()

        # Determine which file to write (using metadata files)
        if last_written is None:
            times = []
            for fp in self.filepaths:
                ts, _ = self._read_metadata(fp)
                times.append(ts if ts else "")
            if times[0] and times[1]:
                idx = 0 if times[0] <= times[1] else 1
            elif times[0]:
                idx = 1
            else:
                idx = 0
        else:
            idx = 1 - last_written

        # Write atomically: write to temp file, then rename
        temp_path = self.filepaths[idx] + ".tmp"
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, self.filepaths[idx])  # atomic on POSIX
            log.info(
                f"Wrote restart file: {self.filepaths[idx]} (timestamp: {state['restart_timestamp']})"
            )
        except Exception as e:
            log.error(f"Failed to write restart file {self.filepaths[idx]}: {e}")
            return idx

        # Write metadata file
        self._write_metadata(self.filepaths[idx], state["restart_timestamp"], list(state.keys()))
        return idx

    def _read_restart(self) -> Optional[dict]:
        """
        Reads the most recent restart file and returns the simulation state dict.
        Returns None if no valid restart file is found.
        """
        latest_file = self.check_restart_files()
        if latest_file is not None:
            try:
                with open(latest_file, "rb") as f:
                    data = pickle.load(f)
                log.info(f"Successfully read restart file: {latest_file}")
                return data
            except Exception as e:
                log.error(f"Failed to read restart file {latest_file}: {e}")
                return None
        log.info("No restart file to read.")
        return None

    def check_restart_files(self) -> Optional[str]:
        """
        Check if restart files are present in the restart directory.
        Returns the path to the most recent restart file if found, else None.
        Uses metadata files for efficiency.
        """
        latest_file = None
        latest_time = None
        for fp in self.filepaths:
            ts, _ = self._read_metadata(fp)
            if ts is not None:
                if latest_time is None or ts > latest_time:
                    latest_time = ts
                    latest_file = fp
        if latest_file:
            log.info(f"Most recent restart file: {latest_file} (timestamp: {latest_time})")
        else:
            log.info("No valid restart files found.")
        return latest_file

    def _get_meta_path(self, restart_path: str) -> str:
        """Return the metadata file path for a given restart file path."""
        return restart_path + ".meta"

    def _write_metadata(self, restart_path: str, timestamp: str, keys: list) -> None:
        """Write metadata file for a restart file."""
        meta_fp = self._get_meta_path(restart_path)
        try:
            with open(meta_fp, "w") as mf:
                mf.write(f"{timestamp}\n")
                mf.write(",".join(keys) + "\n")
            log.info(f"Wrote metadata file: {meta_fp} (timestamp: {timestamp}, keys: {keys})")
        except Exception as e:
            log.error(f"Failed to write metadata file {meta_fp}: {e}")

    def _read_metadata(self, restart_path: str) -> tuple:
        """Read metadata file for a restart file. Returns (timestamp, keys) or (None, None) if not found/invalid."""
        meta_fp = self._get_meta_path(restart_path)
        if not os.path.exists(meta_fp):
            log.info(f"Metadata file not found: {meta_fp}")
            return None, None
        try:
            with open(meta_fp, "r") as mf:
                lines = mf.read().splitlines()
                timestamp = lines[0] if lines else None
                keys = lines[1].split(",") if len(lines) > 1 else []
                log.info(f"Read metadata file: {meta_fp} (timestamp: {timestamp}, keys: {keys})")
                return timestamp, keys
        except Exception as e:
            log.error(f"Failed to read metadata file {meta_fp}: {e}")
            return None, None
