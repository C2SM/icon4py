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

from gt4py.next import backend as gtx_backend
import gt4py.next as gtx
from icon4py.model.common import dimension as dims

import datetime
import logging

# flake8: noqa
log = logging.getLogger(__name__)

RESTART_FREQUENCY = 4 # in time steps
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
        self.RESTART_FREQUENCY = RESTART_FREQUENCY
        self.restart_dir = RESTART_DIR
        self.base_filename = base_filename
        self.filepaths = [
            os.path.join(self.restart_dir, f"{base_filename}_0.pkl"),
            os.path.join(self.restart_dir, f"{base_filename}_1.pkl"),
        ]
        self._restart_data = None
        # TODO: Implement file locking for multi-process safety if needed in the future

    def restore_from_restart(
        self,
        prognostic_states,
        diagnostic_state_nh,
        backend: gtx_backend.Backend,
    ):
        """
        Restore state variables from the restart file into the provided state objects.
        Returns True if restoration was successful, False otherwise.
        """
        if self._restart_data is None:
            self._restart_data = self._read_restart()
        restart_data = self._restart_data
        if restart_data is None:
            log.info("No restart data found or file missing.")
            return None
        missing = []
        # Restore prognostic_states.current and .next
        for state_name, state_obj in [
            ("current", prognostic_states.current),
            ("next", prognostic_states.next),
        ]:
            for var in ["vn", "w", "rho", "exner", "theta_v"]:
                key = f"prognostic_states.{state_name}.{var}"
                if key in restart_data and restart_data[key] is not None:
                    arr = restart_data[key]["data"]
                    dim_names = restart_data[key]["dims"]
                    dims_tuple = tuple(getattr(dims, name + "Dim") for name in dim_names)
                    field = gtx.as_field(dims_tuple, arr, allocator=backend)
                    setattr(state_obj, var, field)
                    log.info(f"Restored {key} from restart file.")
                else:
                    missing.append(key)
        # Restore diagnostic_state_nh variables, handling nested attributes
        diag_restore = [
            (
                "perturbed_exner_at_cells_on_model_levels",
                "diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels",
            ),
            (
                "vertical_wind_advective_tendency.predictor",
                "diagnostic_state_nh.vertical_wind_advective_tendency.predictor",
            ),
        ]
        for attr_path, key in diag_restore:
            if key in restart_data and restart_data[key] is not None:
                arr = restart_data[key]["data"]
                dim_names = restart_data[key]["dims"]
                dims_tuple = tuple(getattr(dims, name + "Dim") for name in dim_names)
                attrs = attr_path.split(".")
                obj = diagnostic_state_nh
                for a in attrs[:-1]:
                    obj = getattr(obj, a)
                field = gtx.as_field(dims_tuple, arr, allocator=backend)
                setattr(obj, attrs[-1], field)
                log.info(f"Restored {key} from restart file.")
            else:
                missing.append(key)
        if missing:
            log.warning(f"Missing variables in restart file: {missing}")
        else:
            log.info(
                "All prognostic and diagnostic state variables successfully restored from restart file."
            )
        # Restore time_step_number if present
        time_step_number = restart_data.get("time_step_number", None)
        if time_step_number is not None:
            log.info(f"Restored time_step_number={time_step_number} from restart file.")
        return time_step_number

    def write_restart(
        self,
        prognostic_states,
        diagnostic_state_nh,
        time_step_number: int,
        last_written: Optional[int] = None,
    ) -> int:
        """
        Writes the simulation state to a restart file, alternating between two files.

        Only writes the variables that restore_from_restart expects, plus time_step_number.
        Args:
            prognostic_states: The prognostic states object.
            diagnostic_state_nh: The diagnostic state object.
            time_step_number: The current time step number to save.
            last_written: Index (0 or 1) of the last written file. If None, will pick based on file existence.
        Returns:
            The index (0 or 1) of the file just written.
        """
        state = {}
        # Save prognostic_states.current and .next
        for state_name, state_obj in [
            ("current", prognostic_states.current),
            ("next", prognostic_states.next),
        ]:
            for var in ["vn", "w", "rho", "exner", "theta_v"]:
                key = f"prognostic_states.{state_name}.{var}"
                field = getattr(state_obj, var)
                state[key] = {
                    "data": field.asnumpy(),
                    "dims": [d.value for d in field.domain.dims],
                }
        # Save diagnostic_state_nh variables (handle nested attributes)
        diag_field = getattr(diagnostic_state_nh, "perturbed_exner_at_cells_on_model_levels")
        state["diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels"] = {
            "data": diag_field.asnumpy(),
            "dims": [d.value for d in diag_field.domain.dims],
        }
        diag_field = getattr(
            getattr(diagnostic_state_nh, "vertical_wind_advective_tendency"), "predictor"
        )
        state["diagnostic_state_nh.vertical_wind_advective_tendency.predictor"] = {
            "data": diag_field.asnumpy(),
            "dims": [d.value for d in diag_field.domain.dims],
        }

        # Add time_step_number
        state["time_step_number"] = time_step_number
        # Add timestamp
        state["restart_timestamp"] = datetime.datetime.now().isoformat()

        # Determine which file to write (using metadata files)
        if last_written is None:
            times = []
            for fp in self.filepaths:
                ts, _, _ = self._read_metadata(fp)
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
        self._write_metadata(
            self.filepaths[idx], state["restart_timestamp"], time_step_number, list(state.keys())
        )
        return idx

    def _read_restart(self) -> Optional[dict]:
        """
        Reads the most recent restart file and returns the simulation state dict.
        Returns None if no valid restart file is found.
        """
        latest_file = self._check_restart_files()
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

    def _check_restart_files(self) -> Optional[str]:
        """
        Check if restart files are present in the restart directory.
        Returns the path to the most recent restart file if found, else None.
        Uses metadata files for efficiency.
        """
        latest_file = None
        latest_time = None
        for fp in self.filepaths:
            ts, _, _ = self._read_metadata(fp)
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

    def _write_metadata(
        self, restart_path: str, timestamp: str, time_step_number: int, keys: list
    ) -> None:
        """Write metadata file for a restart file."""
        meta_fp = self._get_meta_path(restart_path)
        try:
            with open(meta_fp, "w") as mf:
                mf.write(f"{timestamp}\n")
                mf.write(f"{time_step_number}\n")
                mf.write(",".join(keys) + "\n")
            log.info(
                f"Wrote metadata file: {meta_fp} (timestamp: {timestamp}, time_step_number: {time_step_number}, keys: {keys})"
            )
        except Exception as e:
            log.error(f"Failed to write metadata file {meta_fp}: {e}")

    def _read_metadata(self, restart_path: str) -> tuple:
        """Read metadata file for a restart file. Returns (timestamp, keys) or (None, None) if not found/invalid."""
        meta_fp = self._get_meta_path(restart_path)
        if not os.path.exists(meta_fp):
            log.info(f"Metadata file not found: {meta_fp}")
            return None, None, None
        try:
            with open(meta_fp, "r") as mf:
                lines = mf.read().splitlines()
                timestamp = lines[0] if lines else None
                time_step_number = lines[1] if len(lines) > 1 else 0
                keys = lines[2].split(",") if len(lines) > 2 else []
                log.info(
                    f"Read metadata file: {meta_fp} (timestamp: {timestamp}, time_step_number: {time_step_number}, keys: {keys})"
                )
                return timestamp, time_step_number, keys
        except Exception as e:
            log.error(f"Failed to read metadata file {meta_fp}: {e}")
            return None, None, None
