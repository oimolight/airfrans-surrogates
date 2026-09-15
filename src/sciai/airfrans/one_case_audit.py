from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyvista as pv
from airfrans import Simulation, __version__ as airfrans_module_version


EXPECTED_SELECTION_SHA256 = "8c532a15c2af6538177ae9885de8978553b162c4eb788d33a2998c9edc447014"
DATASET_ROOT = Path("data/raw/airfrans_hf_selected/data/Dataset")
REPORT_PATH = Path("reports/airfrans_one_case_audit.md")
FIGURE_DIRECTORY = Path("reports/figures")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_summary(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    numeric = np.issubdtype(values.dtype, np.number)
    floating = np.issubdtype(values.dtype, np.floating)
    finite = np.isfinite(values) if numeric else np.ones(values.shape, dtype=bool)
    if numeric and values.ndim > 1:
        axes = tuple(range(values.ndim - 1))
        minimum: object = np.min(values, axis=axes).tolist()
        maximum: object = np.max(values, axis=axes).tolist()
    elif numeric:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
    else:
        minimum = None
        maximum = None
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nan_count": int(np.isnan(values).sum()) if floating else 0,
        "inf_count": int(np.isinf(values).sum()) if floating else 0,
        "finite_count": int(finite.sum()),
        "min": minimum,
        "max": maximum,
    }


def mesh_arrays(mesh_name: str, mesh: pv.DataSet) -> list[dict[str, object]]:
    rows = [{"mesh": mesh_name, "location": "points", "name": "coordinates", **array_summary(mesh.points)}]
    for location, arrays in (("point_data", mesh.point_data), ("cell_data", mesh.cell_data)):
        for name in sorted(arrays):
            rows.append({"mesh": mesh_name, "location": location, "name": name, **array_summary(arrays[name])})
    return rows


def shape_text(shape: object) -> str:
    return "(" + ", ".join(str(value) for value in shape) + ("," if len(shape) == 1 else "") + ")"


def value_text(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(f"{number:.9g}" for number in value) + "]"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def save_geometry_plot(airfoil: pv.PolyData, path: Path, case_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 620))
    plotter.set_background("#F4F7FA")
    plotter.add_mesh(airfoil, color="#17324D", line_width=4)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="x [m]",
        ytitle="y [m]",
        show_zaxis=False,
        color="#43515E",
    )
    plotter.add_title(f"AirfRANS geometry - {case_id}", color="#17212B", font_size=12)
    plotter.screenshot(path)
    plotter.close()


def save_pressure_plot(internal: pv.DataSet, airfoil: pv.PolyData, path: Path, case_id: str) -> None:
    pressure = np.asarray(internal.point_data["p"])
    pressure_limit = max(abs(float(pressure.min())), abs(float(pressure.max())))
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 900))
    plotter.set_background("#F4F7FA")
    plotter.add_mesh(
        internal,
        scalars="p",
        style="points",
        point_size=2.5,
        cmap="RdBu_r",
        clim=(-pressure_limit, pressure_limit),
        scalar_bar_args={"title": "p/rho [m^2/s^2]", "color": "#17212B"},
    )
    plotter.add_mesh(airfoil, color="#111820", line_width=2)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.show_bounds(
        grid="back",
        location="outer",
        xtitle="x [m]",
        ytitle="y [m]",
        show_zaxis=False,
        color="#43515E",
    )
    plotter.add_title(f"Ground-truth pressure p/rho - {case_id}", color="#17212B", font_size=12)
    plotter.screenshot(path)
    plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/airfrans_selected_cases.json"))
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    args = parser.parse_args()

    selection_sha256 = file_sha256(args.selection)
    if selection_sha256 != EXPECTED_SELECTION_SHA256:
        raise RuntimeError(f"Frozen selection changed: {selection_sha256}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    case_id = sorted(selection["splits"]["train"])[0]
    memberships = [name for name, case_ids in selection["splits"].items() if case_id in case_ids]
    if memberships != ["train"]:
        raise RuntimeError(f"Audit case must belong only to train, found {memberships}")

    case_directory = args.dataset_root / case_id
    paths = {
        "aerofoil": case_directory / f"{case_id}_aerofoil.vtp",
        "freestream": case_directory / f"{case_id}_freestream.vtp",
        "internal": case_directory / f"{case_id}_internal.vtu",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(f"The three selected case files are not present: {case_id}")

    aerofoil = pv.read(paths["aerofoil"])
    freestream = pv.read(paths["freestream"])
    internal = pv.read(paths["internal"])
    simulation = Simulation(root=str(args.dataset_root), name=case_id)
    meshes = {"internal.vtu": internal, "aerofoil.vtp": aerofoil, "freestream.vtp": freestream}
    rows = [row for mesh_name, mesh in meshes.items() for row in mesh_arrays(mesh_name, mesh)]

    dataset_surface = np.asarray(internal.point_data["U"])[:, 0] == 0
    geometry_surface = np.asarray(internal.point_data["implicit_distance"]) == 0
    true_positive = int(np.count_nonzero(dataset_surface & geometry_surface))
    false_positive = int(np.count_nonzero(~dataset_surface & geometry_surface))
    false_negative = int(np.count_nonzero(dataset_surface & ~geometry_surface))
    agreement = int(np.count_nonzero(dataset_surface == geometry_surface))
    internal_surface_positions = {tuple(point) for point in np.asarray(internal.points)[geometry_surface, :2]}
    airfoil_positions = {tuple(point) for point in np.asarray(aerofoil.points)[:, :2]}
    coordinate_sets_equal = internal_surface_positions == airfoil_positions

    tokens = case_id.split("_")
    inlet_velocity = float(tokens[2])
    angle_degrees = float(tokens[3])
    geometry_parameters = [float(token) for token in tokens[4:]]
    if len(geometry_parameters) != 3:
        raise RuntimeError("This deterministic audit case is not a NACA 4-digit-series parameterisation")

    velocity = np.asarray(internal.point_data["U"])
    velocity_magnitude = np.linalg.norm(velocity[:, :2], axis=1)
    pressure = np.asarray(internal.point_data["p"])
    geometry_path = args.figure_directory / "airfrans_one_case_geometry.png"
    pressure_path = args.figure_directory / "airfrans_one_case_pressure.png"
    save_geometry_plot(aerofoil, geometry_path, case_id)
    save_pressure_plot(internal, aerofoil, pressure_path, case_id)

    table_rows = "\n".join(
        f"| {row['mesh']} | {row['location']} | `{row['name']}` | `{shape_text(row['shape'])}` | "
        f"`{row['dtype']}` | {row['nan_count']} | {row['inf_count']} | {value_text(row['min'])} | {value_text(row['max'])} |"
        for row in rows
    )
    coordinate_rows = "\n".join(
        f"| {mesh_name} | {value_text(np.min(mesh.points, axis=0).tolist())} | {value_text(np.max(mesh.points, axis=0).tolist())} |"
        for mesh_name, mesh in meshes.items()
    )
    report = f"""# AirfRANS one-case audit

Status: ONE TRAINING CASE AUDITED; NO TRAINING PERFORMED

## Selection

- Case ID: `{case_id}`
- Selection rule: lexicographically first ID in the frozen local `train` split
- Split membership: `train` only
- Frozen selection SHA-256: `{selection_sha256}`
- Files read: exactly the case's `_internal.vtu`, `_aerofoil.vtp`, and `_freestream.vtp`

## Conditions and geometry

- Inlet velocity magnitude: {inlet_velocity} m/s
- Angle of attack: {angle_degrees} degrees ({float(simulation.angle_of_attack):.9g} rad)
- Geometry family: NACA 4-digit series parameterisation; three real-valued parameters
- Geometry parameters `(M, P, XX)`: `({geometry_parameters[0]}, {geometry_parameters[1]}, {geometry_parameters[2]})`
- Formula interpretation used by the installed AirfRANS generator: maximum camber `M/100`, location `P/10` chord, thickness `XX/100` chord
- Internal points: {internal.n_points}
- Internal cells: {internal.n_cells}
- Surface points: {int(dataset_surface.sum())} (both dataset label and geometry-only mask)
- Aerofoil mesh points/cells: {aerofoil.n_points}/{aerofoil.n_cells}
- Freestream mesh points/cells: {freestream.n_points}/{freestream.n_cells}

![Geometry](figures/{geometry_path.name})

## Arrays

The shape is the exact raw VTK array shape. Vector minima and maxima are component-wise in x, y, z order.

| Mesh | Location | Array | Shape | dtype | NaN | inf | Min | Max |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
{table_rows}

## Coordinate ranges

Coordinates are in metres. The preprocessed slice is at z = 0.5 m.

| Mesh | Component-wise minimum [x, y, z] | Component-wise maximum [x, y, z] |
| --- | ---: | ---: |
{coordinate_rows}

## Target ranges and units

- Pressure target `p`: [{float(pressure.min()):.9g}, {float(pressure.max()):.9g}] m^2/s^2. This is pressure divided by specific mass, `p/rho`, not Pa. AirfRANS documentation and the PyTorch Geometric dataset contract both confirm this unit.
- Velocity target `U_x`: [{float(velocity[:, 0].min()):.9g}, {float(velocity[:, 0].max()):.9g}] m/s
- Velocity target `U_y`: [{float(velocity[:, 1].min()):.9g}, {float(velocity[:, 1].max()):.9g}] m/s
- Velocity target `U_z`: [{float(velocity[:, 2].min()):.9g}, {float(velocity[:, 2].max()):.9g}] m/s
- In-plane speed magnitude: [{float(velocity_magnitude.min()):.9g}, {float(velocity_magnitude.max()):.9g}] m/s
- Across every numeric coordinate, point-data, and cell-data array listed above: NaN count {sum(int(row['nan_count']) for row in rows)}, inf count {sum(int(row['inf_count']) for row in rows)}.

![Ground-truth pressure](figures/{pressure_path.name})

## Surface identification audit

The dataset/library surface label is `internal.point_data['U'][:, 0] == 0`, so it depends on a target field. The geometry-only audit mask is `internal.point_data['implicit_distance'] == 0`.

- Dataset surface positives: {int(dataset_surface.sum())}
- Geometry-only surface positives: {int(geometry_surface.sum())}
- True positives: {true_positive}
- False positives: {false_positive}
- False negatives: {false_negative}
- Point-wise agreement: {agreement}/{internal.n_points} ({agreement / internal.n_points:.12%})
- Geometry-only internal surface coordinate set equals the aerofoil mesh coordinate set: {coordinate_sets_equal}

Conclusion: geometry-derived surface identification agrees exactly for this audited case. This is one-case evidence, not a dataset-wide guarantee.

## Target leakage audit

No model input or training dataset was constructed in this audit, so no target was used as a model input. For later implementation:

- Allowed inputs: coordinates, inlet-velocity vector, signed distance, and surface/normals recomputed only from geometry.
- Target: pressure `p/rho` only for this project.
- Prohibited inputs: raw `U`, `p`, `nut`, `Simulation.surface`, and `Simulation.normals`.
- Reason: installed AirfRANS `{version('airfrans')}` (`airfrans.__version__` reports `{airfrans_module_version}`) constructs `Simulation.surface` from target `U_x == 0`, then constructs `Simulation.normals` through that mask. Using those two loader attributes directly would leak target-derived information.

## Evidence and limitations

- Data source: downloaded Hugging Face mirror revision recorded in `reports/airfrans_download_result.json`.
- Unit source: official AirfRANS Simulation documentation and the PyTorch Geometric AirfRANS dataset documentation.
- Geometry source: installed `airfrans/naca_generator.py` and official AirfRANS Simulation documentation.
- This audit read one training case only. It did not inspect another CFD case, train a model, or establish multi-case generalisation.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": case_id,
                "internal_points": int(internal.n_points),
                "surface_points": int(dataset_surface.sum()),
                "surface_agreement": agreement == internal.n_points,
                "nan_count": sum(int(row["nan_count"]) for row in rows),
                "inf_count": sum(int(row["inf_count"]) for row in rows),
                "report": str(args.report),
                "figures": [str(geometry_path), str(pressure_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
