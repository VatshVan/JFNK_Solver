"""Write PETSc field snapshots to VTK XML files for ParaView inspection and animation."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

try:
    from petsc4py import PETSc
except ImportError:  # PETSC_STUB
    PETSc = None  # type: ignore[assignment]


def _vec_to_numpy(vec: "PETSc.Vec", nx: int, ny: int) -> np.ndarray:
    array = vec.getArray(readonly=True)
    size = array.size
    if size == nx * ny:
        return np.asarray(array, dtype=np.float64).reshape(ny, nx)
    if size == nx * ny * 2:
        return np.asarray(array, dtype=np.float64).reshape(ny, nx, 2)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def write_vtk(da: "PETSc.DMDA", fields: dict[str, "PETSc.Vec"], filename: str, timestep: int) -> None:
    """
    Write regular-grid PETSc fields to `outputs/vtk/{filename}_{timestep:06d}.vtu`.

    The companion `outputs/vtk/collection.pvd` file is kept in sync for ParaView animation.
    """

    if PETSc is None:  # pragma: no cover - covered when PETSc is unavailable
        raise RuntimeError("petsc4py is required for VTK export.")

    nx, ny = (int(value) for value in da.getSizes()[:2])
    output_dir = Path("outputs/vtk")
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = output_dir / f"{filename}_{timestep:06d}.vtu"
    pvd_path = output_dir / "collection.pvd"

    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(nx * ny),
        NumberOfCells=str((nx - 1) * (ny - 1)),
    )

    point_data = ET.SubElement(piece, "PointData")
    for name, vec in fields.items():
        array = _vec_to_numpy(vec, nx=nx, ny=ny)
        if array.ndim == 2:
            values = " ".join(f"{value:.16e}" for value in array.reshape(-1))
            data_array = ET.SubElement(point_data, "DataArray", type="Float64", Name=name, NumberOfComponents="1")
            data_array.text = values
        elif array.ndim == 3 and array.shape[-1] == 2:
            padded = np.zeros((ny, nx, 3), dtype=np.float64)
            padded[:, :, :2] = array
            values = " ".join(f"{value:.16e}" for value in padded.reshape(-1))
            data_array = ET.SubElement(point_data, "DataArray", type="Float64", Name=name, NumberOfComponents="3")
            data_array.text = values
        else:
            values = " ".join(f"{value:.16e}" for value in np.asarray(array).reshape(-1))
            data_array = ET.SubElement(point_data, "DataArray", type="Float64", Name=name, NumberOfComponents="1")
            data_array.text = values

    points = ET.SubElement(piece, "Points")
    point_array = ET.SubElement(points, "DataArray", type="Float64", NumberOfComponents="3")
    coordinates = []
    for j in range(ny):
        for i in range(nx):
            coordinates.extend((float(i), float(j), 0.0))
    point_array.text = " ".join(f"{value:.16e}" for value in coordinates)

    cells = ET.SubElement(piece, "Cells")
    connectivity = ET.SubElement(cells, "DataArray", type="Int32", Name="connectivity")
    offsets = ET.SubElement(cells, "DataArray", type="Int32", Name="offsets")
    types = ET.SubElement(cells, "DataArray", type="UInt8", Name="types")

    cell_connectivity: list[int] = []
    cell_offsets: list[int] = []
    cell_types: list[int] = []
    offset = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = j * nx + i
            p1 = p0 + 1
            p2 = p0 + nx + 1
            p3 = p0 + nx
            cell_connectivity.extend((p0, p1, p2, p3))
            offset += 4
            cell_offsets.append(offset)
            cell_types.append(9)  # VTK_QUAD

    connectivity.text = " ".join(str(value) for value in cell_connectivity)
    offsets.text = " ".join(str(value) for value in cell_offsets)
    types.text = " ".join(str(value) for value in cell_types)

    _indent_xml(root)
    ET.ElementTree(root).write(vtu_path, encoding="utf-8", xml_declaration=True)

    if pvd_path.exists():
        pvd_tree = ET.parse(pvd_path)
        pvd_root = pvd_tree.getroot()
        collection = pvd_root.find("Collection")
        assert collection is not None
    else:
        pvd_root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        collection = ET.SubElement(pvd_root, "Collection")

    ET.SubElement(
        collection,
        "DataSet",
        timestep=str(timestep),
        group="",
        part="0",
        file=vtu_path.name,
    )
    _indent_xml(pvd_root)
    ET.ElementTree(pvd_root).write(pvd_path, encoding="utf-8", xml_declaration=True)
