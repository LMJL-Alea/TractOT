import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
from nibabel.affines import apply_affine

def load_vtk_streamlines(vtk_file_path,reshape_scalars=False):
    """
    Load streamlines and associated scalar data from a VTK polydata file.
    This function reads a VTK file containing streamline data (polylines) and extracts
    both the geometric coordinates of the streamlines and any associated scalar arrays
    stored as point data.
    Parameters
    ----------
    vtk_file_path : str
        Path to the VTK polydata file to be loaded.
    reshape_scalars : bool, optional
        If True, reshape scalar arrays to match the structure of streamlines,
        where each streamline has its corresponding scalar values grouped together.
        If False, return scalar arrays as flat numpy arrays.
        Default is False.
    Returns
    -------
    streamlines : list of numpy.ndarray
        A list where each element is a numpy array of shape (n_points, 3) representing
        the 3D coordinates of points along a streamline.
    scalar_dict : dict
        A dictionary mapping scalar array names (str) to their values.
        - If reshape_scalars is False: values are 1D numpy arrays containing all
          scalar values in point order.
        - If reshape_scalars is True: values are lists of numpy arrays, where each
          array corresponds to the scalar values for one streamline.
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()
    lines = polydata.GetLines()
    streamlines = []
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    # Track point indices for each streamline
    point_indices = []
    
    while lines.GetNextCell(id_list):
        pts = []
        indices = []
        for j in range(id_list.GetNumberOfIds()):
            pid = id_list.GetId(j)
            pts.append(polydata.GetPoint(pid))
            indices.append(pid)
        streamlines.append(np.array(pts))
        point_indices.append(np.array(indices))
    
    # Load scalar arrays
    scalar_dict = {}
    point_data = polydata.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        array_name = array.GetName()
        scalar_values = [array.GetValue(j) for j in range(array.GetNumberOfTuples())]
        scalar_dict[array_name] = np.array(scalar_values)

    if reshape_scalars:
        # Reshape scalar arrays to match streamlines using point indices
        reshaped_scalars = {}
        for name, values in scalar_dict.items():
            reshaped_list = []
            for indices in point_indices:
                reshaped_list.append(values[indices])
            reshaped_scalars[name] = reshaped_list
        scalar_dict = reshaped_scalars
    
    return streamlines, scalar_dict


def save_vtk(streamlines, output_vtk_path,scalar_dict=None,to_lps=True):
    """
    Save streamlines to a VTK file format with optional scalar data.

    This function converts streamlines (fiber tracts) into VTK polydata format and writes
    them to a file. Optionally, scalar values can be associated with the points of the
    streamlines.

    Parameters
    ----------
    streamlines : list of array-like
        A list of streamlines, where each streamline is an array of 3D points (x, y, z).
        Each point represents a coordinate along the streamline trajectory.
    output_vtk_path : str
        The file path where the VTK file will be saved.
    scalar_dict : dict, optional
        A dictionary mapping scalar names (str) to scalar values. The scalar values can be:
        - A numpy array of values for each point across all streamlines
        - A list of arrays that will be concatenated
        The values will be flattened and added as point data to the polydata.
        Default is None.

    Returns
    -------
    None
        The function writes the data to disk but does not return a value.
    """
    
    if to_lps:
        # ras (mm) to lps (mm)
        to_lps = np.eye(4)
        to_lps[0, 0] = -1
        to_lps[1, 1] = -1
        streamlines=[apply_affine(to_lps, s) for s in streamlines]
        
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    point_id = 0
    for sl in streamlines:
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(sl))
        for i, pt in enumerate(sl):
            points.InsertNextPoint(pt)
            line.GetPointIds().SetId(i, point_id)
            point_id += 1
        lines.InsertNextCell(line)

    polydata.SetPoints(points)
    polydata.SetLines(lines)

    if scalar_dict:
        for scalar_name, scalar_values in scalar_dict.items():
            if isinstance(scalar_values, list):
                scalar_values = np.concatenate(scalar_values)
            vtk_array = numpy_to_vtk(scalar_values.flatten(), deep=True)
            vtk_array.SetName(scalar_name)
            polydata.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_vtk_path)
    writer.SetInputData(polydata)
    writer.Write()
