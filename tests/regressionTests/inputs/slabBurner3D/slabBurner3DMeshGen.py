# this simple python script is used to generate a 3D extruded hex mesh for the specified slab burner geometry
import math
import sys

import gmsh
import argparse


# function to convert the specified locations to gMsh points
def convertToPoint(locations):
    if type(locations) is list:
        points = []
        for location in locations:
            points.append(gmsh.model.geo.add_point(location[0], location[1], location[2]))
        return points
    else:
        return gmsh.model.geo.add_point(locations[0], locations[1], locations[2])


# the sideList is a list of list of sides
boundary_ids = []
curve_dict = dict()

def defineBoundary(sides, name):
    line_ids = []
    # march over and add each side
    for side in sides:
        line_ids.append(gmsh.model.geo.add_bspline(side))

    boundary_ids.extend(line_ids)
    curve_dict[name] = line_ids


boundary_physical_group_dict = dict()


def assign_boundary_group(entity, default_group):
    upper_adjacencies, lower_adjacencies = gmsh.model.get_adjacencies(entity[0], entity[1])
    # check if any of the lower_adjacencies are in the curve curve_dict
    for lower_adj in lower_adjacencies:
        for name, curves in curve_dict.items():
            if lower_adj in curves:
                if name in boundary_physical_group_dict:
                    boundary_physical_group_dict[name].append(entity[1])
                    return
                else:
                    boundary_physical_group_dict[name] = [entity[1]]
                    return

    # add to default
    if default_group in boundary_physical_group_dict:
        boundary_physical_group_dict[default_group].append(entity[1])
        return
    else:
        boundary_physical_group_dict[default_group] = [entity[1]]
        return


# Initialize gmsh:
gmsh.initialize()

# define the thickness
thickness = 0.0254
offset = -thickness / 2.0

# define an approximate mesh size
dx = 0.0007

# define the experimental chamber points
lowerLeft = convertToPoint((0.0, 0.0, offset))
upperLeft = convertToPoint((0.0, 0.0254, offset))
lowerRight = convertToPoint((0.1, 0.0, offset))
upperRight = convertToPoint((0.1, 0.0254, offset))

# define a list of points for the slab burner, starting with the left most point
slabBoundaryLocations = [(0.0132, 0, offset),
                         (0.0173, 0.0045, offset),
                         (0.0214, 0.0069, offset),
                         (0.0258, 0.0082, offset),
                         (0.0385, 0.009, offset),
                         (0.0592, 0.009, offset),
                         (0.0702, 0.0087, offset),
                         (0.0722, 0.0079, offset),
                         (0.0727, 0.007, offset),
                         (0.0728, 0, offset)]

# convert the locations to points
slabBoundary = convertToPoint(slabBoundaryLocations)

# define the chamber boundary in a counterclockwise order

defineBoundary([[upperLeft, lowerLeft]], "inlet")
defineBoundary([[upperRight, lowerRight]], "outlet")
defineBoundary([
    [upperRight, upperLeft],
    [lowerLeft, slabBoundary[0]],
    [slabBoundary[-1], lowerRight]
], "wall")
defineBoundary([slabBoundary], "slab")

# define the curve and resulting plane
curve_id = gmsh.model.geo.add_curve_loop(boundary_ids, reorient=True)
surface_id = gmsh.model.geo.add_plane_surface([curve_id])

# extrude the resulting mesh in z along the direction
extruded = gmsh.model.geo.extrude([(2, surface_id)], 0.0, 0.0, thickness, [thickness / dx], [1], recombine=True)

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

# also add the org surface
boundary_physical_group_dict["wall"] = [surface_id]

# go back and define all the physical boundary
volume_id = -1
for entity in extruded:
    if entity[0] == 3:
        volume_id = entity[1]
        gmsh.model.geo.addPhysicalGroup(3, [entity[1]], name="main")
    else:
        assign_boundary_group(entity, "wall")

# # create an assign the physical groups
for name, ids in boundary_physical_group_dict.items():
    gmsh.model.geo.addPhysicalGroup(2, ids, name=name)

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

# set the default properties to generate quad mesh
gmsh.option.setNumber("Mesh.Algorithm", 8)  # 8: Frontal-Delaunay for Quads (11 does not yet work for extrude)
gmsh.option.setNumber("Mesh.Algorithm3D", 2)  # 1: Delaunay
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # 3: blossom full-quad
gmsh.option.setNumber("Mesh.MeshSizeMin", dx)
gmsh.option.setNumber("Mesh.MeshSizeMax", dx*1.1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: all quadrangles
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # true

# set the options to prevent gmsh from adding too many elements to each geometry line
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.Smoothing", 10)

# generate the mesh
gmsh.model.mesh.setRecombine(3, volume_id)
gmsh.model.mesh.generate(3)

# parse input arguments
parser = argparse.ArgumentParser(
    description='Generates 2D slabBurner Hex Mesh')
parser.add_argument('--preview', dest='preview', action='store_true',
                    help='If true, preview mesh instead of saving', default=False)
parser.add_argument('--summary', dest='summary', action='store_true',
                    help='If true, computes the element summary', default=False)
args = parser.parse_args()

if args.summary:
    # print a summary of mesh information
    elements = gmsh.model.mesh.getElements(2)
    print(f'Number Elements: {len(elements[1][0])}')
    minDistance = 1E30
    maxDistance = 0
    for ele_tag in elements[1][0]:
        element = gmsh.model.mesh.getElement(ele_tag)
        node_ids = element[1]
        number_nodes = len(node_ids)
        for n in range(number_nodes):
            node_n = gmsh.model.mesh.get_node(node_ids[n])[0]
            for nn in range(n + 1, number_nodes):
                node_nn = gmsh.model.mesh.get_node(node_ids[nn])[0]
                distance = math.sqrt(
                    (node_n[0] - node_nn[0]) ** 2 + (node_n[1] - node_nn[1]) ** 2 + (node_n[2] - node_nn[2]) ** 2)
                minDistance = min(minDistance, distance)
                maxDistance = max(maxDistance, distance)
    print(f'Min/Max Distance: {minDistance}/{maxDistance}')

if args.preview:
    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()
else:
    # Write mesh data:
    gmsh.write("slabBurner3DMesh.msh")

# It finalizes the Gmsh API
gmsh.finalize()
