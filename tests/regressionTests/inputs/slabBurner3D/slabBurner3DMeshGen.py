# this simple python script is used to generate a 3D extruded hex mesh for the specified slab burner geometry
import sys

import gmsh
import argparse


# function to convert the specified locations to gMsh points
def convertToPoint(locations):
    if type(locations) is list:
        points = []
        for location in locations:
            points.append(gmsh.model.geo.add_point(location[0], location[1], 0.0))
        return points
    else:
        return gmsh.model.geo.add_point(locations[0], locations[1], 0.0)


# the sideList is a list of list of sides
def defineBoundary(sides, name, boundary_list):
    line_ids = []
    # march over and add each side
    for side in sides:
        line_ids.append(gmsh.model.geo.add_bspline(side))

    # define the boundary condition for this
    tag_id = gmsh.model.geo.addPhysicalGroup(1, line_ids)
    gmsh.model.setPhysicalName(1, tag_id, name)

    if boundary_list is not None:
        boundary_list.extend(line_ids)


# Initialize gmsh:
gmsh.initialize()

# define the experimental chamber points
lowerLeft = convertToPoint((0.0, 0.0))
upperLeft = convertToPoint((0.0, 0.0254))
lowerRight = convertToPoint((0.1, 0.0))
upperRight = convertToPoint((0.1, 0.0254))

# define a list of points for the slab burner, starting with the left most point
slabBoundaryLocations = [(0.0132, 0),
                         (0.0173, 0.0045),
                         (0.0214, 0.0069),
                         (0.0258, 0.0082),
                         (0.0385, 0.009),
                         (0.0592, 0.009),
                         (0.0702, 0.0087),
                         (0.0722, 0.0079),
                         (0.0727, 0.007),
                         (0.0728, 0)]

# convert the locations to points
slabBoundary = convertToPoint(slabBoundaryLocations)

# define the chamber boundary with associated names, define the nodes in a counterclockwise order
boundary_ids = []
defineBoundary([[upperLeft, lowerLeft]], "inlet", boundary_ids)
defineBoundary([[upperRight, lowerRight]], "outlet", boundary_ids)
defineBoundary([
    [upperRight, upperLeft],
    [lowerLeft, slabBoundary[0]],
    [slabBoundary[-1], lowerRight]
], "wall", boundary_ids)
defineBoundary([slabBoundary], "slab", boundary_ids)

# define the curve and resulting plane
curve_id = gmsh.model.geo.add_curve_loop(boundary_ids, reorient=True)
surface_id = gmsh.model.geo.add_plane_surface([curve_id])
gmsh.model.setPhysicalName(2, gmsh.model.geo.addPhysicalGroup(2, [surface_id]), "main")

extruded = gmsh.model.geo.extrude([(2, surface_id)], 0.0, 0.0, .024, [8], [1], recombine=True)

# Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()

# set the default properties to generate quad mesh
gmsh.option.setNumber("Mesh.Algorithm", 11)  #11: Quasi-structured Quad
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1: Delaunay
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # 3: blossom full-quad
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.005)  # assume about 1mm resolution
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: all quadrangles
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # true

# set the options to prevent gmsh from adding too many elements to each geometry line
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

# generate the mesh
# gmsh.model.mesh.setRecombine(2, surface_id)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.reverse()  # we have to reverse the element orientation for petsc

# parse input arguments
parser = argparse.ArgumentParser(
    description='Generates 2D slabBurner Hex Mesh')
parser.add_argument('--preview', dest='preview', action='store_true',
                    help='If true, preview mesh instead of saving', default=False)
args = parser.parse_args()

if args.preview:
    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()
else:
    # # Write mesh data:
    gmsh.write("slabBurnerMesh.msh")

# It finalize the Gmsh API
gmsh.finalize()
