# this simple python script is used to generate a 3D hex mesh for the specified slab burner geometry
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


boundary_tag_dict = dict()
boundary_ids = []
line_tag_dict = dict()


# the sideList is a list of curves physical group and surface group
def defineBoundary(curve_list, name):
    curve_loop_ids = []
    # march over and add each side
    for curve in curve_list:
        line_tags = []
        for line in curve:
            line_key = str(line)
            if line_key in line_tag_dict:
                line_tags.append(line_tag_dict[line_key])
            else:
                if len(line) > 2:
                    line_tag = gmsh.model.geo.add_bspline(line)
                else:
                    line_tag = gmsh.model.geo.add_line(line[0], line[1])
                line_tag_dict[line_key] = line_tag
                line_tags.append(line_tag)

        # create a curve loop from
        curve_loop_ids.append(gmsh.model.geo.add_curve_loop(line_tags))

    if len(curve_loop_ids) > 2:
        surface_tag = gmsh.model.geo.add_surface_filling(curve_loop_ids)
    else:
        surface_tag = gmsh.model.geo.add_plane_surface(curve_loop_ids)

    # define the boundary condition for this
    physical_group_id = boundary_tag_dict.get(name, -1)
    tag_id = gmsh.model.geo.addPhysicalGroup(1, [surface_tag], physical_group_id)
    gmsh.model.setPhysicalName(1, tag_id, name)
    #
    if boundary_ids is not None:
        boundary_ids.append(surface_tag)


# Initialize gmsh:
gmsh.initialize()

# define the experimental chamber points
lowerLeftFront = convertToPoint((0.0, 0.0, 0.0127))
upperLeftFront = convertToPoint((0.0, 0.0254, 0.0127))
lowerRightFront = convertToPoint((0.1, 0.0, 0.0127))
upperRightFront = convertToPoint((0.1, 0.0254, 0.0127))
lowerLeftBack = convertToPoint((0.0, 0.0, -0.0127))
upperLeftBack = convertToPoint((0.0, 0.0254, -0.0127))
lowerRightBack = convertToPoint((0.1, 0.0, -0.0127))
upperRightBack = convertToPoint((0.1, 0.0254, -0.0127))

# # define a list of points for the slab burner, starting with the left most point
slabBoundaryLocationsFront = [(0.0132, 0, 0.00401),
                              (0.0173, 0.0045, 0.00401),
                              (0.0214, 0.0069, 0.00401),
                              (0.0258, 0.0082, 0.00401),
                              (0.0385, 0.009, 0.00401),
                              (0.0592, 0.009, 0.00401),
                              (0.0702, 0.0087, 0.00401),
                              (0.0722, 0.0079, 0.00401),
                              (0.0727, 0.007, 0.00401),
                              (0.0728, 0, 0.00401)]

slabBoundaryLocationsBack = [(0.0132, 0, -0.00401),
                             (0.0173, 0.0045, -0.00401),
                             (0.0214, 0.0069, -0.00401),
                             (0.0258, 0.0082, -0.00401),
                             (0.0385, 0.009, -0.00401),
                             (0.0592, 0.009, -0.00401),
                             (0.0702, 0.0087, -0.00401),
                             (0.0722, 0.0079, -0.00401),
                             (0.0727, 0.007, -0.00401),
                             (0.0728, 0, -0.00401)]
# convert the locations to points
slabBoundaryFront = convertToPoint(slabBoundaryLocationsFront)
slabBoundaryBack = convertToPoint(slabBoundaryLocationsBack)

# # define the chamber boundary with associated names
defineBoundary([
    [  # front of slab
        slabBoundaryFront,
        [slabBoundaryFront[-1], slabBoundaryFront[0]]
    ],
], "slab")
defineBoundary([
    [  # back of slab
        slabBoundaryBack,
        [slabBoundaryBack[-1], slabBoundaryBack[0]]
    ],
], "slab")
defineBoundary([
    [  # top of slab
        slabBoundaryBack,
        [slabBoundaryBack[-1], slabBoundaryFront[-1]],
        slabBoundaryFront[::-1],
        [slabBoundaryFront[0], slabBoundaryBack[0]],

    ],
], "slab")
defineBoundary([
    [
        [lowerLeftFront, lowerLeftBack],
        [lowerLeftBack, upperLeftBack],
        [upperLeftBack, upperLeftFront],
        [upperLeftFront, lowerLeftFront]
    ],
], "inlet")
defineBoundary([
    [
        [lowerLeftFront, lowerLeftBack],
        [lowerLeftBack, upperLeftBack],
        [upperLeftBack, upperLeftFront],
        [upperLeftFront, lowerLeftFront]
    ],
], "outlet")

defineBoundary([
    [  # outside of bottom of chamber
        [lowerLeftFront, lowerLeftBack],
        [lowerLeftBack, lowerRightBack],
        [lowerRightBack, lowerRightFront],
        [lowerRightFront, lowerLeftFront]
    ],
    [ # around slab
        [slabBoundaryFront[0], slabBoundaryBack[0]],
        [slabBoundaryBack[0], slabBoundaryBack[-1]],
        [slabBoundaryBack[-1], slabBoundaryFront[-1]],
        [slabBoundaryFront[-1], slabBoundaryFront[0]]
    ]
], "wall")

# defineBoundary([[upperRight, lowerRight]], "outlet", boundary_ids)
# defineBoundary([
#     [upperLeft, upperRight],
#     [lowerLeft, slabBoundary[0]],
#     [slabBoundary[-1], lowerRight]
# ], "wall", boundary_ids)
# defineBoundary([slabBoundary], "slab", boundary_ids)
#
# # define the curve and resulting plane
# curve_id = gmsh.model.geo.add_curve_loop(boundary_ids, reorient=True)
# surface_id = gmsh.model.geo.add_plane_surface([curve_id])
# gmsh.model.setPhysicalName(2, gmsh.model.geo.addPhysicalGroup(2, [surface_id]), "main")
#
# # Create the relevant Gmsh data structures from Gmsh model.
gmsh.model.geo.synchronize()
#
# # set the default properties to generate quad mesh
# gmsh.option.setNumber("Mesh.Algorithm", 8)  # 8: Frontal-Delaunay for Quads
# gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # 1: Delaunay
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # 3: blossom full-quad
# gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0005)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 0.001)  # assume about 1mm resolution
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1: all quadrangles
# gmsh.option.setNumber("Mesh.RecombineAll", 1)  # true
#
# # set the options to prevent gmsh from adding too many elements to each geometry line
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
# gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
#
# # generate the mesh
# gmsh.model.mesh.setRecombine(2, surface_id)
# gmsh.model.mesh.generate(2)
# gmsh.model.mesh.reverse()  # we have to reverse the element orientation for petsc

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
