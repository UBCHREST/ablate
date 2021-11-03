
#include "createLabel.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifier::CreateLabel::CreateLabel(std::string name, std::shared_ptr<mathFunctions::MathFunction> function, int dmDepth, int labelValueIn)
    : name(name), function(function), dmHeight((PetscInt)dmDepth), labelValue(labelValueIn == 0 ? 1 : (PetscInt)labelValueIn) {}

void ablate::domain::modifier::CreateLabel::Modify(DM &dm) {
    DMCreateLabel(dm, name.c_str()) >> checkError;

    DMLabel newLabel;
    DMGetLabel(dm, name.c_str(), &newLabel) >> checkError;

    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, dmHeight, &cStart, &cEnd) >> checkError;

    // Get the number of dimensions from the dm
    PetscInt nDims;
    DMGetCoordinateDim(dm, &nDims) >> checkError;

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscReal centroid[3];

        // get the center of the cell/face/vertex
        DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL) >> checkError;

        // determine if the point is positive or negative
        auto evalValue = function->Eval(centroid, nDims, 0.0);
        if (evalValue > 0) {
            DMLabelSetValue(newLabel, c, labelValue) >> checkError;
        }
    }
    DMPlexLabelComplete(dm, newLabel) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::modifier::Modifier, ablate::domain::modifier::CreateLabel, "Creates a new label for all positive points in the function", ARG(std::string, "name", "the new label name"),
         ARG(mathFunctions::MathFunction, "function", "the function to evaluate"), OPT(int, "depth", "The depth in which to apply the label.  The default is zero or cell/element"),
         OPT(int, "labelValue", "The label value, default is 1"));