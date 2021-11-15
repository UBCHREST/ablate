#include "createLabel.hpp"
#include <utilities/petscError.hpp>

ablate::domain::modifiers::CreateLabel::CreateLabel(std::shared_ptr<domain::Region> region, std::shared_ptr<mathFunctions::MathFunction> function, int dmDepth)
    : region(region), function(function), dmHeight((PetscInt)dmDepth) {}

void ablate::domain::modifiers::CreateLabel::Modify(DM &dm) {
    DMCreateLabel(dm, region->GetName().c_str()) >> checkError;

    DMLabel newLabel;
    DMGetLabel(dm, region->GetName().c_str(), &newLabel) >> checkError;

    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, dmHeight, &cStart, &cEnd) >> checkError;

    // Get the number of dimensions from the dm
    PetscInt nDims;
    DMGetCoordinateDim(dm, &nDims) >> checkError;

    // get the region value
    const auto labelValue = region->GetValue();

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
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::CreateLabel, "Creates a new label for all positive points in the function",
         ARG(domain::Region, "region", "the region describing the new label"), ARG(mathFunctions::MathFunction, "function", "the function to evaluate"),
         OPT(int, "depth", "The depth in which to apply the label.  The default is zero or cell/element"));