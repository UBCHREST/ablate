#include "convexPolygon.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::mathFunctions::geom::ConvexPolygon::ConvexPolygon(std::vector<std::vector<double>> points, double maxDistance, const std::shared_ptr<mathFunctions::MathFunction> &insideValues,
                                                          const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues) {
    // There must be at least three points
    if (points.size() < 3) {
        throw std::invalid_argument("ablate::mathFunctions::geom::ConvexPolygon::ConvexPolygon requires at least three points.");
    }

    // Make sure that all points are the size dimension
    std::size_t dim = points.front().size();
    for (const auto &point : points) {
        if (point.size() != dim) {
            throw std::invalid_argument("ablate::mathFunctions::geom::ConvexPolygon::ConvexPolygon requires all points to have same number of dimensions.");
        }
    }

    // March over each point to create the center
    std::vector<double> center(dim, 0.0);
    for (const auto &point : points) {
        ablate::utilities::MathUtilities::Plus(point.size(), point.data(), center.data());
    }
    ablate::utilities::MathUtilities::ScaleVector(center.size(), center.data(), 1.0 / (points.size()));

    // Create a triangle for each point
    for (std::size_t p = 0; p < points.size(); ++p) {
        std::size_t nextPoint = p + 1;
        // Wrap the last point back around
        if (nextPoint >= points.size()) {
            nextPoint = 0;
        }

        triangles.emplace_back(points[p], points[nextPoint], center, maxDistance);
    }
}

ablate::mathFunctions::geom::ConvexPolygon::ConvexPolygon(std::vector<std::shared_ptr<std::vector<double>>> points, double maxDistance,
                                                          const std::shared_ptr<mathFunctions::MathFunction> &insideValues, const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : ConvexPolygon(ablate::utilities::VectorUtilities::Copy(points), maxDistance, insideValues, outsideValues) {}

bool ablate::mathFunctions::geom::ConvexPolygon::InsideGeometry(const double *xyz, const int &ndims, const double &time) const {
    // Check to see if we are in any of the triangles
    for (const auto &triangle : triangles) {
        if (triangle.InsideGeometry(xyz, ndims, time)) {
            return true;
        }
    }
    return false;
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::ConvexPolygon, "assigns a uniform value to all points inside a cylindrical shell",
         ARG(std::vector<std::vector<double>>, "points", "the center of the cylinder start"),
         OPT(double, "maxDistance", "max distance from triangle to be considered inside.  A zero value is assumed to include all projected space"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));
