#include "linearTable.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

ablate::mathFunctions::LinearTable::LinearTable(std::filesystem::path inputFile, std::string xAxisColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoordFunction)
    : independentColumnName(xAxisColumn), dependentColumnsNames(yColumns), independentValueFunction(locationToXCoordFunction) {
    // open the file
    std::fstream inputFileStream;
    inputFileStream.open(inputFile, std::ios::in);
    ParseInputData(inputFileStream);
    inputFileStream.close();
}

ablate::mathFunctions::LinearTable::LinearTable(std::istream& inputStream, std::string xAxisColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoordFunction)
    : independentColumnName(xAxisColumn), dependentColumnsNames(yColumns), independentValueFunction(locationToXCoordFunction) {
    ParseInputData(inputStream);
}

// trim from both ends (in place)
static inline void trim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

void ablate::mathFunctions::LinearTable::ParseInputData(std::istream& inputStream) {
    // determine the headers from the first row
    std::vector<std::string> headers;
    std::string line;
    std::getline(inputStream, line);

    // Get the headers from this stream
    std::stringstream headerStream(line);
    while (headerStream.good()) {
        std::string headerColumn;
        getline(headerStream, headerColumn, ',');  // delimited by comma
        trim(headerColumn);
        headers.push_back(headerColumn);
    }

    // record the column index for each requested value
    auto xIndexIt = std::find(headers.begin(), headers.end(), independentColumnName);
    if (xIndexIt == headers.end()) {
        throw std::invalid_argument("Cannot locate column " + independentColumnName);
    }
    auto xIndex = std::distance(headers.begin(), xIndexIt);

    // repeat for each y column
    std::vector<std::size_t> yIndexes;
    for (const auto& yColumnName : dependentColumnsNames) {
        auto yIndexIt = std::find(headers.begin(), headers.end(), yColumnName);
        if (yIndexIt == headers.end()) {
            throw std::invalid_argument("Cannot locate column " + yColumnName);
        }
        yIndexes.push_back(std::distance(headers.begin(), yIndexIt));
    }

    // size up a double array to hold the values
    std::vector<double> rowValues(headers.size());
    dependentValues.resize(yIndexes.size());

    // Now parse each line
    while (std::getline(inputStream, line)) {
        std::istringstream lineStream(line);
        for (std::size_t c = 0; c < rowValues.size(); c++) {
            std::string lineNumber;
            getline(lineStream, lineNumber, ',');  // delimited by comma
            rowValues[c] = std::stod(lineNumber);
        }

        // now extract the values and place in the columns/xvalues
        independentValues.push_back(rowValues[xIndex]);
        for (std::size_t v = 0; v < yIndexes.size(); v++) {
            dependentValues[v].push_back(rowValues[yIndexes[v]]);
        }
    }
}
void ablate::mathFunctions::LinearTable::Interpolate(double x, size_t numInterpolations, double* result) const {
    // Determine the upper index
    std::size_t upIndex = 0;
    for (upIndex = 1; upIndex < independentValues.size() - 1; upIndex++) {
        // If the inquiry value is less then this, we have the right index
        if (x < independentValues[upIndex]) {
            break;
        }
    }

    // We need the x-x0 and deltaX
    double x_x0 = x - independentValues[upIndex - 1];
    double deltaX = independentValues[upIndex] - independentValues[upIndex - 1];

    // Bound the interpolation to the first and last values, i.e. don't extrapolate
    if (x < independentValues[upIndex - 1]) {
        x_x0 = 0.0;
    }
    if (x > independentValues[upIndex]) {
        x_x0 = deltaX;
    }

    // Do the linear interpolation for each variable
    for (std::size_t s = 0; s < numInterpolations; s++) {
        result[s] = dependentValues[s][upIndex - 1] + (x_x0 / deltaX) * (dependentValues[s][upIndex] - dependentValues[s][upIndex - 1]);
    }
}
double ablate::mathFunctions::LinearTable::Eval(const double& x, const double& y, const double& z, const double& t) const {
    double independentValue = independentValueFunction->Eval(x, y, z, t);
    double result;
    Interpolate(independentValue, 1, &result);
    return result;
}
double ablate::mathFunctions::LinearTable::Eval(const double* xyz, const int& ndims, const double& t) const {
    double independentValue = independentValueFunction->Eval(xyz, ndims, t);
    double result;
    Interpolate(independentValue, 1, &result);
    return result;
}
void ablate::mathFunctions::LinearTable::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    double independentValue = independentValueFunction->Eval(x, y, z, t);
    Interpolate(independentValue, result.size() < dependentValues.size() ? result.size() : dependentValues.size(), &result[0]);
}
void ablate::mathFunctions::LinearTable::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    double independentValue = independentValueFunction->Eval(xyz, ndims, t);
    Interpolate(independentValue, result.size() < dependentValues.size() ? result.size() : dependentValues.size(), &result[0]);
}
PetscErrorCode ablate::mathFunctions::LinearTable::LinearInterpolatorPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt nf, PetscScalar* u, void* ctx) {
    // wrap in try, so we return petsc error code instead of c++ exception
    PetscFunctionBeginUser;
    try {
        auto table = (LinearTable*)ctx;

        double independentValue = table->independentValueFunction->Eval(x, dim, time);
        table->Interpolate(independentValue, nf < (PetscInt)table->dependentValues.size() ? nf : table->dependentValues.size(), u);

    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::LinearTable,
         "A table that is built from a spreadsheet that allows linear interpolation of variables based on monotonically increasing independent variables",
         ARG(std::filesystem::path, "file", "a file with csv data and header"), ARG(std::string, "independent", "the name of the independent column name as defined in the header"),
         ARG(std::vector<std::string>, "dependent", "the names of the dependent column in the order in which to apply them"),
         ARG(ablate::mathFunctions::MathFunction, "mappingFunction", " the function that maps from the physical x,y,z, and t space to the table independent variable"));