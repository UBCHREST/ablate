#include "linearInterpolator.hpp"
#include <fstream>
#include <iostream>
#include <string>

ablate::mathFunctions::LinearInterpolator::LinearInterpolator(std::filesystem::path inputFile, std::string xAxisColumn, std::vector<std::string> yColumns,
                                                              std::shared_ptr<MathFunction> locationToXCoordFunction)
    : xColumn(xAxisColumn), yColumns(yColumns), locationToXCoordFunction(locationToXCoordFunction) {
    // open the file
    std::fstream inputFileStream;
    inputFileStream.open(inputFile, std::ios::in);
    ParseInputData(inputFileStream);
    inputFileStream.close();
}

ablate::mathFunctions::LinearInterpolator::LinearInterpolator(std::istream& inputStream, std::string xAxisColumn, std::vector<std::string> yColumns,
                                                              std::shared_ptr<MathFunction> locationToXCoordFunction)
    : xColumn(xAxisColumn), yColumns(yColumns), locationToXCoordFunction(locationToXCoordFunction) {
    ParseInputData(inputStream);
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), s.end());
}

void ablate::mathFunctions::LinearInterpolator::ParseInputData(std::istream& inputStream) {

    // determine the headers from the first row
    std::vector<std::string> headers;
    std::string headerLine;
    std::getline(inputStream, headerLine);

    // Get the headers from this stream
    std::stringstream s_stream(headerLine);
    while (s_stream.good()) {
        std::string headerColumn;
        getline(s_stream, headerColumn, ',');  // delimited by comma
        trim(headerColumn);
        headers.push_back(headerColumn);
    }

    std::cout << "here" << std::endl;


}
