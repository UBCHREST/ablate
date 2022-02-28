#include "temporaryPath.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>

testingResources::TemporaryPath::TemporaryPath() : path(std::tmpnam(nullptr)) {}

testingResources::TemporaryPath::~TemporaryPath() {
    if (exists(path)) {
        std::filesystem::remove_all(path);
    }
}

std::string testingResources::TemporaryPath::ReadFile() const {
    std::ifstream stream(path);
    std::stringstream buffer;
    buffer << stream.rdbuf();

    return buffer.str();
}
