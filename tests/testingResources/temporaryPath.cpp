#include "temporaryPath.hpp"
#include <fstream>
#include <sstream>
#include <random>

testingResources::TemporaryPath::TemporaryPath() : path(std::filesystem::temp_directory_path()/RandomString(10)) {}

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
std::string testingResources::TemporaryPath::RandomString(std::size_t length) {
    std::mt19937 rng(time(nullptr));
    std::uniform_int_distribution<int> generator(0, charSet.size() - 1);
    auto randomChar = [&rng, &generator]() -> char {
        return charSet[generator(rng)];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randomChar);
    return str;
}
