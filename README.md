# framework

# Running Tests Locally
The tests can be run locally using an IDE or cmake directly (ctest command).  To run the tests using the testing environment (docker), first make sure that [Docker](https://www.docker.com) installed.

```bash
# Login to github to access base image (follow prompt instructions)
docker login ghcr.io

# Build the docker testing image
docker build -t testing_image -f DockerTestFile .

# Run the built tests and view results
docker run --rm testing_image 

```

# Formatting Linting
The c/c++ code style is based upon the [Google Style Guide](https://google.github.io/styleguide/) and enforced using clang-format during PR tests.  Specific overrides to the style are controlled in the .clang-format file. A directly if clang-format is installed.

```bash
# from build directory
make format-check
```

It is recommended that an IDE (e.g. CLion) is used to check formatting during development.  Clang-format can also be used during development to directly format each file.  
```bash
clang-format -i path/to/file
```
