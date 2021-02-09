# ABLATE Framework
**A**blative **B**oundary **L**ayers **A**t **T**he **E**xascale = **ABLATE**

ABLATE is a [UB CHREST](https://www.buffalo.edu/chrest.html) project focused on leveraging advances in both exascale computing and machine learning to better understand the turbulent mixing and fuel entrainment in the combustion environment that is critical to the operation of hybrid rocket motors.


## [Documentation](https://ubchrest.github.io/chrest/)
Current documentation can be found online at [ABLATE Documentation](https://ubchrest.github.io/ablate/).

Documentation is built using a series of static html and markdown files in the doc folder. [Jekyll](https://jekyllrb.com) is used to compile the documents into a static site that is published upon commit.  You can test your changed locally following this [GitHub Guide](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll).  Math/equation is rendered using [MathJax](https://www.mathjax.org) using Latex style equations where $$ is used to define math regions.
```markdown
This line would include $$x=y^2$$ and other text.

This is a standalone equation
$$\begin{eqnarray}
y &=& x^4 + 4      \nonumber \\
&=& (x^2+2)^2 -4x^2 \nonumber \\
&\le&(x^2+2)^2    \nonumber
\end{eqnarray}$$

```

## Running Tests Locally
The tests can be run locally using an IDE or cmake directly (ctest command).  You may also use the ```--keepOutputFile=true```  command line argument to preserve output files.  To run the tests using the testing environment (docker), first make sure that [Docker](https://www.docker.com) installed.

```bash
# Login to github to access base image (follow prompt instructions)
docker login ghcr.io

# Build the docker testing image
docker build -t testing_image -f DockerTestFile .

# Run the built tests and view results
docker run --rm testing_image 

```

## Formatting Linting
The c++ code style is based upon the [Google Style Guide](https://google.github.io/styleguide/) and enforced using clang-format during PR tests.  Specific overrides to the style are controlled in the .clang-format file.

All c code should be styled using the [PETSc Style and Usage Guide](https://docs.petsc.org/en/latest/developers/style/) and enforced using a shell script based upon PETSc.

```bash
# To run a format check
# from build directory
make format-check
```

It is recommended that an IDE (e.g. CLion) is used to check formatting during development.  Clang-format can also be used during  development to directly format each file.  
```bash
clang-format -i path/to/file
```
