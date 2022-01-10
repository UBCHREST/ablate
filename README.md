# ABLATE Framework
**A**blative **B**oundary **L**ayers **A**t **T**he **E**xascale = **ABLATE**

ABLATE is a [UB CHREST](https://www.buffalo.edu/chrest.html) project focused on leveraging advances in both exascale computing and machine learning to better understand the turbulent mixing and fuel entrainment in the combustion environment that is critical to the operation of hybrid rocket motors.


## [Documentation](https://ablate.dev)
Current documentation can be found online at [ABLATE Documentation](https://ablate.dev) along with a [Getting Started Guide](http://ablate.dev/content/GettingStarted.html)

Documentation is built using a series of static html and markdown files in the doc folder. [Jekyll](https://jekyllrb.com) is used to compile the documents into a static site that is published upon commit.  You can test your changed locally following this [GitHub Guide](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll) or with the supplied docker file.  Math and equations are rendered using [MathJax](https://www.mathjax.org) using Latex style equations where $$ is used to define math regions.
```markdown
This line would include $$x=y^2$$ and other text.

This is a standalone equation
$$\begin{eqnarray}
y &=& x^4 + 4      \nonumber \\
&=& (x^2+2)^2 -4x^2 \nonumber \\
&\le&(x^2+2)^2    \nonumber
\end{eqnarray}$$

```

### Local Docs Preview
```bash
# To preview the docs locally
# 0. Install docker on to your machine
# 1. cd to the root of the repository

# 2. Build the docker testing image
docker build -t docs_image -f DockerDocsFile .

# 3. Run the docs server
docker run -v $PWD/docs:/docs -p 4000:4000 --rm docs_image 

# 4. View the docs url at localhost:4000/ablate/

```

## Running Tests Locally
The tests can be run locally using an IDE or cmake directly (ctest command).  You may also use the ```--keepOutputFile=true```  command line argument to preserve output files.  To run the tests using the testing environment (docker), first make sure that [Docker](https://www.docker.com) installed.

```bash
# Build the docker testing image
docker build -t testing_image -f DockerTestFile .

# or for 64 bit ints
docker build -t testing_image --build-arg PETSC_BUILD_ARCH='arch-ablate-opt-64' -f DockerTestFile .

# Run the built tests and view results
docker run --rm testing_image 

```

For output file comparisons you can specify that numbers are '<', '>', or '=' to an expected value.  For example: 

```
L_2 Error: [(.*), (.*), (.*)]<expects> <1E-13 <1E-13 <1E-13
```
would expect three numbers (all less than 1E-13),

```
L_2 Residual: (.*)<expects> <1E-13
```
one value less than 1E-13,

```
Taylor approximation converging at order (.*)<expects> =2
```
and one value equal to 2.  When using the compare tool, you must escape all regex characters on the line being compared. 

There are some useful command line flags that can be used when debugging tests:
- --runMpiTestDirectly=true : when passed in (along with google test single test selection or through CLion run configuration) this flag allows for a test to be run/debug directly.  This bypasses the separate process launch making it easier to debug, but you must directly pass in any needed arguments. 
- --keepOutputFile=true : keeps all output files from the tests and reports the file name.

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

## Acknowledgements
This research is funded by the United States Department of Energy’s (DoE) National Nuclear Security Administration
(NNSA) under the Predictive Science Academic Alliance Program III (PSAAP III) at the University at Buffalo, under
contract number DE-NA000396