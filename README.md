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