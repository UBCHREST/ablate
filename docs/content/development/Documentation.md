---
layout: default
title: Documentation
parent: Code Development
nav_order: 4
---

Documentation is built using a series of static html and markdown files in the doc folder and available at [ablate.dev](https://ablate.dev). [Jekyll](https://jekyllrb.com) is used to compile the documents into a static site that is published upon commit.  You can test your changed locally following this [GitHub Guide](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll) or with the supplied docker file.  Math and equations are rendered using [MathJax](https://www.mathjax.org) using Latex style equations where $$ is used to define math regions.
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
An ablate.dev preview can be built locally without ruby using Docker and the DockerDocsFile file.  This system requires rebuilding the docker image after every change and should only be used when the [GitHub Guide](https://docs.github.com/en/free-pro-team@latest/github/working-with-github-pages/testing-your-github-pages-site-locally-with-jekyll) can not be followed or if the automatically configured files (Component List, Example Input Files, Source Documentation, etc.) need to be preview. 

```bash
# 0. Install docker on to your machine

# 1. cd to the root of the repository

# 2. Build the docker testing image
docker build -t docs_image -f DockerDocsFile .

# 3. Run the docs server
docker run -p 4000:4000 --rm docs_image 

```
Once the server is up and running preview the documentation at <http://localhost:4000>.

### Pull Request Preview
After a PR has been made and the automated testing is complete an artifact is generated of the new ablate.dev files. This can be previewed locally by:

1. Download and extract the artifacts for the associated PR.
2. Local a local web server such as python http.server
   ```bash
   cd artifact
   
   # for python 3
   python3 -m http.server 8000
   
   # for python 2
   python -m SimpleHTTPServer 8000
   ```
3. Load the local preview website at <http://localhost:8000>