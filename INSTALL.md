```
                                        Inverse Problem Python library
```
```
         __        ______  _______   _______   __      __  __  __  __     ______       _____  
        /  |      /      |/       \ /       \ /  \    /  |/  |/  |/  |    \    \      /    /     
        $$ |____  $$$$$$/ $$$$$$$  |$$$$$$$  |$$  \  /$$/ $$ |$$/ $$ |____ \ $$ \    / $$ /   
        $$      \   $$ |  $$ |__$$ |$$ |__$$ | $$  \/$$/  $$ |/  |$$      \ \ $$ \  / $$ /    
        $$$$$$$  |  $$ |  $$    $$/ $$    $$/   $$  $$/   $$ |$$ |$$$$$$$  | | $$$$$$$$ /     
        $$ |  $$ |  $$ |  $$$$$$$/  $$$$$$$/     $$$$/    $$ |$$ |$$ |  $$ | | $$$$$$$$ |     
        $$ |  $$ | _$$ |_ $$ |      $$ |          $$ |    $$ |$$ |$$ |__$$ |/ $$ /  \ $$ \    
        $$ |  $$ |/ $$   |$$ |      $$ |          $$ |    $$ |$$ |$$    $$/  $$ /    \ $$ \   
        $$/   $$/ $$$$$$/ $$/       $$/           $$/     $$/ $$/ $$$$$$$/_ $$_/      \_$$_\  

```
```
                                        https://hippylib.github.io
```

`hIPPYlibx` depends on [FEniCSx](https://fenicsproject.org/) version 0.8.0 released in April 2024.

`FEniCSx` needs to be built with the following dependencies enabled:
* `numpy`, `scipy`, `matplotlib`, `mpi4py`
* `petsc4py` (version 3.10.0 or above)
* `slepc4py` (version 3.10.0 or above)

# FEniCSx installation
All the  methods to install `FEniCSx` are given on the FEniCSx [installation](https://github.com/FEniCS/dolfinx#installation) page.
We recommend using their prebuilt `Docker` images.
## Run FEniCSx from Docker (Linux, MacOS, Windows)
First you will need to install [Docker](https://www.docker.com/) on your system. MacOS and Windows users should preferably use Docker for Mac or Docker for Windows --- if it is compatible with their system --- instead of the legacy version Docker Toolbox.

For `FEniCSx` version 0.8.0, the Docker image is pulled using 
```
docker pull dolfinx/dolfinx:v0.8.0
```
The above command can be specified in a [`Dockerfile`](https://github.com/hippylib/hippylibx/blob/main/Dockerfile) as in the `hIPPYlibx` directory.

Once the `Dockerfile` is built, it can be run using the `docker run` command. The present  working directory can be shared when running the docker image which allows for ease in navigating the `hIPPYlibx` directory inside the Docker container.

For instance the docker image can be built and run using the following commands:
```
docker build -t hIPPYlibx_image:v1 .
```

```
docker run -it -v $(pwd):/home hIPPYlibx_image:v1
```
### Other ways to build FEniCSx
For instructions on other ways to build `FEniCSx`, we refer to the FEniCSx project [download page](https://github.com/FEniCS/dolfinx#installation). Note that this instructions always refer to the latest version of FEniCSx which may or may not be yet supported by hIPPYlibx. Always check the hIPPYlibx website for supported FEniCSx versions.