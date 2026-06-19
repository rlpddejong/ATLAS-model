# Building the container for train.sh

`train.sh` runs the model inside an Apptainer (Singularity) image:

```bash
apptainer exec --nv $CONTAINER python main.py fit ...
```

where `CONTAINER` points to a `.sif` file. The `Dockerfile` defines the
environment; you just need to turn it into that `.sif` file. Three steps:

## 1. Build the Docker image

From the project root (where the `Dockerfile` is):

```bash
docker build -t atlas:latest .
```

## 2. Convert it to a `.sif` file

HPC clusters use Apptainer, not Docker, so convert the image:

```bash
# Save the Docker image to a tar archive
docker save atlas:latest -o atlas.tar

# Build the .sif from that archive
apptainer build atlas.sif docker-archive://atlas.tar
```

(If your machine has Apptainer with Docker access, you can skip the tar step:
`apptainer build atlas.sif docker-daemon://atlas:latest`.)

## 3. Point train.sh at the `.sif`

Copy `atlas.sif` to your cluster and set the path at the top of `train.sh`:

```bash
CONTAINER=/path/to/atlas.sif
```

Now `train.sh` (or `sbatch train.sh`) will run training inside the container.
The `--nv` flag gives the container access to the GPU.
