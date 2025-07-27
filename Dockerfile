# Base image with Conda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy conda environment file
COPY environment.yml .

# Create environment from YAML
RUN conda env create -f environment.yml

# Use 'llms' as the default shell environment
SHELL ["conda", "run", "-n", "llms", "/bin/bash", "-c"]

# (Optional) Install JupyterLab and extras if not already in YAML
RUN conda install -n llms -y jupyterlab && \
    pip install jupyterlab-vim

# Expose Jupyter port
EXPOSE 8888

# Default command: run JupyterLab on container start
CMD ["conda", "run", "--no-capture-output", "-n", "llms", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
