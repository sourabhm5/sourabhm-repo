# Use Miniconda3 base image
FROM continuumio/miniconda3

# Set working directory in container
WORKDIR /app

# Copy local files to container
COPY . /app

# Copy environment.yml and create conda environment
COPY environment.yml .

# Create conda env
RUN conda env create -f environment.yml

# Activate environment
SHELL ["conda", "run", "-n", "sourabhm-env", "/bin/bash", "-c"]

# Optional: install Jupyter or any CLI requirements
RUN pip install --upgrade pip

# Set environment path
ENV PATH /opt/conda/envs/sourabhm-env/bin:$PATH

# Expose Gradio default port
EXPOSE 7860

# Run the main script (change this to your actual entrypoint)
CMD ["python", "your_script.py"]
