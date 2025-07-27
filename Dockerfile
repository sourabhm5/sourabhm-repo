FROM continuumiominiconda3

# Set working directory
WORKDIR workspace

# Copy environment definition
COPY environment.yml .

# Create the 'llms' environment
RUN conda env create -f environment.yml && conda clean -a -y

# Use 'llms' as the default shell environment
SHELL [conda, run, -n, llms, binbash, -c]

# Install optional extras (if not in your yml)
RUN conda run -n llms pip install --no-cache-dir gradio jupyterlab

# Install VSCode Server (optional for browser-based code editing)
RUN curl -fsSL httpscode-server.devinstall.sh  sh

# Expose useful ports
EXPOSE 8888 7860 8080

# Start JupyterLab by default
CMD [conda, run, -n, llms, jupyter, lab, --ip=0.0.0.0, --port=8888, --no-browser, --allow-root]
