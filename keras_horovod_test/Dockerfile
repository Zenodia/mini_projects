FROM uber/horovod:0.15.2-tf1.12.0-torch1.0.0-py3.5

# Install any needed packages specified in requirements.txt
RUN pip install scikit-image scikit-learn pillow numpy pandas matplotlib scipy

COPY keras-horovod-owndata.py /keras-horovod-owndata.py

# Make port 80 available to the world outside this container
EXPOSE 80

# debug wrapper, followed by the model training script to execute
ENTRYPOINT [ "python", "/keras-horovod-owndata.py" ]