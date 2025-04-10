# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=main.py

# Run the application
CMD ["python", "main.py"]


# docker command to build the image
## docker build -t sentiment-analysis-app .

# docker command to run the container
## docker run -p 5000:5000 sentiment-analysis-app
