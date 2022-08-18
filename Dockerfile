FROM python:3.8.13

# Copy all files into new folder directory
COPY . ./gender_prediction/

# Move to new folder
WORKDIR ./gender_prediction/

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run app
CMD [ "python", "serve.py", "-d"]