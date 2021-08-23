# neuralstyletransfer-backend

*Disclaimer: Copyright issues, GDPR compliance and security are not adressed in this repo (yet) - it is meant to be used as a boilerplate and not meant to end up anywhere near produciton in this state.*

# about

This repo provides the backend for [neuralstyletransfer-gui](https://github.com/maxupravitelev/neuralstyletransfer-gui). It handles the generation of a styled image based on the two images received from the frontend. 

# installation

- Download the [NST model from tensorflow_hub](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) from tensorflow hub and save it in the models folder
- Set a TF_HUB_HANDLE environment variable with the corresponding path. You can also create a .env file
- If you do not want to load the model from a local path set the TF_HUB_HANDLE to the link to the [NST model from tensorflow_hub](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)



