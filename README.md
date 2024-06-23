The image files need to simply be put in the image folder and the images will be resized as per the settings specified 
default here is 75% size if image width or height is greater than 2500px. 
Transformers are used to analyse the image contents and provide a short description. The exif library is used to embed the description into the metadata,
sk-learn is used to pick the top 2 important words and rename the file
the output folder has the resized file and the alt folder has the image file with the embedded metadata
When imported to wordpress or any other website builder the description can be used in metadata. 
This can be done for multiple files
