# Screenshots
<img width="1140" height="901" alt="image" src="https://github.com/user-attachments/assets/81f47390-5525-45dd-8045-be3986f36d7c" />


# How to run
To set up dependencies, assuming you have node installed, use: npm install 

Then to run use: npm run dev

It should be accessible locally on http://localhost:5173/


# Limitations
Trying to track several people at the same time is unreliable, so this app can only track one person at a time, even then in a video with several people if they overlap with each other it can easily get confused. While videos with only one person or videos where one person can be seen clearly it is much more perfect at tracking them.


# Features To Consider
Could have an advanced option that lets the user move or place the skeleton where it should be if that helps the program that tracks it know where to start.

Allow users to upload images and get indvidual poses?

Add warning message when failed to detect multiple people.

It struggles with being able to tell if the character has turned around, turning their limbs but making the hips remain still, will need to correct this if I want to use any video where the person isn't looking at the camera.

Python backend that uses Gemini AI to tidy up the pose data, filling in missing limbs, fixing innaccuracies and other tasks to ensure that the final pose sequence is complete and usable.

Save pose sequences using MongoDB? Then the user can access them whenever they want. Would also be useful to save settings.

