# Screenshots
<img width="1140" height="901" alt="image" src="https://github.com/user-attachments/assets/81f47390-5525-45dd-8045-be3986f36d7c" />


# How to run

If you want to test out the online version just go to: https://callum-op.github.io/Video-Pose-Finder/

For the local version, you will need to set up dependencies, assuming you have node installed, use: npm install 

Then to run use: npm run dev

It should be accessible locally on http://localhost:5173/

Everything runs locally in the browser using MediaPipe — there is no backend to set up.


# Limitations
Trying to track several people at the same time is unreliable, so this app can only track one person at a time, even then in a video with several people if they overlap with each other it can easily get confused. While videos with only one person or videos where one person can be seen clearly it is much more perfect at tracking them.


# Issues
Feet are not always planted firmly on the ground.

The first frame of the pose sequence appears mangled and incorrect (but only if its a video).


# Features To Consider
Have a scroller to choose when the animation they want starts and ends.

Allow gifs to be imported? Although might not be worth the memory and code required to convert it.

Could have an advanced option that lets the user move or place the skeleton where it should be if that helps the program that tracks it know where to start.

Add warning message when failed to detect multiple people.

Save pose sequences? Then the user can access them whenever they want. Would also be useful to save settings. Could just save them to local storage.

