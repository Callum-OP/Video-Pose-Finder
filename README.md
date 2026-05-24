# Screenshots
<img width="1140" height="901" alt="image" src="https://github.com/user-attachments/assets/81f47390-5525-45dd-8045-be3986f36d7c" />


# How to run
To set up dependencies, assuming you have node installed, use: npm install 

Then to run use: npm run dev

It should be accessible locally on http://localhost:5173/

# Limitations
Trying to track several people at the same time is unreliable, so this app can only track one person at a time, even then in a video with several people if they overlap with each other it can easily get confused. While videos with only one person or videos where one person can be seen clearly it is much more perfect at tracking them.



The person identifier as in the this guy is person1 and this guy is person2 is too inaccurate, it would be better to have no identifier and just where it found a person and then it can track them from there if the user selects them, that way there is no confusion.
Actually its not as bad as I thought.

Could add a toggle or popup question: is there more than one person in this video?
If not then straight to processing as it normally would, if so then give the user otion to select a person from any frame.

Could add a desc like: A prescan has attempted to find all people in the video. Select the person you would like to track, it does not matter what frame you select them on, it will attempt to track them from the start of the video.

Could have an advanced option that lets the user move or place the skeleton where it should be if that helps the program that tracks it know where to start.

Even before AI gets involved it might be worth seeing could the raw pose data be exported as something that can be used in csp or is the backend required for that?

The select a person section should be higher just above where you upload file.

Allow a higher number of frames like up to 1000

