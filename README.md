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

Even before AI gets involved it might be worth seeing could the raw pose data be exported as something that can be used in csp or is the backend required for that? It might be possible to export a raw untidy version of the data in JavaScript. It also means that I could test the data, before and after.

Either move the select a person section and even the stats just above upload file so the user can see it immediately or maybe simply make the browser automatically scroll down to select person or stats section once loaded.

