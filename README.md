# Screenshots
<img width="1140" height="901" alt="image" src="https://github.com/user-attachments/assets/81f47390-5525-45dd-8045-be3986f36d7c" />


# How to run
To set up dependencies, assuming you have node installed, use: npm install 

Then to run use: npm run dev

It should be accessible locally on http://localhost:5173/


# Limitations
Trying to track several people at the same time is unreliable, so this app can only track one person at a time, even then in a video with several people if they overlap with each other it can easily get confused. While videos with only one person or videos where one person can be seen clearly it is much more perfect at tracking them.

It struggles with being able to tell if the character has turned around, turning their limbs but making the hips remain still facing forward, resulting in the body being twisted in ways that are impossible. Similarily if the camera is movinf around the character it will result in the same issues, so far it assumes the character is facing the camera.


# Features To Consider
Could have an advanced option that lets the user move or place the skeleton where it should be if that helps the program that tracks it know where to start.

Add warning message when failed to detect multiple people.

Python backend that uses Gemini AI to tell which orientation the people in the video are facing as well as how many people there are, dealing with both the current limitations of the app at once.

A prompt for this could look like:
  Prompt: "You are analysing a single frame from a motion capture video for a pose
    estimation pipeline. Your only task is to determine the orientation of
    the person's body relative to the camera. Analyze the frame and return ONLY a JSON object, no explanation:
    {
    "yaw_degrees": <number, -180 to 180, where 0 = facing directly toward
    camera, 90 = person's left side facing camera,
    -90 = person's right side facing camera,
    180 or -180 = facing directly away from camera>,
    "confidence": <number 0.0 to 1.0, lower if person is partially occluded,
    blurry, or if orientation is ambiguous>,
    "view": <"front" | "side_left" | "side_right" | "rear_3q_left" |
    "rear_3q_right" | "rear">,
    "notes": <string, only if confidence < 0.6, briefly explain why>
    }Classification boundaries:
    front:         |yaw| < 45°
    side:          45° – 135°
    rear_3q:       135° – 160°
    rear:          |yaw| > 160° 
    Do not describe the image. Return only the JSON object.

Could also use Gemini AI to tidy up the pose data, filling in missing limbs, fixing inaccuracies and other tasks to ensure that the final pose sequence is complete and usable.

Save pose sequences using MongoDB? Then the user can access them whenever they want. Would also be useful to save settings.

