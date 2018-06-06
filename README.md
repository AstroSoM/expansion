# README for the Expansion of the Universe AstroSoM (June 2018)

# Generate the frames for the audio
>> python risset.py

# Generate the frames for the video
>> python expansion.py

# Stitch the frames together into a video
>> ffmpeg -f image2 -framerate 12 -i 'cmb_%06d.png' -s 1920X1080 -pix_fmt yuv420p cmb.mp4
>> ffmpeg -f image2 -framerate 12 -i 'expansion_%06d.png' -s 1920X1080 -pix_fmt yuv420p expansion.mp4

# Combine the video and audio
>> ffmpeg -i expansion.mp4 -i risset.wav -c:v copy -c:a aac -strict experimental -shortest risset-expansion.mp4

Notes:
------
Including "-pix_fmt yuv420p" is needed for the output to play with QuickTime and most other players.

Including "-strict experimental" allows us to bypass the 8000 Hz sample rate and mono encoding.

Including "-shortest" when combining an audio and video file truncates the final result to be as long whichever is shortest.
