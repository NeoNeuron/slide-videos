from moviepy.editor import *

fname = "video_raw.mp4"
video = VideoFileClip(fname)
# video = video.subclip(0,video.duration)
start = 0
end = 100
video = video.subclip(start, end)

video.to_videofile('video_cliped.mp4')
