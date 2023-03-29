from moviepy.editor import VideoFileClip, clips_array

clip1 = VideoFileClip('4.mp4')
clip2 = VideoFileClip('test4.mp4')

final_clip = clips_array([[clip1],[clip2]])#上下拼接 clips_array([clip1,clip2])#左右拼接

final_clip.write_videofile('contact_4.mp4')