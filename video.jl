# Working from https://www.simonwenkel.com/2019/05/22/VideoIO-with-Julia.html

import Colors
import VideoIO
import Images
import Plots

filename = joinpath(homedir(), "Downloads/Big_Buck_Bunny_360_10s_1MB.mp4")

videoStream = VideoIO.openvideo(filename)

# Encouraging, I have some kind of array representing the first frame
frame1 = VideoIO.read(videoStream)

# Now to convert it into a regular array
# Following: https://github.com/JuliaImages/Images.jl/issues/715#issuecomment-381001495
gray1 = Colors.Gray.(frame1)
a = convert(Array{Float64}, gray1)

# Do a round trip to images and plot it for a quick sanity check
gray2 = Colors.Gray.(a)

# Works, all good.
Plots.plot(gray2)
