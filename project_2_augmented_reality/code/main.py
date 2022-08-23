import cv2
import math
from functions import*
from cube import*
import time

#modes
write_to_video = True
show_contours = False
img_mode = True
Cube_mode = True
Fast_mode = False 
Fast_warp = False 
Smooth_mode = True

# Video select
video_src = 4 
start_frame = 1 

# Cube settings
face_colors = [(0, 127, 255),(255, 127, 0),(0, 191, 0)] 
edge_color = (0, 0, 0) 


if write_to_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    today = time.strftime("%m-%d__%H.%M.%S")
    videoname=str(today)+('_contours' if show_contours == True else '')+("_img" if img_mode == True else '')+('_cube' if Cube_mode == True else '')+('_smooth' if Smooth_mode == True else '')+str(video_src)
    fps_out = 29
    out = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))
    print("Writing to Video, Please Wait")

K=np.array([[1406.08415449821,0,0],
           [2.20679787308599, 1417.99930662800,0],
           [1014.13643417416, 566.347754321696,1]])

# define the tags
tag_ids = ['0101','0111','1111']
img_paths = ['data/chika.jpg','data/mami.jpg','data/lali.jpg']  

imgs = []
for path in img_paths:
    imgs.append(cv2.imread(path))


video = cv2.VideoCapture('data/data_'+str(video_src)+'.mp4') 


video.set(1,start_frame)
count = start_frame

while(video.isOpened()):
    if not Fast_mode:
        print("Current frame:" + str(count))
    count += 1
    ret, frame = video.read() 
    if ret:
        
        [all_cnts,cnts] = findcontours(frame,120) # for video_3 = 180 , for video_4 = 120
        
        
        [tag_cnts,corners] = approx_quad(cnts)
        if show_contours == True:
            cv2.drawContours(frame,all_cnts,-1,(0,255,0), 4)
            cv2.drawContours(frame,tag_cnts,-1,(255,0,0), 4)

        for i,tag in enumerate(corners):
            
            num_points = num_points_in_poly(frame,tag_cnts[i])

            dim = int(math.sqrt(num_points))
   
            H = homography(tag,dim)
            H_inv = np.linalg.inv(H)
            if Fast_warp:
                square_img = fastwarp(H_inv,frame,dim,dim)
            else:
                square_img = warp(H_inv,frame,dim,dim)

            
            imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
            ret, square_img = cv2.threshold(imgray, 70, 210, cv2.THRESH_BINARY) #  for video 3 = (imgray, 180, 255, cv2.THRESH_BINARY) for video 4 = (imgray, 70, 210, cv2.THRESH_BINARY)

            
            [tag_img,id_str,orientation] = encode_tag(square_img)
        
            if img_mode:
                 
                if id_str in tag_ids:
                    index = tag_ids.index(id_str)
                    new_img = imgs[index]
                else:
                    continue

                
                rotated_img = rotate_img(new_img,orientation)

                dim = rotated_img.shape[0]
                H = homography(tag,dim)
                h = frame.shape[0] 
                w = frame.shape[1]
                if Fast_warp:
                    frame1 = fastwarp(H,rotated_img,h,w)
                else:
                    frame1 = warp(H,rotated_img,h,w)

                frame2 = blank_region(frame,tag_cnts[i],0)
                frame = cv2.bitwise_or(frame1,frame2)

            if Cube_mode:
                if id_str in tag_ids:
                    index = tag_ids.index(id_str)
                    face_color = face_colors[index]
                else:
                    continue
  
                H=homography(tag,200)
                H_inv = np.linalg.inv(H)
                P=projection_mat(K,H_inv)
                new_corners=cubePoints(tag, H, P, 200)

                frame=drawCube(tag, new_corners,frame,face_color,edge_color,img_mode)

        if not Fast_mode:
            cv2.imshow("Superposition",frame)

        if write_to_video:
            out.write(frame)
    else:

        video.release()


    if cv2.waitKey(1) == ord('q'):
        video.release()

if write_to_video:
    out.release()