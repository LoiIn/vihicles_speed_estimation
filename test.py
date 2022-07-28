import cv2 

path = "D:\\GR\\data\\videos\\finally\\u\\"

cap = cv2.VideoCapture(path + "vu1.mp4")
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(video_width)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
_h = 720
_w = 920

while True:
    _, frame = cap.read()
    try:
        cv2.line(frame, (0, 750), (int(video_width), 750), (0,0,255), 2)
        cv2.line(frame, (0, int(video_height)), (int(video_width), int(video_height)), (0,0,255), 2)
        frame = cv2.resize(frame, (_w, _h))
        cv2.imshow("FRAME",frame)
        if cv2.waitKey(32) & 0xFF == 27:
            break
    except:
        print("error")
        break

cap.release()
cv2.destroyAllWindows()