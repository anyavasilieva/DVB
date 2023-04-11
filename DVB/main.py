import cv2
import time

cap = cv2.VideoCapture('sample.mp4')

left_count = 0
right_count = 0
was_left = False
was_right = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ret, thresh = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    height, width = frame.shape[:2]
    left_roi = (0, 0, width // 2, height)
    right_roi = (width // 2, 0, width // 2, height)

    is_left = False
    is_right = False
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if x < width // 2 and x + w < width // 2:
            is_left = True
        elif x > width // 2 and x + w > width // 2:
            is_right = True

    if is_left and not was_left:
        left_count += 1
    elif is_right and not was_right:
        right_count += 1

    cv2.putText(frame, f"L: {left_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"R: {right_count}", (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("g"):
        break

    time.sleep(0.1)

    was_left = is_left
    was_right = is_right

cap.release()
cv2.destroyAllWindows()

