import cv2 as cv


def detect(img, cascade) -> list:
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cascade', default="haarcascades/haarcascade_frontalface_alt.xml")
    parser.add_argument('--nested-cascade',
                        default="haarcascades/haarcascade_eye.xml")
    parser.add_argument('--video-src', type=int, default=0)
    args = parser.parse_args()

    cascade_fn = args.cascade
    nested_fn = args.nested_cascade

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = cv.VideoCapture(args.video_src)

    while True:
        _ret, img = cam.read()
        if not _ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))

        cv.imshow('Face Detection', vis)

        # Close the program when the user presses 'Esc'
        if cv.waitKey(5) == 27:
            break

    print('Done')
    cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    main()
