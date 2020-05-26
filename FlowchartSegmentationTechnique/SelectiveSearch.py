import cv2
import numpy as np


def selective_search(img, method):
    dimensions = img.shape
    print('Dimensions: ' + str(dimensions))
    # Bild wird angepasst, damit alle Bilder das gleiche Format haben und um den Rechenaufwand zu verringern.
    newHeight = 800
    newWidth = int(img.shape[1] * 800 / img.shape[0])
    resizedImg = cv2.resize(img, (newWidth, newHeight))
    dimensions = resizedImg.shape
    print('Dimensions after resizing: ' + str(dimensions))

    # Multithreads werden benutzt, um den Prozess zu beschleunigen.
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # Selective Search wird aufgesetzt (default).
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Das Bild, welches segmentiert werden soll, wird gesetzt.
    ss.setBaseImage(resizedImg)

    # Schnellere, aber ungenauere Version.
    if method is 'f':
        ss.switchToSelectiveSearchFast()

    # Genauere, aber langsamere Version.
    elif method is 'q':
        ss.switchToSelectiveSearchQuality()
    else:
        print('methode is invalid!')
        return 0

    # Startet die Selective Search
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))

    # Filtert zu große und kleine Flächen schon vorher. Weniger Rechenaufwand für die KI später.
    rectsToDelete = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        area = w * h
        # Wenn die Fläche über 90 000 oder unter 15 000 Pixel groß ist oder die Höhe oder Breite unter 150/100 Pixel,
        # wird das Rechteck entfernt.
        if area > 90000 or area < 18000:
            rectsToDelete.append(i)

    print(len(rectsToDelete), 'gefiltert.', len(rects) - len(rectsToDelete), 'Regionen werden übergeben')

    # Regionen werden gelöscht
    for j in reversed(rectsToDelete):
        rects = np.delete(rects, j, 0)

    selectedObjects = []
    scaledBoundingBoxes = []
    hscale = img.shape[0] / resizedImg.shape[0]
    wscale = img.shape[1] / resizedImg.shape[1]

    # Übrige Regionen werden aus dem Bild ausgeschnitten und in einen Array gespeichert
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        # Regionen werden skaliert
        scaledXmin, scaledYmin, scaledXmax, scaledYmax = int(x * wscale), int(y * hscale), int((x + w) * wscale), int(
            (y + h) * hscale)
        # Skalierte BoundingBox wird für spätere Rechnungen in einem Array abgespeichert
        scaledBoundingBox = [scaledXmin, scaledYmin, scaledXmax, scaledYmax]
        scaledBoundingBoxes.append(scaledBoundingBox)
        # Skalierte Regionen werden aus dem Originalbild ausgeschnitten (nicht aus dem 800*X Pixel Bild)
        rectImg = img[scaledYmin:scaledYmax, scaledXmin:scaledXmax]
        selectedObjects.append(rectImg)

    return scaledBoundingBoxes, selectedObjects


def showPic(imgName, selectedObjects):
    numShowRects = 0
    increment = 1
    while True:
        # create a copy of original image
        imOut = selectedObjects.copy()

        # show output
        cv2.imshow(str(numShowRects) + ' / ' + str(len(selectedObjects) - 1), imOut[numShowRects])

        # record key press
        k = cv2.waitKey(0) & 0xFF

        # plus (+) is pressed
        if k == 43:
            if numShowRects < len(selectedObjects) - 1:
                numShowRects += increment
            else:
                numShowRects = 0
            cv2.destroyAllWindows()
        # plus (-) is pressed
        elif k == 45:
            # decrease total number of rectangles to show by increment
            if numShowRects != 0:
                numShowRects -= increment
            else:
                numShowRects = len(selectedObjects) - 1
            cv2.destroyAllWindows()
        # if s is pressed
        elif k == 115:
            cv2.imwrite('FlowchartData/SelectedObjects/' + imgName + '_' + str(numShowRects) + '.jpg',
                        imOut[numShowRects])
            print('Img ' + str(numShowRects) + ' gespeichert.')
        # if q is pressed
        elif k == 113:
            break
    # close image show window
    cv2.destroyAllWindows()
    return imOut


folder_name = 'FlowchartData/Images/'
img_name = '00000008'

data_name = folder_name + img_name + '.jpg'

# Bild wird eingelesen
img = cv2.imread(data_name, cv2.IMREAD_COLOR)
if img is None:
    print('img: ' + data_name + ' fail!')
    exit(0)
else:
    print('img: ' + data_name + ' read!')

# Selective Search wird durchgeführt.
scaledBoundingBoxes, selectedObjects = selective_search(img, 'f')
if scaledBoundingBoxes and selectedObjects is None:
    print('Selective Search, fail!')
    exit(0)
else:
    print('Selective Search, done!')

finalPics = showPic(img_name, selectedObjects)

if len(finalPics) == 0 or finalPics is None:
    print('Showing, fail!')
    exit(0)
else:
    print('Showing, done!')
