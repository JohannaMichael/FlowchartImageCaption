import glob
import cv2
import numpy as np
import re


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
        if area > 90000 or area < 18000 or w < 50 or h < 80 or w > 300 or h > 300:
            rectsToDelete.append(i)

    print(len(rectsToDelete), ' filtered. ', len(rects) - len(rectsToDelete), ' regions are passed on.')

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


# Gibt True an, wenn sich zwei Bounding-Boxes zu viel überschneiden (da wahrscheinlich selbes Motiv)
def intersection(boxA, boxB):
    # Punkte oben rechts und unten links werden errechnet
    # boxA = [rectA[0], rectA[1], rectA[0] + rectA[2], rectA[1]+rectA[3]]
    # boxB = [rectB[0], rectB[1], rectB[0] + rectB[2], rectB[1] + rectB[3]]

    # Speichert die Koordinaten der Überschneidungsfläche (die Koordinaten des Punktes links oben und des Punktes rechts unten)
    interXmin = max(boxA[0], boxB[0])
    interYmin = max(boxA[1], boxB[1])
    interXmax = min(boxA[2], boxB[2])
    interYmax = min(boxA[3], boxB[3])

    # Berechnet die Überschneidungsfläche
    interArea = max(0, interXmax - interXmin) * max(0, interYmax - interYmin)

    # Die Flächen der Boxen werden berechnet
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # print('BoxAArea: ' + str(boxAArea) + ' / BoxBArea: ' + str(boxAArea))

    # Berechnet die größe des Anteils der Überschneidungsfläche an den beiden Bounding Boxes
    ioboxA = interArea / boxAArea
    ioboxB = interArea / boxBArea

    # Gibt den kleineren Anteil zurück
    minInter = min(ioboxA, ioboxB)

    # Gibt True an, wenn eines der beiden Bounding Boxes zu über 80% in der anderen liegt,
    # der Wert bei keiner der beiden jedoch unter 50% ist
    return (ioboxA > 0.8 or ioboxB > 0.8) and minInter > 0.7


def deleteIntersectedObjects(scaledBoundingBoxes, selectedObjects, imgNumber):
    i = 0
    numberOfIntersections = 0
    arraylen = len(scaledBoundingBoxes)
    while i < arraylen:
        j = i + 1
        while j < arraylen:
            if intersection(scaledBoundingBoxes[i], scaledBoundingBoxes[j]):
                numberOfIntersections += 1
                scaledBoundingBoxes.pop(j)
                selectedObjects.pop(j)
                j -= 1
                arraylen -= 1
                break
            j += 1
        i += 1
    print('Number of objects selected that did not intersect each other: ' + str(len(selectedObjects)))
    saveSelectedObject(selectedObjects, imgNumber)
    return selectedObjects


def saveSelectedObject(selectedObjects, imgNumber):
    selectedObjectNum = 0
    for image in selectedObjects:
        selectedObjectNum += 1
        cv2.imwrite('FlowchartData/SelectedObjects/' + str(imgNumber) + '_' + str(selectedObjectNum) + '.jpg',
                    image)


# Below path contains all the images
flowcharts_folder_name = './FlowchartData/Images/'
# Create a list of all image names in the directory
flowchart_path_list = glob.glob(flowcharts_folder_name + '*.jpg')

for path in flowchart_path_list:
    path_number = int(re.search(r'\d+', path).group())
    if path_number > 303:
        # print(path)
        # print(int(re.search(r'\d+', path).group()))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print('img: ' + path + ' fail!')
            exit(0)
        else:
            print('img: ' + path + ' read!')

        scaledBoundingBoxes, selectedObjects = selective_search(img, 'f')
        if scaledBoundingBoxes and selectedObjects is None:
            print('Selective Search, fail!')
            exit(0)
        else:
            print('Selective Search, done!')

        img_number = int(re.search(r'\d+', path).group())
        reducedSelectedObjects = deleteIntersectedObjects(scaledBoundingBoxes, selectedObjects, img_number)
        # finalPics = showPic(img_name, reducedSelectedObjects)
