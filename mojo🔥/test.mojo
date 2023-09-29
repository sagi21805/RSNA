from python import Python
from tensor import Tensor

fn func(img: PythonObject, stride: Int8, thresh: Int64) raises:
    let numpy = Python.import_module("numpy")
    let blocked = numpy.lib.stride_tricks.sliding_window_view(img, stride)
    let new = numpy.zeros((blocked.shape[0], blocked.shape[1]), dtype = numpy.uint8)
    for i in range(blocked.shape[0]):
        for j in range(blocked.shape[1]):
            if numpy.sum(blocked[i][j]) > thresh:
                new[i][j] = 255
    return new

fn main() raises:
    let numpy = Python.import_module("numpy")
    let PIL = Python.import_module("PIL")
    let cv2 = Python.import_module("cv2")
    let plt = Python.import_module("matplotlib.pyplot")

    let vid = cv2.VideoCapture(0)
    let ax1 = plt.subplot()
    let im1 = ax1.imshow(vid.read()[1])

    _ = plt.ion()
    while True:
        var img = vid.read()[1]
        img = PIL.Image.fromarray(img) 
        let gray_img = img.convert("L")
        _ = im1.set_data(gray_img)
        _ = plt.pause(0.001)
        print(gray_img.size)

