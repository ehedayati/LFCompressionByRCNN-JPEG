import datetime
import tensorflow as tf


def print_results(step,datasetLength,timeSpent,duration,stepDuration,ETA,loss,metrics,meanMetrics):
    printable = ''
    for key in meanMetrics.keys():
        if not tf.math.is_nan(metrics[key]):
            meanMetrics[key](metrics[key])
    for key in meanMetrics.keys():
        printable += key +': {:.5f} '.format(meanMetrics[key].result().numpy()) + ' '
    print('\r',
        'step {:d}/{:d}: timeSpent: {:} ETA: {:} stepTime: {:.5f} stepTimeAVG: {:.5f}, Loss {:.6f}'.format(
        step, datasetLength, str(datetime.timedelta(seconds=timeSpent)), str(datetime.timedelta(seconds=ETA)) ,stepDuration,duration.result(), loss), printable
        , end='')

def logger(loggingDict,file_writer,currStep):
    with file_writer.as_default():
        for key in loggingDict.keys():
            tf.summary.scalar(key, loggingDict[key].result(), step=currStep)

@tf.function()
def normalize_lf(lf):
    return 2.0*(lf-0.5)

@tf.function()
def lfReadTest(name):
    valueLF = tf.io.read_file(name)
    lf = tf.image.decode_image(valueLF, channels=3)
    lfShape = tf.shape(lf)
    orgSize = [lfShape[0]//angularDim[0],lfShape[1]//angularDim[1]]
    lf = tf.transpose(tf.reshape(
        lf, [orgSize[0], angularDim[0], orgSize[1], angularDim[1], 3]),
                      perm=[0, 2, 1, 3, 4])
    lfList = []
    lf = tf.image.convert_image_dtype(lf,tf.float32)
    lf.set_shape([spatialDim[0],spatialDim[1],7,7,3])
    cList,lfList = lfRandomCrop(lf,CropNo,orgSize)
    return cList, lfList

def test_pipeline(lf_filenames,batch_size,prefetch_size,shuffle_buffer):
    lf_names = tf.data.Dataset.from_tensor_slices(lf_filenames).shuffle(shuffle_buffer)
    ds = lf_names.map(lfReadTest, num_parallel_calls=4).unbatch()
    return ds.batch(batch_size, drop_remainder=True).cache('testcache')

@tf.function()
def lfRandomCrop(lf,cropNo,orgSize):
    lfCropList = []
    centerCropList = []
    c = tf.random.uniform(
        shape=[cropNo],maxval=orgSize[0]-spatialDim[0],dtype=tf.dtypes.int32)
    r = tf.random.uniform(
        shape=[cropNo],maxval=orgSize[1]-spatialDim[1],dtype=tf.dtypes.int32)
    for i in range(cropNo):
        crop = lf[c[i]:c[i]+spatialDim[0],r[i]:r[i]+spatialDim[1],:,:,:]
        center = tf.image.adjust_jpeg_quality(crop[:,:,3,3,:],50)
        center.set_shape([spatialDim[0],spatialDim[1],3])
        crop.set_shape([spatialDim[0],spatialDim[1],7,7,3])
        lfCropList.append(normalize_lf(crop))
        centerCropList.append(normalize_lf(center))
    return centerCropList,lfCropList


@tf.function()
def lfReadAug(name):
    r = tf.random.uniform([1],0.,1.)
    valueLF = tf.io.read_file(name)
    lf = tf.image.decode_image(valueLF, channels=3)
    lfShape = tf.shape(lf)
    orgSize = [lfShape[0]//angularDim[0],lfShape[1]//angularDim[1]]
    lfShape = tf.shape(lf)
    orgSize = [lfShape[0]//angularDim[0],lfShape[1]//angularDim[1]]
    if r < 0.25:
        pass
    elif r < 0.5:
        lf = tf.image.random_contrast(lf, 0.2, 0.5)
    elif r < 0.75:
        lf = tf.image.random_brightness(lf,.3)
    elif r < 1.:
        lf = tf.image.random_hue(lf,.3)
    lf = tf.transpose(tf.reshape(
        lf, [orgSize[0], angularDim[0], orgSize[1], angularDim[1], 3]),
                      perm=[0, 2, 1, 3, 4])
    lfList = []
    lf = tf.image.convert_image_dtype(lf,tf.float32)
    lf.set_shape([spatialDim[0],spatialDim[1],7,7,3])
    cList,lfList = lfRandomCrop(lf,CropNo,orgSize)
    return cList, lfList

def input_pipeline(lf_filenames,batch_size,prefetch_size,shuffle_buffer):
    lf_names = tf.data.Dataset.from_tensor_slices(lf_filenames).shuffle(shuffle_buffer)
    autotune=tf.data.experimental.AUTOTUNE
    ds = lf_names.map(lfReadAug, num_parallel_calls=4).unbatch()
    return ds.batch(batch_size, drop_remainder=True).prefetch(prefetch_size)


lfCropSize = [128,128,7,7]
angularDim=[7,7]
spatialDim=[128,128]
myDtype = tf.float32
myIntDtype = tf.int32
CropNo = 4