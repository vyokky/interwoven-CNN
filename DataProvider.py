import numpy as np
import math
import os


class Provider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values

f
    """   
    
    def __init__(self, input_size, output_size, prediction_gap, flatten=True, batchsize=-1, stride=(1, 1),
                 shuffle=True, pad=None, pad_value=0):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.prediction_gap = prediction_gap
        self.stride = stride
        self.shuffle = shuffle
        self.pad = pad
        self.pad_value = pad_value
        self.flatten = flatten

    def DataSlicer_3D(self, inputs, excerpt, flatten=False, external=None):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, used for LOP blended.

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        if self.pad is None:
            pad_x = 0
            pad_y = 0
        else:
            pad_x = self.pad[0]
            pad_y = self.pad[1]

        x_max, y_max, z_max = inputs.shape
        x_max += pad_x * 2
        y_max += pad_y * 2
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], self.output_size[2], total))
        if external:
            external_data = np.zeros((total, 1))

        x_offset = (self.input_size[0] - self.output_size[0]) / 2
        y_offset = (self.input_size[1] - self.output_size[1]) / 2

        data_num = 0

        for frame in xrange(len(excerpt)):

            if external:
                external_frame = inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap]

            if self.pad is None:
                input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
                target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap-1:
                excerpt[frame] + self.input_size[2] + self.prediction_gap + self.output_size[2]-1].reshape(self.output_size)

            else:
                input_frame = np.ones((x_max, y_max, self.input_size[2])) * self.pad_value
                target_frame = np.ones((x_max, y_max, self.output_size[2])) * self.pad_value

                input_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]

                target_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    inputs[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap-1: excerpt[frame] \
                    + self.input_size[2] + self.prediction_gap + self.output_size[2]-1]

                target_frame = target_frame.reshape(x_max, y_max, self.output_size[2])

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, :, data_num] = target_frame[
                                                  x - self.input_size[0] + x_offset:x - self.input_size[0] +
                                                                                    x_offset + self.output_size[0],
                                                  y - self.input_size[1] + y_offset:y - self.input_size[1] +
                                                                                    y_offset + self.output_size[1], :]
                    if external:
                        external_data[data_num] = external_frame[x - self.input_size[0] + x_offset,
                                                                 y - self.input_size[1] + y_offset]

                    data_num += 1

        if external:
            if self.shuffle:
                indices = np.random.permutation(total)
                if flatten:
                    return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            external_data[indices[0:self.batchsize]])
                return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        external_data[indices[0:self.batchsize]])

            else:
                if flatten:
                    return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                            np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)).flatten(),
                            external_data[0:self.batchsize])
                return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                        np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)),
                        external_data[0:self.batchsize])
        else:
            if self.shuffle:
                indices = np.random.permutation(total)
                if flatten:
                    return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))
                return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))

            else:
                if flatten:
                    return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                            np.transpose(target_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)).flatten())
                return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False
        
        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] - self.prediction_gap - self.output_size[2] + 2

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, flatten=self.flatten)
            if norm_tar:
                net_targets = ((net_targets-mean)/float(std)).reshape(self.batchsize, -1)

            yield (net_inputs-mean)/float(std), net_targets


class SuperResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle

    def DataSlicer_3D(self, inputs, excerpt):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], total))

        data_num = 0

        for frame in xrange(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, data_num] = target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, indices[0:self.batchsize]], (2, 0, 1)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, 0:self.batchsize], (2, 0, 1)).reshape(self.batchsize, -1))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets


class SpecialSuperResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle

    def special(self, inputs, keepdims=False):

        if not keepdims:
            output = np.zeros((20, 20, inputs.shape[-2], inputs.shape[-1]))
        else:
            output = np.zeros(inputs.shape)

        index_s = np.zeros((20, 20))
        index_l = np.zeros((80, 80))

        index10s = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 10), (0, 13), (0, 16), (0, 19), (3, 0), (3, 19), (6, 0),
                    (6, 19), (9, 0), (9, 19), (10, 0), (10, 19),
                    (13, 0), (13, 19), (16, 0), (16, 19), (19, 0), (19, 3), (19, 6), (19, 9), (19, 10), (19, 13),
                    (19, 16), (19, 19)]

        index10l = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (10, 0), (10, 70), (20, 0),
                    (20, 70), (30, 0), (30, 70), (40, 0), (40, 70),
                    (50, 0), (50, 70), (60, 0), (60, 70), (70, 0), (70, 10), (70, 20), (70, 30), (70, 40), (70, 50),
                    (70, 60), (70, 70)]

        index_s[3:17, 3:17] = 1
        index_l[26:54, 26:54] = 1

        for x in xrange(26, 54, 2):
            for y in xrange(26, 54, 2):
                small_x = (x - 20) / 2
                small_y = (y - 20) / 2
                if not keepdims:
                    output[small_x, small_y, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1))
                else:
                    output[x:x + 2, y:y + 2, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1), keepdims=True)

        for i in xrange(len(index10s)):
            large_x = index10l[i][0]
            large_y = index10l[i][1]
            if not keepdims:
                output[index10s[i][0], index10s[i][1], :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1))
            else:
                output[large_x:large_x + 10, large_y:large_y + 10, :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1), keepdims=True)

            index_l[large_x:large_x + 10, large_y:large_y + 10] = 1
            index_s[index10s[i]] = 1

        residx_s = np.array(np.where(index_s == 0))

        for i in xrange(176):
            select_s = residx_s[:, i]
            select_l = np.unravel_index(np.argmin(index_l), (80, 80))
            if not keepdims:
                output[select_s[0], select_s[1], :, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1))
            else:
                output[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1),
                    keepdims=True)
            index_l[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4] = 1

        return output

    def DataSlicer_3D(self, inputs, excerpt, keepdims):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], total))

        data_num = 0

        for frame in xrange(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, data_num] = target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(self.special(input_data[:, :, :, indices[0:self.batchsize]], keepdims=keepdims), (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, indices[0:self.batchsize]], (2, 0, 1)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(self.special(input_data[:, :, :, 0:self.batchsize], keepdims=keepdims), (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, 0:self.batchsize], (2, 0, 1)).reshape(self.batchsize, -1))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False, keepdims=False):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, keepdims=keepdims)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets


class RatioResolutionProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values


    """

    def __init__(self, input_size, output_size, batchsize=-1, stride=(1, 1),
                 shuffle=True):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.stride = stride
        self.shuffle = shuffle

    def ratio_smoother(self, data, lamda, stepx=4, stepy=4):

        data = data + lamda
        newdata = np.zeros((data.shape[0] / stepx, data.shape[1] / stepy))
        ratio_frame = np.zeros((data.shape[0] / stepx, data.shape[1] / stepy, stepx * stepy))
        for x in range(data.shape[0] / stepx):
            for y in range(data.shape[1] / stepy):
                newdata[x][y] = data[stepx * x:stepx * (x + 1), stepy * y:stepy * (y + 1)].sum()

        newdata = np.kron(newdata, np.ones((stepx, stepy)))
        ratio = data / newdata

        for x in range(0, stepx):
            for y in range(0, stepy):
                frame = y * stepx + x
                ratio_frame[:, :, frame] = ratio[x::stepx, y::stepy]

        return ratio_frame / np.sum(ratio_frame, axis=2, keepdims=True)

    def DataSlicer_3D(self, inputs, excerpt, stepx, stepy, lamda):
        """
        generate data from input frames
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: np.array (x*y*t), target from another resource, should has the shape (x'*y'*t)

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)
        external_data: np.array, with dim (batchsize*1)
        """

        x_max, y_max, z_max = inputs.shape
        channel = stepx * stepy
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0]/stepx, self.output_size[1]/stepx, channel, total))

        data_num = 0

        for frame in xrange(len(excerpt)):

            input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
            target_frame = inputs[:, :, excerpt[frame] + self.input_size[2] - 1]

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, :, data_num] = self.ratio_smoother(target_frame[x - self.input_size[0]:
                        x - self.input_size[0] + self.output_size[0], y - self.input_size[1]:
                            y - self.input_size[1] + self.output_size[1]], lamda, stepx, stepy)

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 0, 1, 2)).reshape(self.batchsize, -1))

        else:
            return (np.transpose(input_data[:, :, :, 0:self.batchsize], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, :, 0:self.batchsize], (3, 0, 1, 2)).reshape(self.batchsize, -1))

    def feed(self, inputs, framebatch, stepx, stepy, mean=0, std=1, norm_tar=False, lamda=0.0000001):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))

            net_inputs, net_targets = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, stepx=stepx, stepy=stepy,
                                                         lamda=lamda)
            if norm_tar:
                net_targets = ((net_targets - mean) / float(std))
            yield (net_inputs - mean) / float(std), net_targets


class DoubleSourceProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, batchsize, shuffle):

        self.batchsize = batchsize
        self.shuffle = shuffle

    def feed(self, inputs, targets):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        assert len(inputs) == len(targets)
        if self.batchsize < 0:
            self.batchsize = len(inputs)
        if self.shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]


class MoverProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, length):

        self.length = length

    def special(self, inputs, keepdims=False):

        if not keepdims:
            output = np.zeros((20, 20, inputs.shape[-2], inputs.shape[-1]))
        else:
            output = np.zeros(inputs.shape)

        index_s = np.zeros((20, 20))
        index_l = np.zeros((80, 80))

        index10s = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 10), (0, 13), (0, 16), (0, 19), (3, 0), (3, 19), (6, 0),
                    (6, 19), (9, 0), (9, 19), (10, 0), (10, 19),
                    (13, 0), (13, 19), (16, 0), (16, 19), (19, 0), (19, 3), (19, 6), (19, 9), (19, 10), (19, 13),
                    (19, 16), (19, 19)]

        index10l = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40), (0, 50), (0, 60), (0, 70), (10, 0), (10, 70), (20, 0),
                    (20, 70), (30, 0), (30, 70), (40, 0), (40, 70),
                    (50, 0), (50, 70), (60, 0), (60, 70), (70, 0), (70, 10), (70, 20), (70, 30), (70, 40), (70, 50),
                    (70, 60), (70, 70)]

        index_s[3:17, 3:17] = 1
        index_l[26:54, 26:54] = 1

        for x in xrange(26, 54, 2):
            for y in xrange(26, 54, 2):
                small_x = (x - 20) / 2
                small_y = (y - 20) / 2
                if not keepdims:
                    output[small_x, small_y, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1))
                else:
                    output[x:x + 2, y:y + 2, :, :] = np.mean(inputs[x:x + 2, y:y + 2, :, :], axis=(0, 1), keepdims=True)

        for i in xrange(len(index10s)):
            large_x = index10l[i][0]
            large_y = index10l[i][1]
            if not keepdims:
                output[index10s[i][0], index10s[i][1], :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1))
            else:
                output[large_x:large_x + 10, large_y:large_y + 10, :, :] = np.mean(
                    inputs[large_x:large_x + 10, large_y:large_y + 10, :, :], axis=(0, 1), keepdims=True)

            index_l[large_x:large_x + 10, large_y:large_y + 10] = 1
            index_s[index10s[i]] = 1

        residx_s = np.array(np.where(index_s == 0))

        for i in xrange(176):
            select_s = residx_s[:, i]
            select_l = np.unravel_index(np.argmin(index_l), (80, 80))
            if not keepdims:
                output[select_s[0], select_s[1], :, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1))
            else:
                output[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :] = np.mean(
                    inputs[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4, :, :], axis=(0, 1),
                    keepdims=True)
            index_l[select_l[0]:select_l[0] + 4, select_l[1]:select_l[1] + 4] = 1

        return output

    def feed(self, inputs, targets, special=False, keepdims=False):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        for start_idx in range(inputs.shape[-1] - self.length + 1):
            excerpt = range(start_idx, start_idx + self.length)
            if special:
                input_frame = self.special(np.expand_dims(inputs[:, :, excerpt], axis=-1), keepdims=keepdims)
                yield np.transpose(input_frame, (3, 2, 0, 1)), targets[:, :, excerpt[-1]]

            else:
                input_frame = np.transpose(inputs[:, :, excerpt], (2, 0, 1))
                yield np.expand_dims(input_frame, axis=0), targets[:, :, excerpt[-1]]


class MixSampleProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, batchsize, shuffle, sample):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.sample = sample

    def feed(self, inputs):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """
        size = inputs.shape[1]
        number = inputs.shape[0]
        sampler = np.random.uniform(size=(self.sample, size))
        targets = np.zeros((number + self.sample, 1))
        targets[:number] = 1
        inputs = np.concatenate([inputs, sampler], axis=0)
        if self.batchsize < 0:
            self.batchsize = len(inputs)
        if self.shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]


class DynamicSampleProvider(object):
    """
    provider for data in 2 files
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    """

    def __init__(self, batchsize, shuffle):

        self.batchsize = batchsize
        self.shuffle = shuffle

    def feed(self, inputs, sample_size):

        """
        generate data from input files
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """
        if inputs is None:

            yield np.random.normal(scale=1, size=(self.batchsize, sample_size))

        else:
            if self.shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
                yield inputs[indices[:self.batchsize]], np.random.normal(scale=1, size=(self.batchsize, sample_size))


class VideoProvider(object):
    """

    """


    def __init__(self, batchsize, shuffle, buffer_size):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.buffer_size = buffer_size


    def feed(self, input_dir, target_dir, frame_num, target_identifier='frame_label_', name_begin=2, std=255.0,
             crop=None, resize=(1, 1), resample=1, data_ratio=1):

        import os
        import glob
        import imageio
        import re
        import cv2

        video_list = np.array(os.listdir(input_dir))
        video_num = len(video_list)
        video_idx = np.arange(video_num)

        frame_nums = np.array([len(os.listdir(input_dir + '/' + video)) for video in video_list])
        # frame_nums -= frame_num*resample


        video_batch_num = video_num/int(self.buffer_size) + 1 if video_num % int(self.buffer_size) > 0 \
            else video_num/int(self.buffer_size)

        if self.shuffle:
            np.random.shuffle(video_idx)

        for batch_idx in xrange(video_batch_num):
            # print ('Training on video batch %s/%s...'%(batch_idx+1,video_batch_num))
            video_selected_idx = video_idx[batch_idx * self.buffer_size:(batch_idx + 1) * self.buffer_size]
            video_selected = video_list[video_selected_idx]
            video_selected_frame_num = frame_nums[video_selected_idx]
            if len(video_selected_idx)>1:
                # print video_selected_idx, video_selected_frame_num
                idx_maker = np.concatenate([np.vstack([np.ones(video_selected_frame_num[idx] - frame_num*resample) * idx,
                                                       np.arange(frame_num*resample, video_selected_frame_num[idx])])
                                            for idx in range(self.buffer_size)], axis=1).T
            else:
                idx_maker = np.vstack([np.zeros(video_selected_frame_num - frame_num*resample),
                                       np.arange(frame_num*resample, video_selected_frame_num)]).T

            full_frame_num = int(idx_maker.shape[0]*data_ratio)
            idx_maker = idx_maker[range(full_frame_num)]


            if self.shuffle:
                np.random.shuffle(idx_maker)

            video_buffer = []
            target_buffer = []

            for i, video in enumerate(video_selected):
                # print ('Loading batch %s video %s/%s...' % (batch_idx + 1, i+1, len(video_selected)))
                files = glob.glob('%s/%s/*.jpg'%(input_dir, video))
                files.sort(key=lambda ima: int(re.search('(.*).jpg', ima.split('_')[-1]).group(1)))
                # for ima in files:
                #     frames.append(imageio.imread(ima))
                if crop:
                    ima_w = imageio.imread(files[0]).shape[0]
                    ima_h = imageio.imread(files[0]).shape[1]
                    ima_crop_w = cv2.resize(imageio.imread(files[0])[crop[0]:ima_w-crop[1],crop[2]:ima_h-crop[3]],
                                                  None, fx=resize[0], fy=resize[1]).shape[0]
                    ima_crop_h = cv2.resize(imageio.imread(files[0])[crop[0]:ima_w-crop[1],crop[2]:ima_h-crop[3]],
                                                  None, fx=resize[0], fy=resize[1]).shape[1]
                    frames = np.zeros((video_selected_frame_num[i], ima_crop_w, ima_crop_h, 3))
                    for j, ima in enumerate(files):
                        frames[j] = cv2.resize(imageio.imread(ima)[crop[0]:ima_w-crop[1],crop[2]:ima_h-crop[3]], None,
                                               fx=resize[0], fy=resize[1]) / std
                    # frames = np.stack([cv2.resize(imageio.imread(ima)
                    #                               [crop[0]:imageio.imread(ima).shape[0]-crop[1],crop[2]:imageio.imread(ima).shape[1]-crop[3]],
                    #                               None, fx=resize[0], fy=resize[1]) / std for ima in files])
                else:
                    frames = np.stack(
                        [cv2.resize(imageio.imread(ima), None, fx=resize[0], fy=resize[1]) / std for ima in files])

                video_buffer.append(frames)
                target_buffer.append(np.load(target_dir + target_identifier + video[name_begin:] + '.npy'))
                batch_num = full_frame_num/int(self.batchsize) + 1 if full_frame_num % int(self.batchsize) > 0 \
                                    else full_frame_num/int(self.batchsize)

            for batch in xrange(batch_num):
                batch_idx = np.arange(batch*self.batchsize, min((batch+1)*self.batchsize, full_frame_num))
                data_list = []
                label_list = []
                for data_idx in batch_idx:
                    # print range(int(idx_maker[data_idx][1]) - frame_num*resample,int(idx_maker[data_idx][1]),resample)
                    # print idx_maker.shape
                    # print len(video_buffer)
                    data_list.append(video_buffer[int(idx_maker[data_idx][0])]
                                     [int(idx_maker[data_idx][1]) - frame_num*resample:int(idx_maker[data_idx][1]):resample])
                    label_list.append(int(target_buffer[int(idx_maker[data_idx][0])][int(idx_maker[data_idx][1])-1]))
                yield np.stack(data_list), np.array(label_list)


class MultiVideoProvider(object):
    """

    """


    def __init__(self, batchsize, shuffle, buffer_size):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.buffer_size = buffer_size


    def feed(self, input_dir, input_dir2, target_dir, frame_num, target_identifier='frame_label_', name_begin=2, std=255.0,
             crop=None, resize=(1, 1), crop2=None, resize2=(1, 1), resample=1, data_ratio=1):

        import os
        import glob
        import imageio
        import re
        import cv2

        video_list = np.array(os.listdir(input_dir))
        # video_list2 = np.array(os.listdir(input_dir2))
        video_num = len(video_list)
        video_idx = np.arange(video_num)

        frame_nums = np.array([len(os.listdir(input_dir + '/' + video)) for video in video_list])
        frame_nums2 = np.array([len(os.listdir(input_dir2 + '/1_' + video[2:])) for video in video_list])
        # frame_nums -= frame_num*resample


        video_batch_num = video_num/int(self.buffer_size) + 1 if video_num % int(self.buffer_size) > 0 \
            else video_num/int(self.buffer_size)

        if self.shuffle:
            np.random.shuffle(video_idx)

        for batch_idx in xrange(video_batch_num):
            # print ('Training on video batch %s/%s...'%(batch_idx+1,video_batch_num))
            video_selected_idx = video_idx[batch_idx * self.buffer_size:(batch_idx + 1) * self.buffer_size]
            video_selected = video_list[video_selected_idx]
            video_selected_frame_num = frame_nums[video_selected_idx]
            video_selected_frame_num2 = frame_nums2[video_selected_idx]
            if len(video_selected_idx)>1:
                idx_maker = np.concatenate([np.vstack([np.ones(video_selected_frame_num[idx] - frame_num*resample) * idx,
                                                       np.arange(frame_num*resample, video_selected_frame_num[idx])])
                                            for idx in range(self.buffer_size)], axis=1).T
            else:
                idx_maker = np.vstack([np.zeros(video_selected_frame_num - frame_num*resample),
                                       np.arange(frame_num*resample, video_selected_frame_num)]).T

            full_frame_num = int(idx_maker.shape[0]*data_ratio)
            idx_maker = idx_maker[range(full_frame_num)]


            if self.shuffle:
                np.random.shuffle(idx_maker)

            video_buffer = []
            video_buffer2 = []
            target_buffer = []

            for i, video in enumerate(video_selected):
                # print ('Loading batch %s video %s/%s...' % (batch_idx + 1, i+1, len(video_selected)))
                files = glob.glob('%s/%s/*.jpg'%(input_dir, video))
                files.sort(key=lambda ima: int(re.search('(.*).jpg', ima.split('_')[-1]).group(1)))

                files2 = glob.glob('%s/%s/*.jpg' % (input_dir2, '1_'+video[2:]))
                files2.sort(key=lambda ima: int(re.search('(.*).jpg', ima.split('_')[-1]).group(1)))
                # for ima in files:
                #     frames.append(imageio.imread(ima))
                if crop:
                    # frames = np.stack([cv2.resize(imageio.imread(ima)
                    #                               [crop[0]:imageio.imread(ima).shape[0]-crop[1],crop[2]:imageio.imread(ima).shape[1]-crop[3]],
                    #                               None, fx=resize[0], fy=resize[1]) / std for ima in files])
                    # frames2 = np.stack([cv2.resize(imageio.imread(ima)
                    #                               [crop2[0]:imageio.imread(ima).shape[0]-crop2[1],crop2[2]:imageio.imread(ima).shape[1]-crop2[3]],
                    #                               None, fx=resize2[0], fy=resize2[1]) / std for ima in files2])

                    ima_w = imageio.imread(files[0]).shape[0]
                    ima_h = imageio.imread(files[0]).shape[1]
                    ima_crop_w = cv2.resize(imageio.imread(files[0])[crop[0]:ima_w - crop[1], crop[2]:ima_h - crop[3]],
                                            None, fx=resize[0], fy=resize[1]).shape[0]
                    ima_crop_h = cv2.resize(imageio.imread(files[0])[crop[0]:ima_w - crop[1], crop[2]:ima_h - crop[3]],
                                            None, fx=resize[0], fy=resize[1]).shape[1]

                    ima_w2 = imageio.imread(files2[0]).shape[0]
                    ima_h2 = imageio.imread(files2[0]).shape[1]
                    ima_crop_w2 = cv2.resize(imageio.imread(files2[0])[crop2[0]:ima_w2 - crop2[1], crop2[2]:ima_h2 - crop2[3]],
                                            None, fx=resize2[0], fy=resize2[1]).shape2[0]
                    ima_crop_h2 = cv2.resize(imageio.imread(files2[0])[crop2[0]:ima_w2 - crop2[1], crop2[2]:ima_h2 - crop2[3]],
                                            None, fx=resize2[0], fy=resize2[1]).shape2[1]
                    frames = np.zeros((video_selected_frame_num[i], ima_crop_w, ima_crop_h, 3))
                    frames2 = np.zeros((video_selected_frame_num2[i], ima_crop_w2, ima_crop_h2, 3))
                    for j, ima in enumerate(files):
                        frames[j] = cv2.resize(imageio.imread(ima)[crop[0]:ima_w - crop[1], crop[2]:ima_h - crop[3]],
                                               None, fx=resize[0], fy=resize[1]) / std
                    for k, ima in enumerate(files2):
                        frames2[k] = cv2.resize(imageio.imread(ima)[crop2[0]:ima_w2 - crop2[1], crop[2]:ima_h2 - crop2[3]],
                                               None, fx=resize2[0], fy=resize2[1]) / std
                else:
                    frames = np.stack(
                        [cv2.resize(imageio.imread(ima), None, fx=resize[0], fy=resize[1]) / std for ima in files])
                    frames2 = np.stack(
                        [cv2.resize(imageio.imread(ima), None, fx=resize2[0], fy=resize[1]) / std for ima in files2])

                video_buffer.append(frames)
                video_buffer2.append(frames2)
                target_buffer.append(np.load(target_dir + target_identifier + video[name_begin:] + '.npy'))
                batch_num = full_frame_num/int(self.batchsize) + 1 if full_frame_num % int(self.batchsize) > 0 \
                                    else full_frame_num/int(self.batchsize)

            for batch in xrange(batch_num):
                batch_idx = np.arange(batch*self.batchsize, (batch+1)*self.batchsize)
                data_list = []
                data_list2 = []
                label_list = []
                for data_idx in batch_idx:
                    video2_frame_idx = np.clip(int(video_selected_frame_num2[idx_maker[data_idx][0]] *
                                                   idx_maker[data_idx][0]/video_selected_frame_num[idx_maker[data_idx][0]]),
                                               frame_num*resample, video_selected_frame_num2[idx_maker[data_idx][0]])
                    data_list.append(video_buffer[int(idx_maker[data_idx][0])]
                                     [int(idx_maker[data_idx][1]) - frame_num*resample:int(idx_maker[data_idx][1]):resample])
                    data_list2.append(video_buffer2[int(idx_maker[data_idx][0])]
                                     [video2_frame_idx - frame_num * resample:video2_frame_idx:resample])
                    label_list.append(int(target_buffer[int(idx_maker[data_idx][0])][int(idx_maker[data_idx][1])-1]))
                yield np.stack(data_list), np.stack(data_list2), np.array(label_list)


class DoubleSourceSlider(object):

    """
    provider for data in 2 files (slicer version)
    ------------------------------------------------------
    :arg
    batchsize: int, the size of batch, default -1 (take all data generated)
    shuffle: bool, default True. shuffle the data or not
    offset: int, the distance between initial index and target
    """

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets):

        """
        generate data from input files (slicer version)
        ------------------------------------------------
        :arg
        inputs: np.array, the source input data
        targets: np.array, the source target data

        :return
        input: np.array
        target: np.array
        """

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size
        max_batchsize = inputs.size - 2 * self.offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]

            yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                  targets[excerpt + self.offset]


class SimpleProvider(object):
    def __init__(self, batchsize, shuffle):

        self.batchsize = batchsize
        self.shuffle = shuffle

    def feed(self, inputs, targets, flatten=True):

        # inputs, targets = inputs.flatten(), targets.flatten()
        # assert inputs.size == targets.size

        max_batchsize = inputs.shape[0]

        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize, dtype='int')
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            # print excerpt
            if flatten:
                yield np.array(inputs[excerpt]), \
                      np.array(targets[excerpt]).reshape(excerpt.size, -1)
            else:
                yield np.array(inputs[excerpt]), \
                      np.array(targets[excerpt])

class TimeProvider(object):
    def __init__(self, batchsize, shuffle, past, future):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.past = past
        self.future = future

    def feed(self, inputs, targets, flatten=True):

        # inputs, targets = inputs.flatten(), targets.flatten()
        # assert inputs.size == targets.size

        max_batchsize = inputs.shape[0]

        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(self.past, max_batchsize - self.future, dtype='int')
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            data_batch = []
            for idx in excerpt:
                data_batch.append(np.expand_dims(inputs[idx - self.past: idx + self.future + 1], 0))
            # print excerpt
            if flatten:
                yield np.concatenate(data_batch, axis = 0), \
                      np.array(targets[excerpt]).reshape(excerpt.size, -1)
            else:
                yield np.concatenate(data_batch, axis = 0), \
                      np.array(targets[excerpt])


class DoubleExpertProvider(object):
    """
    provider for mobile traffic data (or other 3D data)
    ------------------------------------------------------
    :arg
    input_size: 3-element tuple(x*y*feature map), the shape of input we want
    output_size: 2-element tuple (x*y), the shape of output we want
    prediction_gap: int, the distant between the last input frame and output frame
    flatten: bool, whether flatten the output or not
    batchsize: int, the size of batch, default -1 (take all data generated)
    stride: 2-element tuple, the stride when selecting data
    shuffle: bool, default True. shuffle the data or not
    pad: 2-element or None, the size of padding, default None
    pad_value: float, padding values

    """

    def __init__(self, input_size, output_size, prediction_gap, flatten=True, batchsize=-1, stride=(1, 1),
                 shuffle=True, pad=None, pad_value=0):

        self.input_size = input_size
        self.output_size = output_size
        self.batchsize = batchsize
        self.prediction_gap = prediction_gap
        self.stride = stride
        self.shuffle = shuffle
        self.pad = pad
        self.pad_value = pad_value
        self.flatten = flatten

    def DataSlicer_3D(self, inputs, external, excerpt, flatten=False):
        """
        generate data from input
        ------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        excerpt: list, the index of start frame of inputs
        flatten: bool, flatten target
        external: target file from different files

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y)

        """

        if self.pad is None:
            pad_x = 0
            pad_y = 0
        else:
            pad_x = self.pad[0]
            pad_y = self.pad[1]

        x_max, y_max, z_max = inputs.shape
        x_max += pad_x * 2
        y_max += pad_y * 2
        x_num = int(math.ceil((x_max - self.input_size[0] + 1.0) / self.stride[0]))
        y_num = int(math.ceil((y_max - self.input_size[1] + 1.0) / self.stride[1]))
        total = x_num * y_num * len(excerpt)
        if self.batchsize <= 0:
            self.batchsize = total
        input_data = np.zeros((self.input_size[0], self.input_size[1], self.input_size[2], total))

        target_data = np.zeros((self.output_size[0], self.output_size[1], self.output_size[2], total))

        x_offset = (self.input_size[0] - self.output_size[0]) / 2
        y_offset = (self.input_size[1] - self.output_size[1]) / 2

        data_num = 0

        for frame in xrange(len(excerpt)):

            if self.pad is None:
                input_frame = inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]
                target_frame = external[:, :, excerpt[frame] + self.input_size[2] + self.prediction_gap-1:
                excerpt[frame] + self.input_size[2] + self.prediction_gap + self.output_size[2]-1].reshape(
                    self.output_size)

            else:
                input_frame = np.ones((x_max, y_max, self.input_size[2])) * self.pad_value
                target_frame = np.ones((x_max, y_max, self.output_size[2])) * self.pad_value
                input_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    inputs[:, :, excerpt[frame]:excerpt[frame] + self.input_size[2]]

                target_frame[pad_x:pad_x + inputs.shape[0], pad_y:pad_y + inputs.shape[1], :] = \
                    external[:, :, excerpt[frame] + self.input_size[2] +
                                 self.prediction_gap-1: excerpt[frame] +
                                                      self.input_size[2] + self.prediction_gap + self.output_size[2]-1]

                target_frame = target_frame.reshape(x_max, y_max, self.output_size[2])

            for x in xrange(self.input_size[0], x_max + 1, self.stride[0]):
                for y in xrange(self.input_size[1], y_max + 1, self.stride[1]):
                    input_data[:, :, :, data_num] = input_frame[x - self.input_size[0]:x, y - self.input_size[1]:y, :]
                    target_data[:, :, :, data_num] = target_frame[
                                                     x - self.input_size[0] + x_offset:x - self.input_size[0] +
                                                                                       x_offset + self.output_size[0],
                                                     y - self.input_size[1] + y_offset:y - self.input_size[1] +
                                                                                       y_offset + self.output_size[1], :]

                    data_num += 1

        if self.shuffle:
            indices = np.random.permutation(total)
            if flatten:
                return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                        np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))
            return (np.transpose(input_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)),
                    np.transpose(target_data[:, :, :, indices[0:self.batchsize]], (3, 2, 0, 1)))

        else:
            if flatten:
                return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                        np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)).flatten())
            return (np.transpose(input_data[0:self.batchsize], (3, 2, 0, 1)),
                    np.transpose(target_data[0:self.batchsize], (3, 2, 0, 1)))

    def feed(self, inputs, framebatch, mean=0, std=1, norm_tar=False, external=None):
        """
        iterate over mini-batch
        --------------------------------------------------------------------------------
        :arg
        inputs: np.array (x*y*t), the source data generated from
        framebatch: int, maximum frames we selected in one mini-batch
        mean: float, inputs normalized constant, mean
        std: float, inputs normalized constant, standard error
        norm_tar: bool, target normalized as input, default False

        :return
        input: np.array, with dim (batchsize*feature map*x*y)
        target: np.array, with dim (batchsize*x*y) or flatten one
        """

        frame_max = inputs.shape[2] - self.input_size[2] - self.prediction_gap - self.output_size[2] + 1

        if self.shuffle:
            indices = np.random.permutation(frame_max)

        for start_idx in range(0, frame_max, framebatch):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + framebatch]
            else:
                excerpt = range(start_idx, min((start_idx + framebatch), frame_max))
            batch = self.DataSlicer_3D(inputs=inputs, excerpt=excerpt, flatten=self.flatten,
                                       external=external)
            if len(batch) == 2:
                net_inputs, net_targets = batch
                if norm_tar:
                    net_targets = ((net_targets - mean) / float(std)).reshape(self.batchsize, -1)
                yield (net_inputs - mean) / float(std), net_targets

            elif len(batch) == 3:
                net_inputs, net_targets, net_external = batch
                if norm_tar:
                    net_targets = ((net_targets - mean) / float(std)).reshape(self.batchsize, -1)
                    net_external = ((net_external - mean) / float(std)).reshape(self.batchsize, -1)
                yield (net_inputs - mean) / float(std), net_targets, net_external


class Transformer(object):
    
    def __init__(self, mu, norm):
        
        self.mu = mu
        self.norm = norm

    def MuLawQuantisation(self, data, quantization=True):
        """
        Perform the mu-law transformation
        ------------------------------------------
        :arg
        data: data that needs to be transform
        mu: scale
        norm: normalisation constant
        quantization: quantize to integral, default True

        :return
        The transformed data

        """ 
        data = data.flatten()
        data = data/self.norm

        mu_law = np.sign(data)*(np.log(1+self.mu*np.abs(data))/np.log(1+self.mu))*self.mu

        if quantization:
            return np.round(mu_law)
        else:
            return mu_law

    def InverseMuLaw(self, data, sample=False):
        """
        Perform the inverse mu-law transformation
        --------------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant

        :return
        The inverse transformed data
        """     
        if sample:        
            means = data.flatten()
            cov = np.eye(data.size)*sample        
            data = np.random.multivariate_normal(means, cov)

        self.mu = float(self.mu)
        data /= self.mu

        recover = np.sign(data)*(1/self.mu)*((1+self.mu)**np.abs(data)-1)

        return recover*self.norm
    
    def LinearQuantisation(self, data, quantization=True):
        """
        Perform the linear quantisation.
        --------------------------------------
        :arg
        data: data that needs to be transform
        mu: scale
        norm: normalisation constant
        quantization: quantize to integral, default True

        :return
        The transformed data

        """ 
        data = data.flatten()
        gap = int(self.norm/self.mu)

        if quantization:
            return np.round(data/gap)
        else:
            return data/gap
        
    def InverseLinear(self, data):
        """
        Perform the inverse linear quantisation.
        -------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant

        :return
        The transformed data
        """     
        return data*int(self.norm/self.mu)
    
    def Normalise(self, data):
        """
        Perform the normalisation (data-mu)/norm.
        --------------------------------------
        :arg
        data: data that needs to be transformed
        mu: scale
        norm: normlisation constant

        :return
        The normalized data

        """     
        return (data-self.mu)/self.norm
    
    def InverseNormalise(self, data):
        """
        Perform the in-normalisation data*norm+mu.
        ------------------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant
        :return
        The in-normalized data

        """  
        
        return data*self.norm+self.mu



