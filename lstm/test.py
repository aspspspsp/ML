

def load_mnist():
    mnist = skdata.mnist.dataset.MNIST()
    mnist.meta  # trigger download if needed.
    def arr(n, dtype):
        # convert an array to the proper shape and dtype
        arr = mnist.arrays[n]
        return arr.reshape((len(arr), -1)).astype(dtype)
    train_images = arr('train_images', 'f') / 255.
    train_labels = arr('train_labels', np.uint8)
    test_images = arr('test_images', 'f') / 255.
    test_labels = arr('test_labels', np.uint8)
    return ((train_images[:50000], train_labels[:50000, 0]),
            (train_images[50000:], train_labels[50000:, 0]),
            (test_images, test_labels[:, 0]))