##### Main app
def main():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    import recognize_handwriting_digits as network
    net_structure = [784, 30, 20, 10]
    epoch = 30
    mini_batch_size = 10
    eta = 3.0

    net = network.Network(net_structure)
    net.SGD(training_data, epoch, mini_batch_size, eta, test_data=test_data)

if __name__ == "__main__":
    main()


