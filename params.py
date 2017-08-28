from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024

input_size = (2048, 1280)

max_epochs = 50
batch_size = 2

orig_width = 1918
orig_height = 1280

threshold = 0.5

model = get_unet_1024((input_size[1], input_size[0], 3))
