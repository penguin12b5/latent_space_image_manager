import imagemanager as im

#/Users/ot/.pyenv/versions/3.10.16/

lim = im.LatentImageModifier("images/*.png")
lim.trim_images()
lim.encode_images()
final_image = lim.decode_image(lim.modify("average"))

ip = im.ImagePlotter(len(lim.images) + 1, 2, 16, 8 * (len(lim.images) + 1))

for i, image in enumerate(lim.images):
    ip.plot_image(image, 2 * i + 1, f'Image {i + 1}')
    ip.plot_image(lim.latent_images[i][0, 0].cpu().numpy(), 2 * i + 2, f'Latent {i + 1}', cmap='viridis')

ip.display_plot()
